import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.stats import ks_2samp
from collections import Counter
import re
import warnings
import argparse
import json
import os

warnings.filterwarnings("ignore")

class ReviewEvaluator:
    def __init__(self, data_path, model_name="Synthetic"):
        self.model_name = model_name
        self.data_path = data_path
        
        # Determine file type and load
        if data_path.endswith('.jsonl'):
            self.df = pd.read_json(data_path, lines=True)
        else:
            self.df = pd.read_csv(data_path)
            
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.clean_df = pd.DataFrame()
        
        # 1. Base Metrics
        self.df['word_count'] = self.df['review_text'].apply(lambda x: len(str(x).split()))
        self.df['sentiment'] = self.df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        # 2. Efficiency Metrics (Calculated if time data exists)
        if 'generation_time_sec' in self.df.columns:
            self.df['time_per_word'] = self.df['generation_time_sec'] / self.df['word_count']
            self.df['estimated_cost'] = 0.0005  # Example heuristic cost

    def get_embeddings(self):
        if self.embeddings is None:
            print(f"[{self.model_name}] Computing embeddings...")
            self.embeddings = self.embedding_model.encode(self.df['review_text'].tolist(), show_progress_bar=True)
        return self.embeddings

    def get_top_words(self, n=10):
        # Use sklearn's built-in stop words set
        stopwords = ENGLISH_STOP_WORDS
        text = " ".join(self.df['review_text'].astype(str).str.lower())
        
        # Regex to find words with 4+ characters
        words = re.findall(r'\b\w{4,}\b', text) 
        meaningful_words = [w for w in words if w not in stopwords]
        
        return Counter(meaningful_words).most_common(n)

    # ==========================================================================
    # GUARDRAILS
    # ==========================================================================
    def check_quality_heuristics(self):
        """Checks for reviews that are too short or contain repetitive stuttering."""
        # Check 1: Length (< 4 words)
        self.df['is_too_short'] = self.df['word_count'] < 4
        
        # Check 2: Repetitive Patterns (e.g., 'good good good')
        pattern = re.compile(r'\b(\w+)( \1\b)+')
        self.df['has_repetition'] = self.df['review_text'].apply(lambda x: bool(pattern.search(str(x))))
        
        self.df['is_low_quality'] = self.df['is_too_short'] | self.df['has_repetition']

    def check_domain_semantic(self, real_evaluator, threshold_std=2.0):
        real_embs = real_evaluator.get_embeddings()
        domain_centroid = np.mean(real_embs, axis=0).reshape(1, -1)
        
        real_sims = cosine_similarity(real_embs, domain_centroid)
        min_acceptable_sim = np.mean(real_sims) - (threshold_std * np.std(real_sims))
        
        synth_embs = self.get_embeddings()
        self.df['domain_similarity'] = cosine_similarity(synth_embs, domain_centroid)
        self.df['is_off_topic'] = self.df['domain_similarity'] < min_acceptable_sim

    def check_diversity(self, threshold=0.85):
        embs = self.get_embeddings()
        sim_matrix = cosine_similarity(embs)
        np.fill_diagonal(sim_matrix, 0)
        self.df['is_duplicate'] = sim_matrix.max(axis=1) > threshold

    def check_sentiment_alignment(self):
        self.df['sentiment_mismatch'] = (
            ((self.df['rating'] <= 2) & (self.df['sentiment'] > 0.3)) |
            ((self.df['rating'] >= 5) & (self.df['sentiment'] < -0.1))
        )

    def run_filters(self, real_evaluator=None):
        self.check_diversity()
        self.check_sentiment_alignment()
        self.check_quality_heuristics()
        
        if real_evaluator:
            self.check_domain_semantic(real_evaluator)
        
        flags = ['is_duplicate', 'sentiment_mismatch', 'is_off_topic', 'is_low_quality']
        for f in flags:
            if f not in self.df.columns: self.df[f] = False
        
        self.df['REJECT'] = self.df[flags].any(axis=1)
        self.clean_df = self.df[~self.df['REJECT']].copy()

class ComparativeDashboard:
    def __init__(self, evaluators, real_evaluator=None, target_mean=None, output_img="comparison.png", output_report="report.md"):
        self.evals = evaluators
        self.real_evaluator = real_evaluator
        self.target_mean = target_mean
        self.output_img = output_img
        self.output_report = output_report
        sns.set_theme(style="whitegrid")

    def plot_comparisons(self):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # 1. RATING DISTRIBUTION
        rating_list = []
        for ev in self.evals:
            counts = ev.df['rating'].value_counts(normalize=True).mul(100).reset_index()
            counts.columns = ['Rating', 'Percentage']
            counts['Source'] = ev.model_name
            rating_list.append(counts)
        
        pct_rating_df = pd.concat(rating_list)
        sns.barplot(data=pct_rating_df, x='Rating', y='Percentage', hue='Source', ax=axes[0,0])
        axes[0,0].set_title("Rating Distribution (% of Dataset)")
        axes[0,0].set_ylabel("Percentage (%)")

        # 2. TOP KEYWORDS
        word_data = []
        for ev in self.evals:
            top_words = ev.get_top_words(10)
            for word, count in top_words:
                word_data.append({"Source": ev.model_name, "Word": word, "Freq": count})
        sns.barplot(data=pd.DataFrame(word_data), x="Freq", y="Word", hue="Source", ax=axes[0,1])
        axes[0,1].set_title("Top 10 Domain Keywords")

        # 3. WORD COUNT 
        for ev in self.evals:
            sns.kdeplot(ev.df['word_count'], label=ev.model_name, ax=axes[1,0], fill=True, alpha=0.2, common_norm=False)
        axes[1,0].set_title("Review Length Distribution (Normalized)")
        axes[1,0].legend()

        # 4. GENERATION EFFICIENCY
        eff_list = [ev.df[['time_per_word']].assign(Model=ev.model_name) for ev in self.evals if 'time_per_word' in ev.df.columns]
        if eff_list:
            sns.boxplot(data=pd.concat(eff_list), x='Model', y='time_per_word', ax=axes[1,1])
            axes[1,1].set_title("Efficiency: Seconds per Word")
        
        plt.tight_layout()
        plt.savefig(self.output_img)
        print(f"Charts saved to {self.output_img}")

    def generate_report_markdown(self):
        with open(self.output_report, "w") as f:
            f.write("# 📊 Comprehensive Comparison Report\n\n")
            
            if self.target_mean:
                f.write(f"**Target Rating Mean (Config):** {self.target_mean}\n\n")
            
            for ev in self.evals:
                f.write(f"## {ev.model_name}\n")
                
                # Basic Stats
                mean_rating = ev.df['rating'].mean()
                f.write(f"- **Total Samples:** {len(ev.df)}\n")
                f.write(f"- **Mean Rating:** {mean_rating:.2f}\n")
                f.write(f"- **Mean Word Count:** {ev.df['word_count'].mean():.1f}\n")
                
                # Distribution Bias Warning (Using Config Target Mean)
                if self.target_mean and ev.model_name != "Real Data":
                    drift = abs(mean_rating - self.target_mean)
                    if drift > 0.5:
                        f.write(f"  - ⚠️ **WARNING:** Significant rating drift from Target ({self.target_mean}) detected (Diff: {drift:.2f})\n")
                
                # KS Test (Statistical Distance vs Real Data)
                if self.real_evaluator and ev.model_name != self.real_evaluator.model_name:
                    ks_stat, p_val = ks_2samp(ev.df['rating'], self.real_evaluator.df['rating'])
                    f.write(f"- **Distribution Alignment to Real Data (KS Test):** Stat={ks_stat:.4f}, p-val={p_val:.4f}\n")
                    if p_val < 0.05:
                        f.write("  - ⚠️ Distributions are statistically DIFFERENT.\n")

                # Efficiency & Cost
                if 'generation_time_sec' in ev.df.columns:
                    f.write(f"- **Avg Generation Time:** {ev.df['generation_time_sec'].mean():.3f}s\n")
                    if 'time_per_word' in ev.df.columns:
                        f.write(f"- **Avg Time Per Word:** {ev.df['time_per_word'].mean():.4f}s/word\n")
                    # Total Cost
                    if 'estimated_cost' in ev.df.columns:
                        total_cost = ev.df['estimated_cost'].sum()
                        f.write(f"- **Est. Total Cost:** ${total_cost:.2f}\n")

                # Rejection Stats (Safe Access)
                if 'REJECT' in ev.df.columns:
                    f.write(f"- **Guardrail Rejection Rate:** {ev.df['REJECT'].mean()*100:.1f}%\n")
                    if 'is_duplicate' in ev.df.columns:
                        f.write(f"  - *Duplicates:* {ev.df['is_duplicate'].sum()}\n")
                    if 'is_off_topic' in ev.df.columns:
                        f.write(f"  - *Off-Topic:* {ev.df['is_off_topic'].sum()}\n")
                    if 'sentiment_mismatch' in ev.df.columns:
                        f.write(f"  - *Sentiment Mismatch:* {ev.df['sentiment_mismatch'].sum()}\n")
                    if 'is_low_quality' in ev.df.columns:
                        f.write(f"  - *Low Quality (Spam/Short):* {ev.df['is_low_quality'].sum()}\n")
                else:
                    f.write(f"- **Guardrail Rejection Rate:** N/A (Baseline/Not Run)\n")
                
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Compare Synthetic Data Quality")
    
    # Inputs
    parser.add_argument("--synthetic", nargs='+', required=True, help="Path(s) to synthetic data file(s) (csv or jsonl)")
    parser.add_argument("--real", required=True, help="Path to real/baseline data file (csv)")
    parser.add_argument("--config", required=True, help="Path to input_config.json containing target parameters")
    
    # Outputs
    parser.add_argument("--output-report", default="Detailed_Quality_Report.md", help="Output filename for Markdown report")
    parser.add_argument("--output-img", default="comparison_report.png", help="Output filename for comparison charts")

    args = parser.parse_args()

    # 1. Load Config to get Target Mean
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        target_mean = config.get('sampling_parameters', {}).get('rating_distribution', {}).get('mean', 3.5)
        print(f"Loaded Target Mean from config: {target_mean}")
    except Exception as e:
        print(f"Error loading config: {e}. Defaulting target mean to 3.5")
        target_mean = 3.5

    # 2. Load Evaluators
    print("Loading Real Data...")
    real_data = ReviewEvaluator(args.real, model_name="Real Data")
    
    evaluators = []
    for file_path in args.synthetic:
        # Create a readable model name from the filename
        # e.g., 'synthetic_reviews.jsonl' -> 'synthetic_reviews'
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Processing {model_name}...")
        ev = ReviewEvaluator(file_path, model_name=model_name)
        ev.run_filters(real_evaluator=real_data)
        
        # Save cleaned file
        clean_out = f"cleaned_{model_name}.jsonl"
        ev.clean_df.to_json(clean_out, orient='records', lines=True)
        print(f"  Saved cleaned data to {clean_out}")
        
        evaluators.append(ev)

    # 3. Generate Dashboard
    dashboard = ComparativeDashboard(
        evaluators + [real_data], 
        real_evaluator=real_data, 
        target_mean=target_mean,
        output_img=args.output_img,
        output_report=args.output_report
    )
    
    dashboard.plot_comparisons()
    dashboard.generate_report_markdown()
    
    print(f"Pipeline Complete. Report saved to {args.output_report}")