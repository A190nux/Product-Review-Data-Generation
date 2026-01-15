import json
import random
import numpy as np
import pandas as pd
import dspy
import os
import time
import argparse


# ==============================================================================
# 1. DSPy CONFIGURATION
# ==============================================================================
def configure_lm(model_choice):
    """
    Configures the DSPy language model based on CLI selection.
    """
    if model_choice == "ministral":
        print("Configuring for Ministral-8B...")
        lm = dspy.LM(
            'ollama_chat/ministral-3:8b', 
            api_base='http://192.168.1.101:11434',
            cache=False,
            num_ctx=2048,
            temperature=0.7
        )
    elif model_choice == "rwkv":
        print("Configuring for RWKV-7...")
        lm = dspy.LM(
            'openai/rwkv-7',
            api_base='http://192.168.1.101:8000/v1',
            api_key='sk-any-key',
            cache=False,
            temperature=0.7,
            max_tokens=2048
        )
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
    
    dspy.configure(lm=lm)

# ==============================================================================
# 2. DSPy SIGNATURE
# ==============================================================================
class GenerateReview(dspy.Signature):
    """
    Generate a realistic synthetic product review for Canva based on a specific persona 
    and product context. The review should strictly adhere to the persona's voice, 
    writing style, and the assigned star rating.
    """
    
    product_context = dspy.InputField(
        desc="Key details about the product (UI elements, technical constraints, vibe)."
    )
    persona_profile = dspy.InputField(
        desc="The specific user persona (Role, Skill Level, Platform) and their writing style."
    )
    review_scenario = dspy.InputField(
        desc="The specific topic/feature to discuss, the sentiment/emotion, and whether to focus on a Pro or Con."
    )
    star_rating = dspy.InputField(
        desc="The integer rating (1-5) this review must justify."
    )
    
    review_text = dspy.OutputField(
        desc="The generated review content. Do not include the rating in the text, just the body."
    )

# ==============================================================================
# 3. GENERATOR CLASS
# ==============================================================================
class SyntheticReviewGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.predictor = dspy.Predict(GenerateReview)
        
    def _sample_rating(self):
        """
        Samples a rating from a Gaussian distribution, clipped to 1-5.
        """
        params = self.config['sampling_parameters']['rating_distribution']
        rating = np.random.normal(params['mean'], params['std_dev'])
        rating = np.clip(round(rating), params['min'], params['max'])
        return int(rating)

    def _build_persona(self):
        """
        Randomly samples attributes from persona buckets to create a unique user.
        """
        buckets = self.config['persona_buckets']
        return {
            "role": random.choice(buckets['roles']),
            "skill": random.choice(buckets['skill_levels']),
            "style": random.choice(buckets['writing_styles']),
            "platform": random.choice(buckets['platforms'])
        }

    def _get_scenario_and_topic(self, rating):
        """
        Selects a topic and emotion based on the rating to ensure logical consistency.
        (e.g., Low rating -> Select 'Cons' and 'Frustrated' emotion).
        """
        # 1. Pick a random topic (e.g., "Magic Studio" or "Billing")
        themes = self.config['review_themes_matrix']
        topic_data = random.choice(themes)
        
        # 2. Determine Sentiment/Focus based on rating
        sentiments = self.config['sentiment_and_emotion']
        
        if rating <= 2:
            trigger_key = "1_2_star_triggers"
            focus_type = "cons"  # Talk about the negative aspects
        elif rating == 3:
            trigger_key = "3_star_triggers"
            # focus_type = random.choice(["pros", "cons"]) # Mixed bag
            focus_type = "mixed"
        else:
            trigger_key = "4_5_star_triggers"
            focus_type = "pros" # Talk about the positive aspects

        emotion = random.choice(sentiments[trigger_key]['emotions'])

        if focus_type == "mixed":
            detail_focus = f"Pro: {topic_data['pros']}, but Con: {topic_data['cons']}"
        else:
            detail_focus = topic_data[focus_type] # The specific string from JSON (e.g., "Blurry graduation cards")

        return {
            "topic_name": topic_data['topic'],
            "emotion": emotion,
            "technical_detail": detail_focus,
            "keywords": topic_data['keywords']
        }

    def _construct_prompt_inputs(self, rating, persona, scenario):
        """
        Compiles the raw data into rich text strings for the DSPy signature.
        """
        kb = self.config['product_knowledge_base']
        
        # Product Context: Inject specific UI elements to avoid generic AI fluff
        # We grab 3 random UI specifics to make it feel grounded
        random_ui = ", ".join(random.sample(kb['ui_elements']['primary'] + kb['ui_elements']['visual_identifiers'], 3))
        
        product_context = (
            f"Product: {self.config['core_product_definition']['official_name']}.\n"
            f"Vibe: {kb['vibe']}\n"
            f"Key UI Elements the user might mention: {random_ui}."
        )

        persona_profile = (
            f"Role: {persona['role']}\n"
            f"Skill Level: {persona['skill']}\n"
            f"Platform: {persona['platform']}\n"
            f"Writing Style: {persona['style']}"
        )

        review_scenario = (
            f"Main Topic: {scenario['topic_name']}\n"
            f"Specific Detail to Mention: \"{scenario['technical_detail']}\"\n"
            f"Emotion/Tone: {scenario['emotion']}\n"
            f"Keywords to weave in: {', '.join(scenario['keywords'])}"
        )

        return {
            "product_context": product_context,
            "persona_profile": persona_profile,
            "review_scenario": review_scenario,
            "star_rating": str(rating)
        }

    def generate_batch(self, count=5):
        results = []
        start_batch = time.time()
        print(f"Generating {count} reviews...")
        
        for i in range(count):
            start_single = time.time()
            # 1. Sample Data
            rating = self._sample_rating()
            persona = self._build_persona()
            scenario = self._get_scenario_and_topic(rating)
            
            # 2. Build Inputs
            dspy_inputs = self._construct_prompt_inputs(rating, persona, scenario)
            
            # 3. Call DSPy
            try:
                pred = self.predictor(**dspy_inputs)  
                duration = time.time() - start_single

                # 4. Store Result
                review_entry = {
                    "rating": rating,
                    "review_text": pred.review_text,
                    "persona_role": persona['role'],
                    "topic": scenario['topic_name'],
                    "emotion": scenario['emotion'],
                    "platform": persona['platform'],
                    "generation_time_sec": round(duration, 3),
                    "char_count": len(pred.review_text)
                }
                results.append(review_entry)
                print(f"[{i+1}/{count}] Generated {rating}-star review regarding {scenario['topic_name']}")
                
            except Exception as e:
                print(f"Error generating review {i+1}: {e}")

        total_time = time.time() - start_batch
        print(f"Batch Complete. Throughput: {round(count / (total_time/60), 2)} reviews/min")

        return pd.DataFrame(results)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Review Generator CLI")
    
    parser.add_argument("--config", default="input_config.json", help="Path to input config JSON")
    parser.add_argument("--output", required=True, help="Path to save output (e.g. results.jsonl)")
    parser.add_argument("--model", choices=["ministral", "rwkv"], required=True, help="Model to use for generation")
    parser.add_argument("--count", type=int, default=10, help="Number of reviews to generate")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config {args.config} not found.")
    else:
        configure_lm(args.model)
        generator = SyntheticReviewGenerator(args.config)
        df = generator.generate_batch(count=args.count)
        
        # Save based on extension
        if args.output.endswith('.jsonl'):
            df.to_json(args.output, orient='records', lines=True)
        else:
            df.to_csv(args.output, index=False)
        
        print(f"Successfully saved {len(df)} reviews to {args.output}")