# 📊 Comprehensive Comparison Report

**Target Rating Mean (Config):** 3.5

## synthetic_reviews (Ministral)
- **Total Samples:** 400
- **Mean Rating:** 3.45
- **Mean Word Count:** 244.0
- **Distribution Alignment to Real Data (KS Test):** Stat=0.5832, p-val=0.0000
  - ⚠️ Distributions are statistically DIFFERENT.
- **Avg Generation Time:** 7.849s
- **Avg Time Per Word:** 0.0331s/word
- **Est. Total Cost:** $0.20
- **Guardrail Rejection Rate:** 28.5%
  - *Duplicates:* 78
  - *Off-Topic:* 61
  - *Sentiment Mismatch:* 1
  - *Low Quality (Spam/Short):* 3

## synthetic_reviews (RWKV)
- **Total Samples:** 400
- **Mean Rating:** 3.43
- **Mean Word Count:** 172.0
- **Distribution Alignment to Real Data (KS Test):** Stat=0.5707, p-val=0.0000
  - ⚠️ Distributions are statistically DIFFERENT.
- **Avg Generation Time:** 7.753s
- **Avg Time Per Word:** 0.0458s/word
- **Est. Total Cost:** $0.20
- **Guardrail Rejection Rate:** 72.5%
  - *Duplicates:* 268
  - *Off-Topic:* 54
  - *Sentiment Mismatch:* 0
  - *Low Quality (Spam/Short):* 0

## Real Data
- **Total Samples:** 65269
- **Mean Rating:** 4.48
- **Mean Word Count:** 5.5
- **Guardrail Rejection Rate:** N/A (Baseline/Not Run)

