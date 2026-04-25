> An AI pipeline reported PR-AUC of 0.999 on the same 
> dataset where the corrected evaluation yielded 0.11.
> This project shows how that happens — and why it matters.

# ai-generated-mlpipeline-audit
Auditing the AI-generated machine learning pipeline under label noise, class imbalance, and evaluation leakage.

# When Evaluation Fails: K-pop Hit Prediction Case Study

## Problem

This project does not aim to build a reliable K-pop hit prediction model.

Instead, it audits an AI-assisted machine learning workflow on noisy real-world data, demonstrating how plausible modeling steps can lead to unstable or misleading conclusions.

# Workflow 
This project follows a workflow generated from structured prompts to an AI system (e.g., ChatGPT):

1. Dataset inspection:
   Identify data quality issues, leakage risks, and feasibility

2. Label definition:
   Propose multiple definitions of "hit" and analyze label noise

3. Baseline modeling:
   Build models and compare performance across labels

4. Critical audit:
   Evaluate whether results are meaningful or misleading

While the AI correctly identified many data limitations, the workflow still proceeded toward modeling and evaluation, reflecting a common pattern where analysis continues despite weak problem formulation.


---

## Key Finding


A commonly used modeling setup (SMOTE + cross-validation applied incorrectly)
produced near-perfect PR-AUC (~0.97–0.999).

However, when corrected and evaluated on a proper hold-out set,
performance dropped to realistic levels (~0.11–0.25).

This demonstrates how evaluation design, rather than true predictive signal,
can dominate model performance in noisy real-world settings.

---

## Main Results

| Label                  | Positive Rate | Wrong PR-AUC | Correct PR-AUC | Held-out PR-AUC |
|-----------------------|--------------|--------------|----------------|-----------------|
| Strict (artist+title) | 2.1%         | 0.999        | 0.11           | 0.18            |
| Loose (title only)    | 7.7%         | 0.987        | 0.18           | 0.19            |
| Spotify proxy         | 10.0%        | 0.971        | 0.25           | 0.26            |

---

## What Happened?

### 1. Evaluation Failure

Improper evaluation (SMOTE before cross-validation) produced near-perfect results:

> PR-AUC ≈ 0.99

But correct evaluation revealed the true performance:

> PR-AUC ≈ 0.11–0.25

---

### 2. Label Instability

Different definitions of “hit” produced different results:

- Strict label (clean but sparse)
- Loose label (noisy but larger)
- Spotify proxy (different concept entirely)

> The model is not learning “hit songs” — it is learning how the label is defined.

---

### 3. Fragile Conclusions

Feature patterns appear meaningful:

- louder songs
- less instrumental
- less acoustic

But these patterns:

- change across label definitions
- do not produce strong predictive power

---

## Final Insight

The key issue is not that evaluation leakage inflates performance — this is well known.

Rather, this project shows that even when data limitations are clearly identified,
AI-assisted workflows and standard data science practices tend to proceed toward modeling and optimization.

This creates a failure mode where:
- evaluation design dominates performance
- results appear plausible
- conclusions become easy to over-interpret

In other words, the risk is not incorrect results,
but workflows that make misleading results appear reasonable.

---

## Conclusion

## Conclusion

This project does not demonstrate a failure of models, but a failure mode of workflows.

Even when data issues are correctly identified,
AI-assisted pipelines can naturally progress toward modeling and evaluation,
producing results that are technically valid but highly sensitive to assumptions.

Without careful auditing, such workflows can lead to conclusions that appear meaningful but do not generalize.

This highlights the importance of evaluating not just model performance,
but the reasoning process and evaluation design behind it.

---

## Code

See `final_music_audit.py` for full implementation.

---

## Reproducibility

```bash
pip install -r requirements.txt
python final_music_audit.py


## Data Sources
- single_album_track_data.csv: Kaggle — K-pop discography
- kpop_hits_all_years.csv: Kaggle — K-pop hit list  
- spotify_tracks.csv: Kaggle — Spotify tracks
