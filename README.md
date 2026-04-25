# Auditing AI-Generated ML Pipelines: Reliability and Generalization Under Noisy Conditions

> **Research Question**  
> *When AI systems generate end-to-end data science workflows, under what conditions do they produce misleading or unreliable conclusions?*

---

## Overview

This project evaluates the reliability of **AI-generated data science workflows** under realistic data challenges.

Large Language Models (LLMs) can generate technically plausible pipelines.  
However, their outputs can be **highly sensitive to evaluation design, label construction, and problem framing**, often leading to misleading conclusions if not carefully audited.

Importantly, AI-generated workflows do not fail in a single consistent way.  
Instead, they exhibit **unstable and context-dependent behavior**, sometimes producing:

- overconfident results  
- conservative but incomplete assessments  
- partially correct interpretations that miss critical flaws  
- correct reasoning in structured settings  

To investigate this, I audited AI-generated workflows across two domains:

1. **K-pop hit prediction** — weakly defined, label-noisy problem  
2. **Healthcare prediction** — structured but distribution-shifted problem  

---

## Approach

To evaluate AI-generated workflows, I followed a structured audit process:

1. Prompted an AI system to generate an end-to-end ML pipeline  
2. Executed the generated pipeline without modification  
3. Identified inconsistencies and unrealistic results  
4. Issued targeted audit prompts to test reliability  
5. Reconstructed evaluation procedures to remove bias and leakage  
6. Re-implemented corrected pipelines in Python  
7. Compared naive vs corrected performance  

> The goal is not to optimize model performance, but to test whether AI-generated workflows produce **reliable conclusions under scrutiny**.

---

## Key Findings

### 1. Evaluation design dominates performance

Model performance is often determined more by **evaluation setup** than by actual predictive signal.

Example (K-pop task):

- Naive CV (incorrect SMOTE usage) → **PR-AUC ≈ 0.999**
- Corrected CV → **PR-AUC ≈ 0.114**

> High performance can be entirely driven by evaluation artifacts.

---

### 2. Label construction is unstable

Different definitions of “hit” produce dramatically different results:

- strict artist–title match  
- loose title-only match  
- Spotify popularity proxy  

> The model is not learning a stable concept, but artifacts of label construction.

---

### 3. AI behavior is inconsistent

AI-generated workflows do not behave uniformly.

Observed behaviors include:

- **Overconfidence**  
  - near-perfect metrics under flawed evaluation  

- **Conservative behavior**  
  - low confidence when data is clearly noisy  

- **Partial understanding**  
  - detects generalization failure but misses root causes  

- **Correct reasoning (structured cases)**  
  - produces realistic and cautious analysis  

> AI can adjust its confidence, but does not reliably detect when its own evaluation setup is invalid.

---

### 4. External validation is essential

After auditing and re-implementing pipelines:

- PR-AUC dropped from ~0.74 → ~0.01–0.02  
- ROC-AUC dropped from ~1.0 → ~0.76–0.81  

> AI-generated results should be treated as **hypotheses**, not conclusions.

---

## Case Study 1: K-pop Hit Prediction (Label Instability + Evaluation Failure)

The K-pop task is inherently noisy:

- “hit song” is not clearly defined  
- label coverage is incomplete  
- matching between datasets is imperfect  

Three label definitions were evaluated:

| Label Definition | Naive CV | Corrected CV | Held-out Test |
|------------------|---------:|-------------:|--------------:|
| Strict           | **0.999** | **0.114**    | 0.182         |
| Loose            | 0.988     | 0.182        | 0.199         |
| Spotify Proxy    | 0.971     | 0.251        | 0.259         |

### Interpretation

- **Evaluation failure reproduced:**  
  SMOTE applied before CV led to near-perfect performance  

- **Correction reveals reality:**  
  Performance collapses after fixing evaluation  

- **Label instability:**  
  Different definitions lead to different conclusions  

> The model is not learning “hit songs,” but **evaluation and labeling artifacts**.

---

## Case Study 2: Healthcare Prediction (Distribution Shift)

The healthcare dataset shows a different failure mode.

### Random Split (Naive)

- ROC-AUC: 0.920  
- PR-AUC: 0.906  

### Site-held-out Validation (Realistic)

Performance varies significantly across cohorts.

Key example:

> PR-AUC ≈ 0.97 with positive rate ≈ 93.5%  
> → metric inflation driven by class imbalance

### Interpretation

- Random splits hide real-world distribution shifts  
- Metrics can appear strong while generalization is weak  

---

## Observed Failure Modes

### 1. Evaluation Leakage
- SMOTE applied before cross-validation  
- leads to artificial performance inflation  

### 2. Label Instability
- different definitions produce different models  

### 3. Proxy Target Bias
- popularity ≠ true success  

### 4. Distribution Shift
- random splits overestimate generalization  

### 5. Metric Misinterpretation
- PR-AUC heavily depends on class balance  

---

## Why AI-Generated Workflows Fail

These issues arise from structural limitations:

1. **Pattern-based reasoning without deep validation**  
2. **Lack of data-generating process awareness**  
3. **Sensitivity to prompt framing and evaluation setup**  

---

## Final Conclusion

This project is not about building a successful prediction model.

It is an audit of how AI-generated workflows behave under realistic conditions.

> AI-generated ML workflows produce **unstable, context-dependent conclusions** that require external validation.

The key insight:

> High performance metrics should be treated as **hypotheses to audit**, not ground truth.

---

## Code

- `final_music_audit.py` — evaluation and label audit for K-pop task  
- `healthcare_audit.py` — cohort-based validation  

---


## Data

Datasets are sourced from public datasets and are not included.

```text
data/
  kpopfullspotifydiscography/
    single_album_track_data.csv
  kpop_hits_all_years.csv
  spotify_tracks.csv
  heart_disease_uci.csv
```
