# Auditing AI-Generated ML Pipelines: Reliability and Generalization Under Noisy Conditions

> **Research Question**  
> *When AI systems generate end-to-end data science workflows, under what conditions can they produce misleading conclusions?*

---

## Overview

This project evaluates the reliability of **AI-generated data science workflows** under realistic data challenges.

While Large Language Models (LLMs) can generate technically plausible pipelines, this work demonstrates that under specific conditions—such as noisy labels, proxy targets, and distribution shifts—they can produce **misleading performance estimates** unless underlying assumptions are rigorously audited.

To investigate this, I audited AI-generated pipelines across two contrasting domains:

1. **K-pop hit prediction** — a weakly defined, high-noise problem  
2. **Healthcare prediction** — a structured but distribution-shifted problem  

---

## Approach

To evaluate AI-generated workflows, I:

1. Prompted an AI system to generate an end-to-end ML pipeline  
2. Executed the generated pipeline without modification  
3. Audited the underlying assumptions (label validity, evaluation design, data leakage)  
4. Identified failure points in the pipeline  
5. Reconstructed the evaluation procedure to remove leakage and bias  
6. Compared naive and corrected performance across domains  

> I audited the generated pipelines, identified sources of leakage and invalid evaluation design, and restructured validation procedures to obtain more realistic performance estimates.

---

## Key Finding

Across both domains, strong model performance often depended more on **evaluation design and label construction** than on true predictive signal.

> **The central finding is that model performance is highly sensitive not to the model itself, but to label definition, evaluation design, and data-generating assumptions.**

---

## Case Study 1: K-pop Hit Prediction (Label Instability)

The K-pop task is inherently noisy because the concept of a “hit song” is not cleanly defined.

Three label definitions were evaluated:

- Strict artist–title match against a hit list  
- Loose title-only match  
- Spotify popularity threshold (proxy label)  

| Label Definition | Naive CV (Wrong) | Corrected CV | Held-out Test |
|------------------|----------------:|-------------:|--------------:|
| Strict           | **0.999**       | **0.114**    | 0.182         |
| Loose            | 0.988           | 0.182        | 0.199         |
| Spotify Proxy    | 0.971           | 0.251        | 0.259         |

### Interpretation

- Near-perfect performance disappeared after correcting evaluation leakage  
- Different label definitions produced different conclusions  
- Feature importance and model behavior were unstable  

> In the K-pop case, model conclusions changed substantially depending on label definition, indicating that the model was learning **label construction artifacts rather than a stable concept of “hit song.”**

The realistic conclusion is:

> Audio features alone show limited and unstable signal, and the current setup does not support a reliable hit prediction model.

---

## Case Study 2: Healthcare Prediction (Distribution Shift)

The healthcare task used a heart disease dataset with multiple hospital cohorts.

### Random Split (Naive)

| Metric | Value |
|--------|------:|
| ROC-AUC | 0.920 |
| PR-AUC  | 0.906 |

### Site-held-out Validation (Realistic)

| Cohort | Positive Rate | ROC-AUC | PR-AUC |
|--------|--------------:|--------:|-------:|
| Cleveland | 0.457 | 0.855 | 0.854 |
| Hungary   | 0.362 | 0.895 | 0.846 |
| Switzerland | 0.935 | 0.747 | 0.973 |
| VA Long Beach | 0.745 | 0.724 | 0.852 |

### Interpretation

- The model retained predictive signal, but performance varied across cohorts  
- Random split overestimated generalization due to cohort mixing  

A critical example:

> The Switzerland cohort shows PR-AUC ≈ 0.97, but the positive rate is ≈ 93.5%.  
> This indicates that the high PR-AUC is largely driven by **class imbalance**, not true discriminative performance.

The healthcare case shows that:

> Even when models are functional, naive evaluation can produce **over-optimistic conclusions about generalization**.

---

## Observed Failure Modes

### 1. Evaluation Leakage (SMOTE Misplacement)

- SMOTE applied before cross-validation  
- Synthetic samples appear in both training and validation folds  
- Leads to inflated metrics (e.g., PR-AUC ≈ 0.999)

---

### 2. Label Instability

- Different definitions of “hit” produce different results  
- Model conclusions depend on labeling assumptions  
- Particularly severe in weakly defined domains

---

### 3. Proxy Target Bias

- Spotify popularity is not equivalent to cultural or commercial success  
- Models learn platform dynamics rather than underlying signal  

---

### 4. Distribution Shift

- Random splits hide cohort-level differences  
- Site-held-out evaluation reveals generalization gaps  

---

### 5. Metric Misinterpretation

- Metrics like PR-AUC depend heavily on class balance  
- High scores may reflect data distribution rather than model quality  

---

## Why AI-Generated Pipelines Fail

These issues arise from structural limitations in AI-assisted workflows:

1. **Context-Agnostic Pattern Matching**  
   AI applies common techniques (e.g., SMOTE, random split) without reasoning about correct placement or assumptions  

2. **Lack of Data Generating Process Awareness**  
   AI does not fully account for how data is collected (e.g., noisy labels, cohort bias)  

3. **Implicit Optimization for Plausible Results**  
   AI tends to produce pipelines that yield strong metrics, even when those metrics are misleading  

---

## Final Conclusion

This project is not a demonstration of successful prediction models.

Instead, it is an audit of **how AI-generated workflows can produce misleading conclusions if not carefully evaluated**.

> High performance metrics should be treated as **hypotheses to be audited**, not as ground truth.

The key takeaway is:

> AI-generated workflows do not always fail visibly —  
> they can **fail silently**, producing convincing but unreliable results when assumptions are not explicitly examined.

---

## Code

- `final_music_audit.py` — K-pop label and evaluation audit  
- `healthcare_audit.py` — healthcare cohort validation audit  

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
