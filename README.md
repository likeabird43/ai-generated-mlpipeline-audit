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

* overconfident results
* conservative but incomplete assessments
* partially correct interpretations that miss critical flaws
* correct reasoning in structured settings

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

* Naive CV (incorrect SMOTE usage) → **PR-AUC ≈ 0.999**
* Corrected CV → **PR-AUC ≈ 0.18**
* Held-out test → **PR-AUC ≈ 0.11**

> High performance can be entirely driven by evaluation artifacts.

---

### 2. Label construction is unstable

Different definitions of “hit” produce dramatically different results:

* strict artist–title match
* loose title-only match
* Spotify popularity proxy

| Label   | PR-AUC (Held-out) | Baseline |
| ------- | ----------------- | -------- |
| Strict  | 0.114             | 0.024    |
| Loose   | 0.230             | 0.077    |
| Spotify | 0.265             | 0.100    |

> Performance improves as labels become looser,
> but this reflects **label construction**, not stronger predictive signal.

---

### 3. AI behavior is inconsistent

AI-generated workflows do not behave uniformly.

Observed behaviors include:

* **Overconfidence**

  * near-perfect metrics under flawed evaluation

* **Conservative behavior**

  * low confidence when data is clearly noisy

* **Partial understanding**

  * detects generalization failure but misses root causes

* **Correct reasoning (structured cases)**

  * produces realistic and cautious analysis

> AI can adjust its confidence, but does not reliably detect when its own evaluation setup is invalid.

---

### 4. External validation is essential

After auditing and re-implementing pipelines:

* PR-AUC dropped from ~0.74 → ~0.01–0.03
* ROC-AUC dropped from ~1.0 → ~0.70

> AI-generated results should be treated as **hypotheses**, not conclusions.


---
## Example AI-Generated Workflow

Below is a simplified version of an AI-generated pipeline:

> “Apply SMOTE to balance the dataset, then perform cross-validation and report PR-AUC.”

This approach appears reasonable, but introduces a critical flaw:

- SMOTE is applied before cross-validation
- This leads to data leakage
- Resulting in artificially inflated performance (PR-AUC ≈ 0.999)

After auditing and correcting the pipeline:

- PR-AUC drops to ≈ 0.18 (CV)
- PR-AUC drops further to ≈ 0.11 (held-out)

> This demonstrates how AI-generated workflows can produce plausible but invalid evaluation pipelines.

---
## Case Study 1: K-pop Hit Prediction

*(Label Instability + Evaluation Failure)*

The K-pop task is inherently noisy:

* “hit song” is not clearly defined
* label coverage is incomplete
* matching between datasets is imperfect

### Results

| Label Definition |  Naive CV | Corrected CV | Held-out Test |
| ---------------- | --------: | -----------: | ------------: |
| Strict           | **0.999** |    **0.182** |     **0.114** |
| Loose            |     0.988 |        0.220 |         0.230 |
| Spotify Proxy    |     0.971 |        0.263 |         0.265 |

### Interpretation

* **Evaluation failure reproduced**
  → SMOTE before CV inflates performance

* **Correction reveals reality**
  → performance drops sharply

* **Label sensitivity**
  → different definitions produce different conclusions

> The model is not learning “hit songs,”
> but is highly sensitive to how the target is constructed.

---

### Robustness to Label Definition

Across both ChatGPT and Claude:

* PR-AUC remains low (~0.05–0.26)
* Feature importance is unstable
* Results depend strongly on label definition

> There is no stable predictive signal — only label-dependent behavior.

---

## Case Study 2: Healthcare Prediction

*(Distribution Shift)*

### Random Split (Naive)

* ROC-AUC: **0.92**
* PR-AUC: **0.91**

### Site-held-out Validation

| Cohort        | ROC-AUC | PR-AUC | Positive Rate |
| ------------- | ------- | ------ | ------------- |
| Cleveland     | 0.85    | 0.85   | 0.46          |
| Hungary       | 0.89    | 0.85   | 0.36          |
| Switzerland   | 0.75    | 0.97   | 0.93          |
| VA Long Beach | 0.72    | 0.85   | 0.75          |

Note: The extremely high PR-AUC is largely driven by the very high positive rate (93%).

### Interpretation

* Random splits produce inflated performance due to cohort mixing
* Site-held-out results vary significantly across cohorts
* Extremely high PR-AUC (e.g., Switzerland) is driven by class imbalance

> Evaluation results depend strongly on data distribution.

---

## Cross-Model Consistency

Experiments were repeated using multiple LLMs (ChatGPT and Claude).

Across models:

* Both initially produced highly optimistic results
* Both failed under flawed evaluation setups
* Both converged to low, unstable performance after correction

> The failure is not model-specific,
> but reflects a broader limitation of AI-generated workflows.

---

## Observed Failure Modes

### 1. Evaluation Leakage

* SMOTE applied before cross-validation

### 2. Label Instability

* Different definitions → different models

### 3. Proxy Target Bias

* Popularity ≠ true outcome

### 4. Distribution Shift

* Random splits hide generalization failure

### 5. Metric Misinterpretation

* PR-AUC inflated by class imbalance

---

## Final Conclusion

This project is not about building a successful prediction model.

It is an audit of how AI-generated workflows behave under realistic conditions.

> AI-generated ML workflows produce **unstable, context-dependent conclusions** that require external validation.

The key insight:

> Model performance is often determined by **how the problem is defined and evaluated**,
> not by the underlying predictive signal.

--- 
## Contribution

While the individual failure modes of ML pipelines are known,
this project contributes a structured empirical audit of AI-generated workflows.

Specifically, it shows that:

- evaluation design, label construction, and data distribution independently distort model performance
- these failure modes consistently appear across different models (ChatGPT and Claude)
- AI-generated pipelines can produce plausible but invalid workflows without detecting their own errors

Rather than proposing a new model,
this work provides a reproducible framework for auditing the reliability of AI-generated ML pipelines.

---

## Code

* `final_music_audit.py`
* `healthcare_audit.py`

---

## Data

Datasets are sourced from public datasets and are not included.

```
data/
  kpopfullspotifydiscography/
    single_album_track_data.csv
  kpop_hits_all_years.csv
  kpophits2021.csv
  spotify_tracks.csv
  heart_disease_uci.csv
```


---

## Discussion: Can AI Learn Music Beyond Proxy Features?

This project shows that audio features provide only weak and unstable signal for culturally defined outcomes such as “hit songs.”

This raises a broader question:

> Can AI learn music in a way that aligns with human perception?

Current models rely on proxy features (e.g., loudness, energy, mode),
which capture production characteristics rather than musical meaning.

While such features provide partial signal,
they fail to capture:

- contextual dynamics (e.g., trends, timing)
- cultural factors (e.g., fandom, artist identity)
- subjective experience (e.g., emotional resonance)

This suggests a fundamental gap:

> Current ML systems capture structure, but not meaning.

Future work may require integrating richer signals,
such as listener behavior, temporal context, and cultural information,
to better align AI representations with human musical experience.
