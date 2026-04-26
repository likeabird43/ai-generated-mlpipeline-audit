# Auditing AI-Generated ML Pipelines: Reliability and Generalization Under Noisy Conditions

> **Research Question**
> *When AI systems generate end-to-end data science workflows, under what conditions do they produce misleading or unreliable conclusions?*

---

## Overview

This project evaluates the reliability of **AI-generated data science workflows** under realistic data challenges.

Large Language Models (LLMs) can generate technically plausible pipelines.
However, their outputs are often **highly sensitive to evaluation design, label construction, and problem framing**, leading to misleading conclusions if not carefully audited.

Rather than building a better model, this project investigates:

> **Can AI-generated ML pipelines be trusted without external validation?**

---

## Approach

1. Generate an ML pipeline using an AI system
2. Execute the pipeline without modification
3. Identify inconsistencies and unrealistic results
4. Issue structured audit prompts
5. Reconstruct evaluation procedures
6. Re-implement corrected pipelines
7. Compare naive vs corrected performance

---

## Key Findings

### 1. Evaluation design dominates performance

* Naive CV (with leakage) → **PR-AUC ≈ 1.0**
* Corrected CV → **PR-AUC ≈ 0.18**
* Held-out test → **PR-AUC ≈ 0.11**

> High performance can be entirely driven by evaluation artifacts.

---

### 2. Structural leakage is a major hidden factor

Even with correct pipelines, performance can be inflated due to:

* shared artists
* duplicated structure
* dataset-level dependencies

Group-based validation reveals this effect.

---

### 3. Label construction strongly affects conclusions

| Label   | PR-AUC | Baseline | Relative Gain |
| ------- | ------ | -------- | ------------- |
| Strict  | 0.114  | 0.024    | ~4.8x         |
| Loose   | 0.230  | 0.077    | ~3.0x         |
| Spotify | 0.584  | 0.311    | ~1.9x         |

> Higher performance often reflects **easier labels**, not stronger signal.

---

### 4. AI behavior is inconsistent

AI-generated workflows exhibit:

* overconfidence
* incomplete reasoning
* strong dependence on prompt structure

---

### 5. External validation is essential

AI-generated results should be treated as:

> **hypotheses, not conclusions**

---

# Case Study 1: K-pop — Evaluation Failure

## Experiment 1: Pipeline Audit

We audit an AI-generated pipeline for predicting K-pop hit songs.

### Results

| Evaluation Type | PR-AUC    |
| --------------- | --------- |
| Naive CV        | **0.999** |
| Corrected CV    | 0.182     |
| Group CV        | 0.054     |
| Held-out Test   | 0.114     |
| Baseline        | 0.024     |

---

## Interpretation

* Near-perfect performance is caused by **data leakage (SMOTE misuse)**
* Even corrected CV remains optimistic
* **Group-based evaluation reveals generalization failure**

> The model is not learning “hit songs” —
> it is exploiting dataset structure (e.g., artist identity)

---

## Artist Leakage (Key Mechanism)

Standard CV allows:

* same artist in train and validation

Model learns:

```text
artist patterns ❌ general musical patterns
```

Using **GroupKFold (artist-level split)**:

| Evaluation Type | PR-AUC |
| --------------- | ------ |
| Corrected CV    | 0.182  |
| Group CV        | 0.054  |
| Held-out        | 0.114  |
| Baseline        | 0.024  |

> Performance collapses when evaluated on unseen artists.

---

# Case Study 2: K-pop — Label Sensitivity

## Experiment 2: Redefining “Hit”

We test how label definition changes model behavior.

### Labels

* **Strict** → artist + title
* **Loose** → title only
* **Spotify proxy** → popularity-based

---

## Results

| Label   | PR-AUC | Baseline | Relative Gain |
| ------- | ------ | -------- | ------------- |
| Strict  | 0.114  | 0.024    | ~4.8x         |
| Loose   | 0.230  | 0.077    | ~3.0x         |
| Spotify | 0.584  | 0.311    | ~1.9x         |

---

## Interpretation

* Performance increases as labels become easier
* Relative improvement decreases
* Spotify reflects **popularity, not musical quality**

> The model adapts to the label definition, not intrinsic signal.

---

# Case Study 3: Healthcare — Distribution Shift

*(Distribution Shift)*

### Random Split

* ROC-AUC: **0.92**
* PR-AUC: **0.91**

### Site-held-out Validation

| Cohort        | ROC-AUC | PR-AUC | Positive Rate |
| ------------- | ------- | ------ | ------------- |
| Cleveland     | 0.85    | 0.85   | 0.46          |
| Hungary       | 0.89    | 0.85   | 0.36          |
| Switzerland   | 0.75    | 0.97   | 0.93          |
| VA Long Beach | 0.72    | 0.85   | 0.75          |

Note: extremely high PR-AUC is driven by class imbalance.

> Performance depends strongly on data distribution.

---

# Cross-Model Comparison

### ChatGPT

* conservative but incomplete
* recognizes issues but under-analyzes them

### Claude

* overconfident initial outputs
* fails to detect evaluation flaws
* corrects only after explicit prompts

---

### After Audit

Both models converge to:

```text
PR-AUC ≈ 0.01 – 0.11 (strict)
```

---

### Key Insight

> AI does not reliably detect flawed evaluation setups.

---

# Observed Failure Modes

1. Evaluation leakage
2. Structural leakage (artist-level)
3. Label instability
4. Proxy bias
5. Distribution shift

---

# Final Conclusion

This project is not about building a better model.

It is an audit of how models fail.

> Model performance is often determined by
> **evaluation design, label construction, and data structure**
> rather than true predictive signal.

---

## Practical Implication

Audio features alone are insufficient for predicting culturally defined outcomes such as “hit songs.”

However, they may still be useful for:

* similarity-based retrieval
* preference modeling
* recommendation systems

> Audio features capture **structure**, not **meaning**

---

# Contribution

* Demonstrates multiple ML failure modes
* Shows how performance can be systematically misleading
* Provides an audit framework for AI-generated pipelines

---

# Code

* final_audit_metrics.csv
* final_feature_importance.csv
* second_experiment_feature_importance.csv
* second_experiment_metrics.csv
  
* healthcare_audit_metrics.csv


---

## Data

Datasets are sourced from publicly available datasets and are not included in this repository.

The project uses the following data:

### K-pop datasets
- `single_album_track_data.csv`  
  → full K-pop discography (audio features)

- `KPopHits2021.csv`  
  → curated list of K-pop hit songs (2021)

- `kpop_hits_all_years.csv`  
  → extended multi-year K-pop hit list

- `spotify_tracks.csv`  
  → global Spotify dataset used for proxy label (popularity-based)

---

### Healthcare dataset
- `heart_disease_uci.csv`  
  → UCI Heart Disease dataset (multi-site cohorts: Cleveland, Hungary, Switzerland, VA Long Beach)

---

These datasets were used to test:

- label instability (K-pop)
- evaluation leakage (K-pop)
- distribution shift (healthcare)
  

---

## Discussion: Can AI Learn Music Beyond Proxy Features?

This project shows that audio features provide weak signals for culturally defined outcomes.

This raises a broader question:

> Can AI learn music in a way aligned with human perception?

Current models capture structure (e.g., loudness, energy),
but fail to capture:

* cultural context
* temporal dynamics
* emotional meaning

> Current ML systems capture structure, but not meaning.
