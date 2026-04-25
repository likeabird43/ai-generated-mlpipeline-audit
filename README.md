# Auditing AI-Generated ML Pipelines: Reliability and Generalization Under Noisy Conditions

> **Research Question**
> *When AI systems generate end-to-end data science workflows, under what conditions do they produce misleading or unreliable conclusions?*

---

## Overview

This project evaluates the reliability of **AI-generated data science workflows** under realistic data challenges.

Large Language Models (LLMs) can generate technically plausible pipelines.
However, their outputs can be **highly sensitive to evaluation design, label construction, problem framing, and prompt structure**, often leading to misleading conclusions if not carefully audited.

Importantly, AI-generated workflows do not fail in a single consistent way.
Instead, they exhibit **unstable and context-dependent behavior**, including:

* overconfident results
* conservative but incomplete assessments
* partially correct reasoning that misses critical flaws
* correct analysis under structured constraints

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

Additionally, I tested how **prompt structure** affects AI behavior:

* **Open-ended prompts** → optimistic or flawed evaluation pipelines
* **Structured audit prompts** → more conservative and methodologically sound analysis

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
  → near-perfect metrics under flawed evaluation

* **Conservative behavior**
  → low confidence when data is clearly noisy

* **Partial understanding**
  → detects high-level issues but misses root causes

* **Correct reasoning (structured prompts)**
  → produces realistic and cautious analysis

> AI can adjust its confidence, but does not reliably detect when its own evaluation setup is invalid.

---

### 4. External validation is essential

After auditing and re-implementing pipelines:

* PR-AUC dropped from ~0.74 → ~0.01–0.03
* ROC-AUC dropped from ~1.0 → ~0.70

> AI-generated results should be treated as **hypotheses, not conclusions**.

---

### 5. Prompt design significantly affects reliability

Prompt structure directly influences the quality of AI-generated workflows.

* Open-ended prompts → inflated metrics and missed evaluation flaws
* Structured audit prompts → conservative and methodologically sound outputs

> Reliable use of AI requires **explicitly structured prompts**, not just better models.

---

## Example AI-Generated Workflow

A concrete example is provided in:

→ `examples/kpop_pipeline_chatgpt.md`
→ `examples/kpop_pipeline_claude.md`

These show how different models produce **different types of failure** under similar tasks.

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

* Evaluation failure reproduced
* Correction reveals true performance
* Results highly sensitive to label definition

> The model is not learning “hit songs,”
> but reacting to how the target is constructed.

---

## Case Study 2: Healthcare Prediction

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

## Cross-Model Consistency

Experiments were repeated using multiple LLMs (ChatGPT and Claude).

While both models required external audit, their **failure patterns differed**:

* **ChatGPT**
  → conservative but incomplete
  → identifies limitations (e.g., label mismatch)
  → does not fully analyze their implications

* **Claude**
  → overconfident initial results
  → fails to detect evaluation flaws
  → corrects errors only after explicit prompting

After correction, both converge to similarly weak performance.

> AI systems do not fail uniformly —
> failure depends on reasoning path, assumptions, and prompt structure.

---

## Observed Failure Modes

1. Evaluation leakage
2. Label instability
3. Proxy target bias
4. Distribution shift
5. Metric misinterpretation

---

## Final Conclusion

This project is not about building a better model.

It is an audit of how AI-generated workflows behave.

> AI-generated ML workflows produce **unstable, context-dependent conclusions** that require external validation.

The key insight:

> Model performance is often determined by **problem definition and evaluation design**,
> not by underlying predictive signal.

---

## Contribution

This project provides a structured empirical audit of AI-generated workflows.

It shows that:

* evaluation design, label construction, and data distribution independently distort results
* failure patterns vary across models
* prompt design significantly influences reliability
* AI-generated pipelines can appear valid while being fundamentally flawed

Rather than proposing a new model,
this work offers a **framework for auditing AI-generated ML pipelines**.

---

## Code

* `final_music_audit.py`
* `healthcare_audit.py`

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
