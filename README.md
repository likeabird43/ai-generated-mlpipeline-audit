> **Research Question**
> *When AI systems generate end-to-end data science workflows, under what conditions do they produce systematically misleading conclusions?*

---

# Systematic Failure Modes in AI-Generated ML Pipelines

## Overview

This project evaluates the reliability of **AI-generated data science workflows** in real-world, noisy environments.

While large language models (LLMs) can generate plausible end-to-end pipelines, this work shows that under common conditions, they can produce **catastrophically misleading evaluation results**.

By auditing workflows across two distinct domains — **K-pop hit prediction (high-noise)** and **healthcare prediction (high-stakes)** — this project identifies **systematic failure modes** where evaluation design dominates true predictive signal.

---

## Method

To empirically evaluate AI-generated workflows, I:

1. Prompted an AI system (e.g., ChatGPT) to generate a complete ML pipeline
2. Executed the pipeline **without modification**
3. Identified flaws in evaluation design (e.g., leakage, invalid validation)
4. Corrected the pipeline using proper methodology
5. Compared performance between naive and corrected setups
6. Repeated the audit across domains (K-pop and Healthcare)

---

## Key Finding

A commonly generated workflow (**SMOTE applied before cross-validation**) produced near-perfect performance:

> **PR-AUC ≈ 0.97–0.999**

However, when corrected and evaluated properly:

> **PR-AUC ≈ 0.11–0.25**

This demonstrates that:

> **Evaluation design — not true predictive signal — can dominate model performance.**

---

## 1. Observed AI Failure Modes

The audit reveals three systematic failure patterns:

### 1. Evaluation Leakage via Sequential Bias

AI frequently applies preprocessing (e.g., SMOTE) before cross-validation,
introducing leakage and inflating metrics.

---

### 2. Optimization for Plausibility over Validity

Generated pipelines follow “standard best practices”
without adapting to dataset-specific characteristics (e.g., label noise, imbalance).

---

### 3. Overconfidence under Distribution Shift

AI-generated models produce strong metrics under naive validation,
but fail when evaluated under realistic conditions.

---

## 2. Cross-Domain Evidence

### Case A: K-pop (High Noise / Weak Labels)

An AI-generated pipeline achieved near-perfect performance due to evaluation leakage.

| Setup                | PR-AUC    | Interpretation |
| -------------------- | --------- | -------------- |
| AI-generated (naive) | **0.999** | Misleading     |
| Audited (corrected)  | **0.11**  | Realistic      |

👉 The model learned **evaluation artifacts**, not musical signal.

---

### Case B: Healthcare (Distribution Shift)

Under standard validation:

* ROC-AUC: **0.92**
* PR-AUC: **0.91**

Under realistic evaluation (site-held-out):

* ROC-AUC: **0.72**
* Performance becomes unstable

👉 The model fails to generalize across hospitals.

---

## 3. Why AI Systems Fail

These failures are not accidental — they arise from structural limitations in AI-generated workflows:

### 1. Context-Agnostic Pattern Matching

AI prioritizes commonly seen patterns (e.g., SMOTE + CV)
without reasoning about correct placement or assumptions.

---

### 2. Lack of Data Generating Process (DGP) Reasoning

AI struggles to understand how data is created:

* K-pop: playlist curation → label noise
* Healthcare: hospital differences → distribution shift

---

### 3. Implicit Optimization for “Good Results”

AI tends to produce pipelines that **look correct and yield high metrics**,
even when those metrics are invalid.

---

## 4. Final Decision: Stop and Reframe

The appropriate conclusion is **not to improve the model**, but to **reframe the problem**.

This project shows that:

* High performance metrics can be **artifacts of evaluation design**
* AI-generated workflows can produce **hallucinated performance**
* Validation assumptions must be audited, not trusted

---

## Implications

This has direct implications for real-world use:

* **Business / industry:** flawed AI analysis can lead to incorrect strategic decisions
* **Healthcare:** improper validation can create unsafe models
* **AI deployment:** human oversight must focus on **assumptions**, not just code

---

## Conclusion

Across both domains:

* Weakly defined problems (K-pop)
* and structured problems (healthcare)

AI-generated workflows can produce convincing but unreliable results.

> **The core risk is not model performance — but misplaced trust in evaluation.**

---

## Code

* `final_music_audit.py` — K-pop audit
* `healthcare_audit.py` — cross-domain validation

---

## Data

Datasets are sourced from public datasets (e.g., Kaggle) and are not included.

```text
data/
  kpopfullspotifydiscography/
    single_album_track_data.csv
  kpop_hits_all_years.csv
  spotify_tracks.csv
  heart_disease_uci.csv
```

---

## Reproducibility

```bash
pip install -r requirements.txt
python final_music_audit.py
python healthcare_audit.py
```
