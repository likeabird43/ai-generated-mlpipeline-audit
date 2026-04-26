## Audit Case: Conditional Self-Correction in AI-Generated Pipeline (Claude)

---

### Prompt

```text id="2z8t9n"
You are a professional data scientist.

I have two K-pop datasets:
1. single_album_track_data.csv
2. KPopHits2021.csv

Your task:
- Build a model to predict hit songs
- Report performance metrics
- Interpret results
```

---

### AI-Generated Output (verbatim excerpt, Claude)

> “Random Forest and Gradient Boosting both hit ROC-AUC ~0.9997.
> PR-AUC is 0.74 — very strong for a 0.37% positive class.”

---

### Follow-up Audit Prompt

The model was then asked to verify its own methodology and evaluation.

---

### Self-Audit Response (excerpt)

> “SMOTE was applied before cross-validation — this is a data leakage error.”

> “The original PR-AUC of 0.741 was computed with leakage and is not a valid estimate.”

> “Corrected PR-AUC is ~0.034, barely above random baseline (~0.004).”

> “Estimated false match rate: ~38% of positive labels.”

---

### Audit

The initial output presents **extremely high performance**, suggesting near-perfect predictive power.

However, this result is driven by **critical evaluation flaws**, which the model fails to detect independently.

Only after explicit prompting does the model identify:

* **data leakage** (SMOTE applied before cross-validation)
* **label corruption** (~38% false positives due to matching errors)
* **invalid evaluation metrics**

> Performance collapses after correction:

```text id="y8r3sn"
PR-AUC ≈ 0.74 → ≈ 0.01–0.03 (near-random range depending on evaluation setup)
```

---

### Key Insight

> AI systems may produce highly confident but invalid results,
> and only identify their own errors under explicit and structured audit.

This illustrates a distinct failure mode:

* not just overconfidence
* but **conditional self-correction**

---

### Connection to Main Findings

This case demonstrates that:

* AI-generated workflows are **not reliably self-validating**
* critical methodological errors can go undetected in initial outputs
* even when correct reasoning is possible, it is **not triggered by default**

> Reliable conclusions require **external validation and structured audit**,
> not just internal consistency of the generated analysis.
