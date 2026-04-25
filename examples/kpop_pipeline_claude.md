## Example: AI Self-Audit and Collapse (K-pop Task, Claude)

---

### Prompt

```id="claude_prompt"
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

> The model initially produced highly optimistic results,
> but failed to detect its own evaluation flaws.

> Only after explicit prompting did it identify:

* data leakage (pre-CV SMOTE)
* label corruption (~38% false positives)
* invalid evaluation metrics

> Performance collapsed after correction:

* PR-AUC ≈ 0.74 → ≈ 0.034

---

### Key Insight

> AI systems can sometimes identify their own errors,
> but only under explicit and structured audit.

This example illustrates a different failure mode:

* not pure overconfidence
* but **conditional self-correction**

---

### Connection to Main Findings

> AI-generated workflows are not reliably self-validating.

Even when correct reasoning is possible,
it requires external prompting and structured verification.
