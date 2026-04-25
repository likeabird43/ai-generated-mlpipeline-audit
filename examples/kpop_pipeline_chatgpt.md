## Example: Conservative but Incomplete Analysis (K-pop Task, ChatGPT)

---

### Prompt

```text
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

### AI-Generated Output (verbatim excerpt, ChatGPT)

> “A baseline logistic regression model produced:
> ROC-AUC ≈ 0.77
> PR-AUC ≈ 0.014
>
> The ROC-AUC says the model captures signal, but the very low PR-AUC means predicting hits from audio features alone is hard.”

---

### Audit

> The model produced conservative and realistic performance estimates,
> and correctly identified the limitations of audio-only features.

However, key issues were not fully addressed:

* incomplete label matching (only ~20–30 hits aligned)
* potential duplicate leakage
* lack of strict evaluation validation (e.g., group-aware splits)

> While not overconfident, the analysis was incomplete and did not identify deeper data and evaluation risks.

---

### Key Insight

> AI can produce reasonable high-level conclusions,
> but still miss critical methodological weaknesses without explicit auditing.
