## Example: AI-Generated ML Pipeline (K-pop Task)

---

### Prompt

```id="prompt_final"
You are a professional data scientist.

I have two K-pop datasets:
1. single_album_track_data.csv — full discography of 273 K-pop artists with audio features
2. KPopHits2021.csv — list of K-pop hit songs

Your task:
1. Explore the data (EDA)
2. Define a "hit song"
3. Build a model to predict hit songs
4. Report performance metrics
5. Explain what drives hit songs

Work through this step by step like a senior data scientist.
```

---

### AI-Generated Output (verbatim excerpt)

> “I addressed the class imbalance with SMOTE (synthetic oversampling), then trained three models with 5-fold cross-validation. Random Forest and Gradient Boosting both hit ROC-AUC ~0.9997 on the balanced data. The honest metric is PR-AUC on the original imbalanced data: 0.74 — **very strong** for a 0.37% positive class.”

---

### Audit

> PR-AUC 0.74 with a 0.37% positive rate is unusually high, and ROC-AUC ~1.0 further suggested evaluation issues.
> Root cause: SMOTE was applied before cross-validation, causing synthetic samples to leak into validation folds.

> Executing the pipeline confirmed the inflation (PR-AUC ≈ 0.999).
> After correction (SMOTE inside each training fold):
> CV PR-AUC ≈ 0.18 → Held-out PR-AUC ≈ 0.11.

> The pipeline was technically plausible but produced invalid results due to a subtle design flaw — illustrating how evaluation design, not predictive signal, can dominate reported performance.
