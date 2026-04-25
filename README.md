> An AI-generated workflow, when implemented with a flawed but common evaluation setup,
> produced PR-AUC of 0.999 on the same dataset where a corrected evaluation yielded 0.11.
> This project shows how that happens — and why it matters.

# When Evaluation Fails

## Auditing an AI-Generated ML Pipeline on Noisy Data

---

## Problem

This project does not aim to build a reliable K-pop hit prediction model.

Instead, it audits an AI-assisted machine learning workflow on noisy real-world data, demonstrating how plausible modeling steps can lead to unstable or misleading conclusions.

---

## Workflow

This project follows a workflow generated from structured prompts to an AI system (e.g., ChatGPT):

1. **Dataset inspection**  
   Identify data quality issues, leakage risks, and feasibility  

2. **Label definition**  
   Propose multiple definitions of "hit" and analyze label noise  

3. **Baseline modeling**  
   Build models and compare performance across labels  

4. **Critical audit**  
   Evaluate whether results are meaningful or misleading  

While the AI correctly identified many data limitations, the workflow still proceeded toward modeling and evaluation — reflecting a common pattern where analysis continues despite weak problem formulation.

---

## Key Finding

A commonly used modeling setup (**SMOTE + cross-validation applied incorrectly**) produced near-perfect PR-AUC (~0.97–0.999).

However, when corrected and evaluated on a proper hold-out set, performance dropped to realistic levels (~0.11–0.25).

This demonstrates that **evaluation design, rather than true predictive signal, can dominate model performance** in noisy real-world settings.

---

## Main Results (K-pop)

| Label                 | Positive Rate | Wrong PR-AUC | Correct PR-AUC | Held-out PR-AUC |
| --------------------- | ------------- | ------------ | -------------- | --------------- |
| Strict (artist+title) | 2.1%          | 0.999        | 0.11           | 0.18            |
| Loose (title only)    | 7.7%          | 0.987        | 0.18           | 0.20            |
| Spotify proxy         | 10.0%         | 0.971        | 0.25           | 0.26            |

---

## What Happened?

### 1. Evaluation Failure

Improper evaluation (SMOTE applied before cross-validation) produced near-perfect results:

> PR-AUC ≈ 0.99

But correct evaluation revealed the true performance:

> PR-AUC ≈ 0.11–0.25

---

### 2. Label Instability

Different definitions of “hit” produced different results:

- Strict label (clean but sparse)  
- Loose label (noisy but larger)  
- Spotify proxy (different concept entirely)  

> The model is not learning “hit songs” — it is learning how the label is defined.

---

### 3. Fragile Conclusions

Feature patterns appear meaningful:

- louder songs  
- less instrumental  
- less acoustic  

However, these patterns:

- change across label definitions  
- are sensitive to dataset assumptions  
- do not translate into strong predictive performance  

---

## Cross-Domain Validation (Healthcare)

To test whether this failure mode is specific to noisy music data,  
a similar audit was applied to a healthcare dataset for heart disease prediction.

Under a standard random train/test split, the model appeared strong:

- ROC-AUC: 0.92  
- PR-AUC: 0.91  
- Balanced accuracy: 0.85  

However, under a more realistic evaluation setting  
(**site-held-out validation across hospitals**), performance dropped and became unstable:

| Held-out cohort | ROC-AUC | PR-AUC | Balanced Accuracy |
|---|---:|---:|---:|
| Cleveland | 0.855 | 0.854 | 0.774 |
| Hungary | 0.895 | 0.846 | 0.820 |
| Switzerland | 0.747 | 0.973 | 0.740 |
| VA Long Beach | 0.724 | 0.852 | 0.665 |

This shows that even in a **more structured and clinically meaningful domain**,  
performance can appear strong under naive evaluation,  
but fail under realistic validation.

---

## Dataset Construction Notes

The K-pop hit dataset is derived from Apple Music playlists, transferred to Spotify, and manually supplemented.

In some cases, instrumental or cover versions were used as substitutes, introducing additional noise and inconsistency.

As a result, the "hit" label should be interpreted as a proxy rather than a ground-truth measure of commercial success.

---

## Final Insight

Across both domains:

- weakly defined problems (K-pop)  
- and seemingly well-defined problems (healthcare)  

AI-assisted workflows can produce results that appear meaningful,  
but fail under careful audit.

> The core issue is not the dataset alone,  
> but the tendency of workflows to proceed toward modeling  
> without sufficient validation of assumptions and evaluation design.

---

## Final Decision

The appropriate conclusion is not to improve the model,  
but to **stop and reframe the problem**.

This project shows that:

- evaluation design can dominate performance  
- labels and datasets can introduce hidden artifacts  
- strong metrics do not imply reliable models  

Therefore:

> This work is best interpreted as an audit of AI-assisted ML workflows,  
> rather than a predictive modeling solution.

---

## Code

- `final_music_audit.py` — K-pop audit  
- `healthcare_audit.py` — healthcare audit  

---

## Data Sources

Datasets are not included in this repository due to licensing and size constraints.

Please download the original datasets from Kaggle and place them in the following structure:

```text
data/
  kpopfullspotifydiscography/
    single_album_track_data.csv
  kpop_hits_all_years.csv
  spotify_tracks.csv
  heart_disease_uci.csv

---

## Reproducibility

```bash
pip install -r requirements.txt
python final_music_audit.py
python healthcare_audit.py

