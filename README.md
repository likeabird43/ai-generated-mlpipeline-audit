# ai-generated-mlpipeline-audit
Auditing the AI-generated machine learning pipeline under label noise, class imbalance, and evaluation leakage.

# When Evaluation Fails: K-pop Hit Prediction Case Study

## Problem

Can we predict K-pop hit songs using audio features?

This project investigates whether machine learning models can reliably identify hit songs — and more importantly, whether the problem itself is well-defined.

---

## Key Finding

Models appear highly accurate under flawed evaluation,
but performance collapses under proper validation.

---

## Main Results

| Label                  | Positive Rate | Wrong PR-AUC | Correct PR-AUC | Held-out PR-AUC |
|-----------------------|--------------|--------------|----------------|-----------------|
| Strict (artist+title) | 2.1%         | 0.999        | 0.11           | 0.18            |
| Loose (title only)    | 7.7%         | 0.987        | 0.18           | 0.19            |
| Spotify proxy         | 10.0%        | 0.971        | 0.25           | 0.26            |

---

## What Happened?

### 1. Evaluation Failure

Improper evaluation (SMOTE before cross-validation) produced near-perfect results:

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

But these patterns:

- change across label definitions
- do not produce strong predictive power

---

## Final Insight

Model performance is highly sensitive to:

- evaluation setup
- label definition
- dataset assumptions

> Apparent improvements do not necessarily reflect real predictive ability.

---

## Conclusion

This is not a successful hit prediction model.

Instead, this project demonstrates that:

> AI models can produce plausible results even when the problem itself is poorly defined.

---

## Code

See `final_music_audit.py` for full implementation.

---

## Reproducibility

```bash
pip install -r requirements.txt
python final_music_audit.py
