## Audit Case: Realistic but Fragile Healthcare Modeling

---

### Prompt

You are a professional data scientist.

I have a healthcare dataset for predicting heart disease.

Perform a realistic end-to-end data science workflow:
- inspect data quality
- perform EDA
- define the target
- build a baseline model
- evaluate conservatively
- identify failure modes

---

### AI-Generated Output

The initial analysis appeared relatively conservative and clinically aware.

It identified:

- multiple source cohorts/sites  
- heavy missingness  
- invalid values such as `chol = 0`  
- site prevalence differences  
- likely limitations for clinical deployment  

The model reported strong random-split performance:

| Metric  | Approx. Value |
| ------- | ------------: |
| ROC-AUC |         ~0.90 |
| PR-AUC  |         ~0.91 |

---

### Follow-up Audit

A stricter audit showed that the headline performance was likely inflated by:

- site-level prevalence differences  
- non-random missingness  
- `chol = 0` coding artifacts  
- downstream workup variables such as `oldpeak`, `exang`, and `thalch`  
- inconsistent target meaning across cohorts  

In particular, strong performance could be reproduced using only the `dataset` (site) variable,
indicating that the model partially relies on cohort identity rather than clinical signal.

---

### Key Insight

Even realistic and clinically cautious AI-generated analyses can remain fragile,
as apparent performance may still rely on dataset artifacts rather than true signal.

This case differs from the K-pop examples:

- the initial answer was not wildly overconfident  
- the model identified many real limitations  
- but deeper audit still revealed serious deployment risks  

Both ChatGPT and Claude produced relatively conservative analyses,
but neither fully accounted for site-level leakage and dataset artifacts without explicit audit.

---

### Connection to Main Findings

This healthcare case supports the broader conclusion:

AI-generated ML workflows can appear careful and realistic,
while still requiring external audit to detect hidden distribution shift,
leakage, and target-definition problems.

The model should be treated as an educational baseline,
not as evidence of clinically reliable prediction.
