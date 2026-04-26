## Audit Case: Realistic but Fragile Healthcare Modeling

---

### Prompt

```text
You are a professional data scientist.

I have a healthcare dataset for predicting heart disease.

Perform a realistic end-to-end data science workflow:
- inspect data quality
- perform EDA
- define the target
- build a baseline model
- evaluate conservatively
- identify failure modes```

---

### AI-Generated Output

The initial analysis was relatively conservative.

It identified:

multiple source cohorts/sites
heavy missingness
invalid values such as chol = 0
site prevalence differences
likely limitations for clinical deployment

The model reported strong random-split performance:

| Metric  | Approx. Value |
| ------- | ------------: |
| ROC-AUC |         ~0.90 |
| PR-AUC  |         ~0.91 |


---
### Follow-up Audit

A stricter audit showed that the headline performance was likely inflated by:

site-level prevalence differences
non-random missingness
chol = 0 coding artifacts
downstream workup variables such as oldpeak, exang, and thalch
inconsistent target meaning across cohorts

The audit found that a model using only site information could already reach meaningful discrimination, indicating that part of the reported performance came from dataset structure rather than clinical signal.

---

### Key Insight

Even realistic and clinically cautious AI-generated analyses can remain fragile without structured validation.

This case differs from the K-pop examples:

the initial answer was not wildly overconfident
the model identified many real limitations
but deeper audit still revealed serious deployment risks

---
### Connection to Main Findings

This healthcare case supports the broader conclusion:

AI-generated ML workflows can appear careful and realistic, while still requiring external audit to detect hidden distribution shift, leakage, and target-definition problems.

The model should be treated as an educational baseline, not a deployment-ready clinical system.
