## Audit Case: Label Definition Sensitivity (K-pop Task)

This example illustrates how model performance changes depending on how "hit" is defined.

### Setup

Three label definitions were used:

- Strict: artist + title match  
- Loose: title-only match  
- Spotify proxy: popularity-based  

---

### Results

| Label   | PR-AUC | Baseline |
|--------|--------|----------|
| Strict | 0.114  | 0.024    |
| Loose  | 0.230  | 0.077    |
| Spotify| 0.584  | 0.311    |

---

### Audit

At first glance, performance appears to improve significantly.

However:

- label noise increases (Loose)
- problem definition shifts (Spotify)
- baseline also increases

> The apparent performance gain is largely driven by **label construction**, not model improvement.

---

### Key Insight

> Model performance is highly sensitive to how the target variable is defined.

---

### Connection to Main Findings

This example reinforces that:

- performance metrics alone are insufficient  
- label definition must be critically evaluated  

> Changing the label changes the problem.
