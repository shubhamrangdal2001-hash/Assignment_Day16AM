# Day 16 Take-Home Assignment
## Hypothesis Testing & Confidence Intervals
### PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Gitlink: https://github.com/shubhamrangdal2001-hash/Assignment_Day16AM.git
---

## Overview

This repository contains the solution to the Day 16 AM session take-home assignment covering hypothesis testing, confidence intervals, p-hacking, and A/B test implementation.

---

## Files

| File | Description |
|------|-------------|
| `part_a_hypothesis_test.py` | Business hypothesis test — e-commerce checkout time A/B test |
| `part_b_multiple_comparisons.py` | p-hacking simulation + Bonferroni correction |
| `part_c_ab_test.py` | `ab_test()` function + Interview Q answers |
| `part_d_ai_task.py` | AI-augmented task: Type I/II errors in fraud detection |
| `Day16_TakeHome_Solution.docx` | Full written solution document |

---

## How to Run

### Requirements
```bash
pip install numpy scipy
```

### Run each part
```bash
python part_a_hypothesis_test.py
python part_b_multiple_comparisons.py
python part_c_ab_test.py
python part_d_ai_task.py
```

---

## Part Summaries

### Part A
- **Business question**: Does the new checkout flow reduce average checkout time?
- **Test used**: Welch's one-tailed t-test (H1: new < old)
- **Result**: t = -3.15, p = 0.0009 → Reject H0
- **95% CI**: (-11.52s, -2.67s)  |  Cohen's d = -0.315

### Part B
- Running 20 tests at α=0.05 gives **64.2% chance of at least one false positive**
- Simulation confirms: ~64.4% (matches theory)
- Bonferroni corrected α = **0.0025**

### Part C
- `ab_test()` handles normality checks, selects Welch's t-test or Mann-Whitney automatically, returns statistic, p_value, reject_H0, effect_size, ci_95
- Three questions to ask before shipping a p=0.04, effect_size=0.02 result

### Part D
- AI correctly maps Type I/II errors to the precision-recall tradeoff in fraud detection
- Minor gap: causal direction of the tradeoff could be stated more explicitly

---


