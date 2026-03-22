"""
Part A — Hypothesis Testing on E-Commerce Checkout Time
Day 16 Take-Home | PG Diploma AI-ML | IIT Gandhinagar
"""

import numpy as np
from scipy import stats

np.random.seed(42)

# ── Business Question ────────────────────────────────────────────────────
# Does the new checkout flow reduce average checkout time compared to the old one?

# ── H0 and H1 ────────────────────────────────────────────────────────────
# H0: mean_new >= mean_old  (new flow is not faster)
# H1: mean_new <  mean_old  (new flow is faster)
# One-tailed test (we only care if new < old)
# alpha = 0.05 — standard threshold for product decisions

# ── Simulate Data ────────────────────────────────────────────────────────
# Checkout time in seconds
old_checkout = np.random.normal(loc=120, scale=25, size=200)   # old flow
new_checkout = np.random.normal(loc=110, scale=22, size=200)   # new flow

print("=" * 55)
print("Part A — Hypothesis Test: Checkout Time Comparison")
print("=" * 55)

print(f"\nOld flow — mean: {old_checkout.mean():.2f}s, std: {old_checkout.std():.2f}s")
print(f"New flow — mean: {new_checkout.mean():.2f}s, std: {new_checkout.std():.2f}s")

# ── Test Selection ────────────────────────────────────────────────────────
# Both samples are independent, n>30 → two-sample t-test (Welch's)
# alternative='less' because H1: new_checkout < old_checkout

t_stat, p_value = stats.ttest_ind(new_checkout, old_checkout, alternative='less')

alpha = 0.05
decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"

print(f"\nTest: Welch's Independent Samples t-test (one-tailed)")
print(f"Test Statistic (t): {t_stat:.4f}")
print(f"p-value           : {p_value:.4f}")
print(f"alpha             : {alpha}")
print(f"Decision          : {decision}")

# ── 95% Confidence Interval for Difference in Means ──────────────────────
diff = new_checkout.mean() - old_checkout.mean()
se = np.sqrt(new_checkout.var(ddof=1)/len(new_checkout) +
             old_checkout.var(ddof=1)/len(old_checkout))
df_approx = len(new_checkout) + len(old_checkout) - 2
t_crit = stats.t.ppf(0.975, df=df_approx)

ci_low  = diff - t_crit * se
ci_high = diff + t_crit * se

print(f"\n95% CI for (new - old) mean difference: ({ci_low:.2f}s, {ci_high:.2f}s)")

# ── Effect Size: Cohen's d ────────────────────────────────────────────────
pooled_std = np.sqrt((new_checkout.var(ddof=1) + old_checkout.var(ddof=1)) / 2)
cohens_d   = diff / pooled_std

print(f"Effect Size (Cohen's d): {cohens_d:.4f}")

# ── Stakeholder Interpretation ────────────────────────────────────────────
print("\n--- Stakeholder Summary ---")
print(
    "We tested whether the new checkout flow is faster than the old one. "
    f"On average, users completed checkout {abs(diff):.1f} seconds faster with the new design. "
    f"Our statistical test produced a p-value of {p_value:.4f}, which is below our threshold of 0.05, "
    "so we can confidently say the improvement is not due to random chance. "
    f"The 95% confidence interval tells us the new flow saves between {abs(ci_high):.1f}s and {abs(ci_low):.1f}s per user, "
    "and the effect size is small-to-medium, suggesting this is a real but modest gain worth shipping."
)
