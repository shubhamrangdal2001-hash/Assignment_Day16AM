"""
Part C — Interview Ready: ab_test() Implementation
Day 16 Take-Home | PG Diploma AI-ML | IIT Gandhinagar
"""

import numpy as np
from scipy import stats


# ── Q2: ab_test() function ───────────────────────────────────────────────

def ab_test(control, treatment, alpha=0.05):
    """
    Runs an A/B test between control and treatment groups.

    Steps:
      1. Check normality using Shapiro-Wilk (only reliable for n <= 5000)
      2. Pick the right test:
         - Both normal → Welch's t-test
         - Any non-normal → Mann-Whitney U (non-parametric)
      3. Return test stats, decision, effect size, and 95% CI.
    """
    control   = np.array(control,   dtype=float)
    treatment = np.array(treatment, dtype=float)

    # --- Normality check ---
    def check_normality(data):
        if len(data) < 3:
            return False   # too small to test
        _, p = stats.shapiro(data)
        return p > 0.05    # True means approximately normal

    control_normal   = check_normality(control)
    treatment_normal = check_normality(treatment)
    both_normal      = control_normal and treatment_normal

    # --- Choose and run test ---
    if both_normal:
        stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)  # Welch's
        test_used     = "Welch's t-test"
    else:
        stat, p_value = stats.mannwhitneyu(control, treatment, alternative='two-sided')
        test_used     = "Mann-Whitney U"

    reject_h0 = bool(p_value < alpha)

    # --- Effect size (Cohen's d for means) ---
    diff        = treatment.mean() - control.mean()
    pooled_std  = np.sqrt((control.std(ddof=1)**2 + treatment.std(ddof=1)**2) / 2)
    effect_size = diff / pooled_std if pooled_std > 0 else 0.0

    # --- 95% CI for difference in means ---
    se    = np.sqrt(control.var(ddof=1)/len(control) + treatment.var(ddof=1)/len(treatment))
    dof   = len(control) + len(treatment) - 2
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_95 = (round(diff - t_crit * se, 4), round(diff + t_crit * se, 4))

    return {
        "test_used"   : test_used,
        "statistic"   : round(float(stat), 4),
        "p_value"     : round(float(p_value), 4),
        "reject_H0"   : reject_h0,
        "effect_size" : round(float(effect_size), 4),
        "ci_95"       : ci_95,
        "control_normal"  : control_normal,
        "treatment_normal": treatment_normal,
    }


# ── Demo runs ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 55)
    print("Part C — ab_test() Demo")
    print("=" * 55)

    # Case 1: Normal data, clear difference
    ctrl  = np.random.normal(100, 15, 120)
    treat = np.random.normal(110, 15, 120)
    result = ab_test(ctrl, treat)
    print("\n[Case 1] Normal groups, real effect")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    # Case 2: Small, skewed samples → should pick Mann-Whitney
    ctrl2  = np.random.exponential(scale=5, size=20)
    treat2 = np.random.exponential(scale=7, size=20)
    result2 = ab_test(ctrl2, treat2)
    print("\n[Case 2] Non-normal (exponential) groups")
    for k, v in result2.items():
        print(f"  {k:<22}: {v}")

    # Case 3: Very small sample (edge case)
    result3 = ab_test([1, 2], [3, 4])
    print("\n[Case 3] Very small samples (edge case)")
    for k, v in result3.items():
        print(f"  {k:<22}: {v}")

    print("\n" + "=" * 55)
    print("Interview Q3 — Checklist when p=0.04 but effect_size=0.02")
    print("=" * 55)
    print(
        "\n1. What is the minimum detectable effect that actually matters to the business?\n"
        "   (An effect of 0.02 might be statistically significant but commercially worthless.)\n"
        "\n2. Was the sample size powered for this effect size, or are we just drowning noise\n"
        "   in a very large dataset where even trivial differences become 'significant'?\n"
        "\n3. What are the implementation costs and risks of shipping this change?\n"
        "   A near-zero lift rarely justifies engineering work, infra cost, or user disruption."
    )
