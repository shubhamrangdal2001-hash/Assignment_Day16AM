"""
Part B — Multiple Comparisons & Bonferroni Correction
Day 16 Take-Home | PG Diploma AI-ML | IIT Gandhinagar
"""

import numpy as np
from scipy import stats

np.random.seed(0)

print("=" * 55)
print("Part B — Multiple Comparison Problem (p-hacking)")
print("=" * 55)

# ── Theoretical probability of at least one false positive ───────────────
# If we run k independent tests each at alpha = 0.05:
# P(at least one FP) = 1 - (1 - 0.05)^k

k     = 20
alpha = 0.05
p_at_least_one_fp = 1 - (1 - alpha) ** k

print(f"\nk = {k} independent tests, alpha = {alpha}")
print(f"Theoretical P(at least 1 false positive) = 1 - (1-{alpha})^{k}")
print(f"  = {p_at_least_one_fp:.4f}  (~{p_at_least_one_fp*100:.1f}%)")

# ── Simulation to verify ─────────────────────────────────────────────────
n_simulations = 10_000
false_positive_count = 0

for _ in range(n_simulations):
    # All 20 tests are under H0 (no real effect) → any rejection is a FP
    p_values = [
        stats.ttest_ind(
            np.random.normal(0, 1, 50),
            np.random.normal(0, 1, 50)
        ).pvalue
        for _ in range(k)
    ]
    if any(p < alpha for p in p_values):
        false_positive_count += 1

sim_rate = false_positive_count / n_simulations
print(f"\nSimulation ({n_simulations:,} runs): P(at least 1 FP) = {sim_rate:.4f}  (~{sim_rate*100:.1f}%)")
print("  → Simulation matches theory closely ✓")

# ── Bonferroni Correction ────────────────────────────────────────────────
alpha_bonferroni = alpha / k

print(f"\nBonferroni corrected alpha = {alpha} / {k} = {alpha_bonferroni}")
print(f"Original alpha  : {alpha}")
print(f"Corrected alpha : {alpha_bonferroni}")
print(
    f"\nWith Bonferroni correction, each individual test must clear a much stricter"
    f" threshold ({alpha_bonferroni}) to be called significant, keeping the"
    f" family-wise error rate at {alpha}."
)
