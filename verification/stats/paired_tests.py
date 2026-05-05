from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(
    results_a: list[bool],
    results_b: list[bool],
) -> tuple[float, float, str]:
    """Paired t-test on per-item correctness differences.

    Args:
        results_a: Per-item correctness for method A (list of True/False)
        results_b: Per-item correctness for method B (list of True/False)
    Returns:
        (t_statistic, p_value, outcome)
        outcome: "Win" if p<0.05 and A > B,
                 "Tie" if p >= 0.05,
                 "Lose" if p<0.05 and A < B
    """
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    diff = a - b

    t_stat, p_val = stats.ttest_rel(a, b)

    if p_val < 0.05:
        mean_diff = diff.mean()
        if mean_diff > 0:
            outcome = "Win"
        elif mean_diff < 0:
            outcome = "Lose"
        else:
            outcome = "Tie"
    else:
        outcome = "Tie"

    return t_stat, p_val, outcome


def wilcoxon_test(
    results_a: list[bool],
    results_b: list[bool],
) -> tuple[float, float, str]:
    """Wilcoxon signed-rank test (non-parametric paired alternative).

    Used when sample size < 30 or distribution is non-normal.
    """
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    diff = a - b

    # Remove zero differences (required for Wilcoxon)
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) == 0:
        return 0.0, 1.0, "Tie"

    stat, p_val = stats.wilcoxon(diff_nonzero)

    if p_val < 0.05:
        mean_diff = diff.mean()
        if mean_diff > 0:
            outcome = "Win"
        elif mean_diff < 0:
            outcome = "Lose"
        else:
            outcome = "Tie"
    else:
        outcome = "Tie"

    return float(stat), float(p_val), outcome