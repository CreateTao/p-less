from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TOSTResult:
    """Result of Two One-Sided Tests (TOST) for equivalence."""
    method_a: str
    method_b: str
    dataset: str
    model: str
    delta: float
    lower_test_pvalue: float
    upper_test_pvalue: float
    is_equivalent: bool
    mean_diff: float
    diff_ci_lower: float
    diff_ci_upper: float


def tost_equivalence_test(
    results_a: list[bool],
    results_b: list[bool],
    delta: float = 0.02,
    method_a_name: str = "",
    method_b_name: str = "",
    dataset_name: str = "",
    model_name: str = "",
) -> TOSTResult:
    """Two One-Sided Tests (TOST) for equivalence / non-degradation.

    Tests H0: |mean(A) - mean(B)| >= delta  vs  H1: |mean(A) - mean(B)| < delta
    If both one-sided tests reject at p<0.05, methods are equivalent within delta.

    This is the core statistical test for GPU non-degradation verification.
    """
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    diff = a - b
    mean_diff = diff.mean()

    # Lower one-sided test: H0: mean_diff <= -delta  vs  H1: mean_diff > -delta
    t_lower, p_lower = stats.ttest_1samp(diff, -delta, alternative="greater")

    # Upper one-sided test: H0: mean_diff >= delta  vs  H1: mean_diff < delta
    t_upper, p_upper = stats.ttest_1samp(diff, delta, alternative="less")

    # TOST passes if BOTH one-sided tests reject at alpha=0.05
    is_equivalent = (p_lower < 0.05) and (p_upper < 0.05)

    # Compute CI for reporting
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se

    return TOSTResult(
        method_a=method_a_name,
        method_b=method_b_name,
        dataset=dataset_name,
        model=model_name,
        delta=delta,
        lower_test_pvalue=float(p_lower),
        upper_test_pvalue=float(p_upper),
        is_equivalent=is_equivalent,
        mean_diff=float(mean_diff),
        diff_ci_lower=float(ci_lower),
        diff_ci_upper=float(ci_upper),
    )