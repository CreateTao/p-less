from __future__ import annotations

import numpy as np


def bootstrap_ci(
    results: list[bool],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for accuracy.

    Args:
        results: Per-item correctness (list of True/False)
        n_resamples: Number of bootstrap resamples
        ci_level: Confidence level (default 0.95)
    Returns:
        (accuracy, ci_lower, ci_upper)
    """
    data = np.array(results, dtype=float)
    accuracy = data.mean()

    boot_accs = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_accs.append(sample.mean())

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_accs, alpha / 2 * 100)
    ci_upper = np.percentile(boot_accs, (1 - alpha / 2) * 100)

    return float(accuracy), float(ci_lower), float(ci_upper)


def bootstrap_diff_ci(
    results_a: list[bool],
    results_b: list[bool],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the difference in accuracy between two methods.

    Returns:
        (mean_diff, ci_lower, ci_upper) for the difference A - B
    """
    a = np.array(results_a, dtype=float)
    b = np.array(results_b, dtype=float)
    mean_diff = a.mean() - b.mean()

    boot_diffs = []
    for _ in range(n_resamples):
        indices = np.random.choice(len(a), size=len(a), replace=True)
        diff = a[indices].mean() - b[indices].mean()
        boot_diffs.append(diff)

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_diffs, alpha / 2 * 100)
    ci_upper = np.percentile(boot_diffs, (1 - alpha / 2) * 100)

    return float(mean_diff), float(ci_lower), float(ci_upper)