"""Statistics primitives for the validation harness.

We deliberately reimplement these in pure stdlib + numpy so the harness has
zero hidden dependencies and so the math is auditable in one screen.

  - bootstrap_ci(values, ci=0.95, n_bootstrap=10_000, seed=0)
        Percentile bootstrap CI over `mean(values)`.

  - paired_permutation_test(a, b, n_permutations=10_000, seed=0)
        Exact paired permutation when 2^n <= n_permutations,
        Monte Carlo otherwise. Two-sided p-value on mean(a) - mean(b).

  - cohens_d(a, b)
        Paired Cohen's d on the difference vector.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _as_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D sequence, got shape {arr.shape}")
    return arr


def bootstrap_ci(
    values: Sequence[float],
    *,
    ci: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean.

    Returns (point_mean, ci_low, ci_high). Empty input returns (0, 0, 0).
    """
    arr = _as_array(values)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0,1), got {ci}")
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return float(arr.mean()), low, high


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation test on mean(a) - mean(b).

    Uses exact enumeration when 2**n <= n_permutations; otherwise samples
    `n_permutations` random sign vectors. Returns the two-sided p-value.
    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    n = arr_a.size
    if n == 0:
        return 1.0
    diffs = arr_a - arr_b
    observed = float(diffs.mean())

    abs_obs = abs(observed)
    if 2 ** n <= n_permutations:
        signs = np.array(
            [[1 if (mask >> i) & 1 else -1 for i in range(n)] for mask in range(2 ** n)],
            dtype=float,
        )
        means = (signs * diffs).mean(axis=1)
        n_total = int(signs.shape[0])
        n_extreme = int(np.sum(np.abs(means) >= abs_obs - 1e-12))
        return n_extreme / n_total

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    means = (signs * diffs).mean(axis=1)
    n_extreme = int(np.sum(np.abs(means) >= abs_obs - 1e-12))
    # Add 1/(n+1) correction so a true-null observation never reports p=0.
    return (n_extreme + 1) / (n_permutations + 1)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d = mean(diff) / std(diff, ddof=1)."""
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.size < 2:
        return 0.0
    diff = arr_a - arr_b
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0 or math.isnan(sd):
        return 0.0
    return float(diff.mean() / sd)
