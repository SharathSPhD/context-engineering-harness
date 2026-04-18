"""Statistics primitives — bootstrap CI, paired permutation, Cohen's d."""
from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.stats import bootstrap_ci, cohens_d, paired_permutation_test


def test_bootstrap_ci_contains_population_mean_for_iid_normal():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=0.5, scale=0.1, size=200).tolist()
    mean, low, high = bootstrap_ci(sample, ci=0.95, n_bootstrap=2000, seed=7)
    assert low < 0.5 < high
    assert low < mean < high


def test_bootstrap_ci_empty_returns_zeros():
    mean, low, high = bootstrap_ci([])
    assert (mean, low, high) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_invalid_level_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2, 0.3], ci=0.0)


def test_paired_permutation_exact_when_small():
    a = [1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0]
    p = paired_permutation_test(a, b, n_permutations=10_000)
    # 8 sign flips; only the all-positive flip matches the observed.
    # Two-sided ⇒ both extremes count: |+1| and |-1|, so 2/8 = 0.25.
    assert p == pytest.approx(0.25)


def test_paired_permutation_no_difference_high_p():
    a = [0.5, 0.6, 0.5, 0.6, 0.5]
    b = [0.5, 0.6, 0.5, 0.6, 0.5]
    p = paired_permutation_test(a, b, n_permutations=2000)
    assert p > 0.5


def test_paired_permutation_strong_difference_low_p():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, size=30)
    b = rng.normal(0.5, 0.05, size=30)
    p = paired_permutation_test(a.tolist(), b.tolist(), n_permutations=5000)
    assert p < 0.001


def test_paired_permutation_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_permutation_test([0.1, 0.2], [0.3])


def test_cohens_d_zero_for_identical_samples():
    a = [0.4, 0.5, 0.6]
    b = [0.4, 0.5, 0.6]
    assert cohens_d(a, b) == 0.0


def test_cohens_d_positive_when_treatment_better():
    a = [0.9, 0.85, 0.95, 0.92]
    b = [0.5, 0.55, 0.45, 0.50]
    d = cohens_d(a, b)
    assert d > 1.0


def test_cohens_d_too_few_samples_returns_zero():
    assert cohens_d([0.5], [0.5]) == 0.0
