import math

from episode_qc.scoring import (
    Severity,
    robust_stats,
    severity_from_z,
    worse_severity,
)


def test_robust_stats_basic():
    stats = robust_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.median == 3.0
    assert stats.n == 5
    assert stats.mad > 0


def test_robust_stats_z_score_of_median_is_zero():
    stats = robust_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(stats.z(3.0)) < 1e-9


def test_robust_stats_degenerate_population_zero_for_exact_match():
    stats = robust_stats([0.0, 0.0, 0.0, 0.0])
    assert stats.mad == 0.0
    assert stats.z(0.0) == 0.0


def test_robust_stats_degenerate_population_infinite_for_any_deviation():
    stats = robust_stats([0.0, 0.0, 0.0, 0.0])
    assert math.isinf(stats.z(0.5))


def test_severity_bands():
    assert severity_from_z(0.0) == Severity.NONE
    assert severity_from_z(4.5, review_z=4.0, fail_z=7.0) == Severity.REVIEW
    assert severity_from_z(8.0, review_z=4.0, fail_z=7.0) == Severity.FAIL
    assert severity_from_z(-8.0, review_z=4.0, fail_z=7.0) == Severity.FAIL  # symmetric


def test_worse_severity_ordering():
    assert worse_severity(Severity.NONE, Severity.REVIEW) == Severity.REVIEW
    assert worse_severity(Severity.REVIEW, Severity.FAIL) == Severity.FAIL
    assert worse_severity(Severity.NONE, Severity.NONE) == Severity.NONE
