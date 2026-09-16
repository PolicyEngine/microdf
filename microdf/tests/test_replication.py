"""Variance estimation from replicate weights.

Recomputing a statistic once per replicate weight vector and measuring the
spread gives a variance estimate for statistics whose analytic variance is
awkward, such as the Gini coefficient or a quantile.
"""

import numpy as np
import pytest

from microdf import MicroSeries, replicate_standard_error, replicate_variance


@pytest.fixture
def series_and_replicates():
    rng = np.random.default_rng(0)
    values = rng.lognormal(mean=10, sigma=1.0, size=500)
    weights = np.full(500, 40.0)
    replicates = weights[:, None] * rng.poisson(1.0, size=(500, 200))
    return MicroSeries(values, weights=weights), replicates


def test_matches_the_analytic_standard_error_of_a_weighted_mean(
    series_and_replicates,
):
    """The mean has a closed form, so it is the case we can check exactly."""
    series, replicates = series_and_replicates

    replicate_se = replicate_standard_error(
        series, lambda s: s.mean(), replicates, method="bootstrap"
    )

    values = np.asarray(series)
    analytic_se = np.sqrt(np.var(values, ddof=1) / len(values))

    assert replicate_se == pytest.approx(analytic_se, rel=0.15)


def test_works_for_statistics_with_no_analytic_variance(
    series_and_replicates,
):
    """The point of the method: Gini and quantiles come out like anything else."""
    series, replicates = series_and_replicates

    for statistic in (lambda s: s.gini(), lambda s: s.median()):
        se = replicate_standard_error(
            series, statistic, replicates, method="bootstrap"
        )
        assert se > 0
        assert np.isfinite(se)


@pytest.mark.parametrize(
    "method,expected_factor",
    [
        ("jackknife", 199 / 200),
        ("brr", 1 / 200),
        ("bootstrap", 1 / 200),
        ("successive-difference", 4 / 200),
    ],
)
def test_each_method_applies_its_own_scale(
    series_and_replicates, method, expected_factor
):
    """The scale factor is what distinguishes the replication schemes."""
    series, replicates = series_and_replicates

    variance = replicate_variance(
        series, lambda s: s.mean(), replicates, method=method
    )
    reference = replicate_variance(
        series, lambda s: s.mean(), replicates, method="brr"
    )

    assert variance == pytest.approx(reference * expected_factor * 200, rel=1e-9)


def test_fay_requires_and_uses_its_constant(series_and_replicates):
    series, replicates = series_and_replicates

    with pytest.raises(ValueError, match="requires fay_k"):
        replicate_variance(series, lambda s: s.mean(), replicates, method="fay")

    fay = replicate_variance(
        series, lambda s: s.mean(), replicates, method="fay", fay_k=0.5
    )
    brr = replicate_variance(
        series, lambda s: s.mean(), replicates, method="brr"
    )

    # 1 / (R (1 - k)^2) against 1 / R, so a factor of four at k = 0.5.
    assert fay == pytest.approx(brr * 4, rel=1e-9)


def test_rejects_input_that_cannot_be_right(series_and_replicates):
    series, replicates = series_and_replicates

    with pytest.raises(ValueError, match="2-dimensional"):
        replicate_variance(series, lambda s: s.mean(), np.ones(500))

    with pytest.raises(ValueError, match="rows but the series has"):
        replicate_variance(series, lambda s: s.mean(), np.ones((499, 10)))

    with pytest.raises(ValueError, match="At least two"):
        replicate_variance(series, lambda s: s.mean(), np.ones((500, 1)))

    with pytest.raises(ValueError, match="Unknown method"):
        replicate_variance(
            series, lambda s: s.mean(), replicates, method="nonsense"
        )
