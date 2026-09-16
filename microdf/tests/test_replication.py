"""Variance estimation from replicate weights.

Recomputing a statistic once per replicate weight vector and measuring the
spread gives a variance estimate for statistics whose analytic variance is
awkward, such as the Gini coefficient or a quantile.
"""

import numpy as np
import pandas as pd
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
    """The point of the method: Gini and quantiles come out like anything
    else."""
    series, replicates = series_and_replicates

    for statistic in (lambda s: s.gini(), lambda s: s.median()):
        se = replicate_standard_error(series, statistic, replicates, method="bootstrap")
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

    variance = replicate_variance(series, lambda s: s.mean(), replicates, method=method)
    reference = replicate_variance(series, lambda s: s.mean(), replicates, method="brr")

    assert variance == pytest.approx(reference * expected_factor * 200, rel=1e-9)


def test_fay_requires_and_uses_its_constant(series_and_replicates):
    series, replicates = series_and_replicates

    with pytest.raises(ValueError, match="requires fay_k"):
        replicate_variance(series, lambda s: s.mean(), replicates, method="fay")

    fay = replicate_variance(
        series, lambda s: s.mean(), replicates, method="fay", fay_k=0.5
    )
    brr = replicate_variance(series, lambda s: s.mean(), replicates, method="brr")

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
        replicate_variance(series, lambda s: s.mean(), replicates, method="nonsense")


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_replicates_preserve_categorical_values_and_metadata(dtype):
    series = MicroSeries(
        ["employed", "unemployed", "employed", None],
        weights=[1, 1, 1, 1],
        index=["a", "b", "c", "d"],
        name="employment",
        dtype=dtype,
    )
    replicates = np.array([[2, 2, 0, 0], [0, 0, 2, 2], [2, 0, 2, 0], [0, 2, 0, 2]])
    original = pd.Series(series).copy(deep=True)
    original_weights = series.weights.copy(deep=True)
    original_replicates = replicates.copy()

    def count(sample):
        assert sample.dtype == series.dtype
        assert sample.name == "employment"
        pd.testing.assert_index_equal(sample.index, series.index)
        return sample.count()

    # Full count is 3; replicate counts are 4, 2, 4, 2.
    assert replicate_variance(series, count, replicates, method="brr") == 1
    pd.testing.assert_series_equal(pd.Series(series), original)
    pd.testing.assert_series_equal(series.weights, original_weights)
    np.testing.assert_array_equal(replicates, original_replicates)


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_replicates_preserve_boolean_domain_counts(dtype):
    series = MicroSeries([True, False, True, False], weights=[1, 1, 1, 1], dtype=dtype)
    replicates = np.array([[2, 2, 0, 0], [0, 0, 2, 2], [2, 0, 2, 0], [0, 2, 0, 2]])
    # Complement counts are 0, 2, 2, 4, around the full-sample count of 2.
    assert (
        replicate_variance(
            series, lambda sample: (~sample).sum(), replicates, method="brr"
        )
        == 2
    )


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_replicates_preserve_exact_large_integer_categories(dtype):
    category = 2**53
    series = MicroSeries([category, category + 1], weights=[1, 2], dtype=dtype)
    replicates = np.array([[1, 1], [2, 2]])
    # Identical weights must keep the weighted category count exactly 1.
    assert (
        replicate_variance(
            series, lambda sample: (sample == category).sum(), replicates, method="brr"
        )
        == 0
    )


@pytest.mark.parametrize(
    "method,fay_k,full_sample_variance,replicate_mean_variance",
    [
        ("jackknife", None, 1088 / 3, 3136 / 9),
        ("brr", None, 544 / 3, 1568 / 9),
        ("bootstrap", None, 544 / 3, 1568 / 9),
        ("successive-difference", None, 2176 / 3, 6272 / 9),
        ("fay", 0.5, 2176 / 3, 6272 / 9),
    ],
)
@pytest.mark.parametrize("centering", ["full-sample", "replicate-mean"])
@pytest.mark.parametrize("api", ["variance", "standard_error", "series_method"])
def test_centering_uses_exact_nonlinear_replicate_estimates(
    method, fay_k, full_sample_variance, replicate_mean_variance, centering, api
):
    series = MicroSeries([1, 3], weights=[1, 1])
    replicates = np.array([[2, 1, 0], [0, 1, 2]])

    # Squared totals: full sample 16; replicates 4, 16, 36; replicate mean 56/3.
    # Squared deviations sum to 544 from 16 and 1568/3 from 56/3.
    def squared_total(sample):
        return sample.sum() ** 2

    expected = (
        full_sample_variance if centering == "full-sample" else replicate_mean_variance
    )
    kwargs = {"method": method, "fay_k": fay_k, "centering": centering}
    if api == "variance":
        observed = replicate_variance(series, squared_total, replicates, **kwargs)
    elif api == "standard_error":
        observed = replicate_standard_error(series, squared_total, replicates, **kwargs)
        expected = np.sqrt(expected)
    else:
        observed = series.replicate_standard_error(squared_total, replicates, **kwargs)
        expected = np.sqrt(expected)
    assert observed == pytest.approx(expected)


def test_default_centering_retains_full_sample_estimate():
    series = MicroSeries([1, 3], weights=[1, 1])
    replicates = np.array([[2, 1, 0], [0, 1, 2]])
    expected_variance = 544 / 3

    def squared_total(sample):
        return sample.sum() ** 2

    assert replicate_variance(
        series, squared_total, replicates, method="bootstrap"
    ) == pytest.approx(expected_variance)
    assert replicate_standard_error(
        series, squared_total, replicates, method="bootstrap"
    ) == pytest.approx(np.sqrt(expected_variance))
    assert series.replicate_standard_error(
        squared_total, replicates, method="bootstrap"
    ) == pytest.approx(np.sqrt(expected_variance))


@pytest.mark.parametrize("api", ["variance", "standard_error", "series_method"])
def test_rejects_unknown_centering_before_calling_statistic(api):
    series = MicroSeries([1, 3], weights=[1, 1])
    replicates = np.array([[2, 0], [0, 2]])

    def unexpected_statistic(sample):
        raise AssertionError("Invalid centering must be rejected first")

    with pytest.raises(ValueError, match="centering"):
        if api == "variance":
            replicate_variance(
                series, unexpected_statistic, replicates, centering="unknown"
            )
        elif api == "standard_error":
            replicate_standard_error(
                series, unexpected_statistic, replicates, centering="unknown"
            )
        else:
            series.replicate_standard_error(
                unexpected_statistic, replicates, centering="unknown"
            )


def test_replicate_weight_frames_use_positional_rows():
    series = MicroSeries([1, 3], weights=[1, 1], index=["a", "b"])
    replicates = pd.DataFrame([[2, 0], [0, 2]], index=["b", "a"])
    # Position-defined means are 1 and 3 around the full-sample mean of 2.
    assert (
        replicate_variance(
            series, lambda sample: sample.mean(), replicates, method="brr"
        )
        == 1
    )
