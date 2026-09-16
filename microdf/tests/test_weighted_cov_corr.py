import warnings

import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def replicated_moments(x, y, weights, ddof=1):
    """Independent frequency-weight oracle: expand to an ordinary sample."""
    repeated_x = np.repeat(np.asarray(x, dtype=float), weights)
    repeated_y = np.repeat(np.asarray(y, dtype=float), weights)
    return (
        np.cov(repeated_x, repeated_y, ddof=ddof)[0, 1],
        np.corrcoef(repeated_x, repeated_y)[0, 1],
    )


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_cov_corr_match_replicated_frequency_sample(ddof):
    x, y, weights = [1, 4, 8], [5, 2, 9], [1, 3, 2]
    left = mdf.MicroSeries(x, weights=weights)
    right = pd.Series(y)
    expected_cov, expected_corr = replicated_moments(x, y, weights, ddof)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert left.cov(right, ddof=ddof) == pytest.approx(expected_cov)
        assert left.corr(right, ddof=ddof) == pytest.approx(expected_corr)
    assert not any("unweighted" in str(item.message).lower() for item in caught)


def test_cov_corr_align_indices_and_use_only_left_weights():
    left = mdf.MicroSeries([1, 4, 8], index=["a", "b", "c"], weights=[1, 3, 2])
    right = mdf.MicroSeries(
        [9, 5, 2, 100], index=["c", "a", "b", "d"], weights=[99, 1, 1, 9]
    )
    expected_cov, expected_corr = replicated_moments([1, 4, 8], [5, 2, 9], [1, 3, 2])
    assert left.cov(right) == pytest.approx(expected_cov)
    assert left.corr(right) == pytest.approx(expected_corr)
    assert right.weights.tolist() == [99, 1, 1, 9]


def test_cov_corr_use_same_pairwise_nonmissing_sample():
    left = mdf.MicroSeries(
        [1, np.nan, 5, 9, 20], index=list("abcde"), weights=[1, 9, 2, 3, 7]
    )
    right = pd.Series([2, 4, np.nan, 8, 30], index=list("abcdf"))
    expected_cov, expected_corr = replicated_moments([1, 9], [2, 8], [1, 3])
    assert left.cov(right) == pytest.approx(expected_cov)
    assert left.corr(right) == pytest.approx(expected_corr)
    assert np.isnan(left.cov(right, skipna=False))
    assert np.isnan(left.corr(right, skipna=False))


@pytest.mark.parametrize("same_index", [True, False])
def test_cov_corr_follow_pandas_duplicate_index_alignment(same_index):
    left = mdf.MicroSeries([1, 4, 7], index=["a", "a", "b"], weights=[1, 3, 2])
    if same_index:
        right = pd.Series([2, 3, 8], index=["a", "a", "b"])
        x, y, weights = [1, 4, 7], [2, 3, 8], [1, 3, 2]
    else:
        right = pd.Series([2, 5, 8], index=["a", "b", "b"])
        # The shared a/b labels join, repeating the left row weight per pair.
        x, y, weights = [1, 4, 7, 7], [2, 2, 5, 8], [1, 3, 2, 2]
    expected_cov, expected_corr = replicated_moments(x, y, weights)
    assert left.cov(right) == pytest.approx(expected_cov)
    assert left.corr(right) == pytest.approx(expected_corr)


def test_zero_weight_rows_do_not_enter_pairwise_sample():
    left = mdf.MicroSeries([1, np.nan, 5, 1000], weights=[2, 0, 1, 0])
    right = pd.Series([3, np.nan, 9, -1000])
    expected_cov, expected_corr = replicated_moments([1, 5], [3, 9], [2, 1])
    assert left.cov(right, skipna=False) == pytest.approx(expected_cov)
    assert left.corr(right, skipna=False) == pytest.approx(expected_corr)
    assert np.isnan(left.cov(right, min_periods=3))
    assert np.isnan(left.corr(right, min_periods=3))


def test_min_periods_counts_usable_rows_separately_from_frequency_weight():
    left = mdf.MicroSeries([1, 5], weights=[10, 20])
    right = pd.Series([3, 9])
    expected_cov, expected_corr = replicated_moments([1, 5], [3, 9], [10, 20], ddof=0)
    # Existing pandas positional arguments retain their order.
    assert left.cov(right, 2, 0) == pytest.approx(expected_cov)
    assert left.corr(right, "pearson", 2, ddof=0) == pytest.approx(expected_corr)
    assert np.isnan(left.cov(right, min_periods=3))
    assert np.isnan(left.corr(right, min_periods=3))


@pytest.mark.parametrize(
    "values,other,weights",
    [([], [], []), ([np.nan], [1], [3]), ([1], [2], [0]), ([1], [2], [1])],
)
def test_cov_corr_return_nan_when_sample_is_insufficient(values, other, weights):
    left = mdf.MicroSeries(values, weights=weights, dtype=float)
    right = pd.Series(other, dtype=float)
    assert np.isnan(left.cov(right))
    assert np.isnan(left.corr(right))


def test_cov_corr_no_index_overlap():
    left = mdf.MicroSeries([1, 2], index=["a", "b"], weights=[1, 2])
    right = pd.Series([3, 4], index=["c", "d"])
    assert np.isnan(left.cov(right))
    assert np.isnan(left.corr(right))


def test_cov_corr_constant_and_frequency_singleton():
    left = mdf.MicroSeries([4, 4], weights=[2, 3])
    right = pd.Series([1, 5])
    assert left.cov(right) == 0
    assert np.isnan(left.corr(right))
    singleton = mdf.MicroSeries([4], weights=[3])
    assert singleton.cov(pd.Series([2])) == 0
    assert np.isnan(singleton.corr(pd.Series([2])))
    assert np.isnan(singleton.cov(pd.Series([2]), ddof=3))
    assert np.isnan(singleton.corr(pd.Series([2]), ddof=3))


def test_cov_corr_nullable_numeric_data():
    left = mdf.MicroSeries(pd.Series([1, pd.NA, 4], dtype="Int64"), weights=[2, 9, 3])
    right = pd.Series([3, 8, 7], dtype="Float64")
    expected_cov, expected_corr = replicated_moments([1, 4], [3, 7], [2, 3])
    assert left.cov(right) == pytest.approx(expected_cov)
    assert left.corr(right) == pytest.approx(expected_corr)


@pytest.mark.parametrize("method", ["spearman", "kendall", lambda x, y: 1.0])
def test_non_pearson_methods_are_explicitly_unsupported(method):
    left = mdf.MicroSeries([1, 2, 3], weights=[1, 2, 3])
    with pytest.raises(ValueError, match="pearson"):
        left.corr(pd.Series([4, 2, 5]), method=method)


@pytest.mark.parametrize("weights", [[1, -1], [1, np.nan], [1, np.inf]])
def test_cov_corr_reject_invalid_frequency_weights(weights):
    left = mdf.MicroSeries([1, 2], weights=weights)
    for method in (left.cov, left.corr):
        with pytest.raises(ValueError, match="weights"):
            method(pd.Series([2, 4]))


def test_binary_statistics_not_in_dataframe_aggregation_factories():
    assert "cov" not in mdf.MicroSeries.FUNCTIONS
    assert "corr" not in mdf.MicroSeries.FUNCTIONS


@pytest.mark.parametrize(
    "x,y",
    [([0.1, 0.1], [0, 1]), ([0, 1], [0.1, 0.1]), ([0.1, 0.1], [0.1, 0.1])],
)
@pytest.mark.parametrize("with_filtered_rows", [False, True])
def test_corr_exact_decimal_constants_return_nan(x, y, with_filtered_rows):
    # Unequal weights can round the mean away from the identical 0.1 values.
    # Constant detection must inspect usable observations before centering.
    weights = [1, 2]
    if with_filtered_rows:
        x = x + [9, np.nan]
        y = y + [7, 4]
        weights = weights + [0, 3]
    left = mdf.MicroSeries(x, weights=weights)
    assert np.isnan(left.corr(pd.Series(y)))


def test_corr_does_not_treat_nearby_distinct_values_as_constant():
    x = [0.1, np.nextafter(0.1, np.inf)]
    left = mdf.MicroSeries(x, weights=[1, 2])
    assert np.isfinite(left.corr(pd.Series([0, 1])))
    assert np.isfinite(mdf.MicroSeries([0, 1], weights=[1, 2]).corr(pd.Series(x)))
