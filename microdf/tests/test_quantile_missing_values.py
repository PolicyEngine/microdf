import microdf as mdf
import numpy as np
import pandas as pd
import pytest


def test_quantile_skips_nan():
    """NaN weight must not inflate the cumulative distribution.

    Dropping a NaN row should give the same answer as never having had
    it: the inverse-CDF quantile of [1, nan, 3] equals that of [1, 3].
    """
    with_nan = mdf.MicroSeries([1.0, np.nan, 3.0], weights=[1, 1, 1])
    without_nan = mdf.MicroSeries([1.0, 3.0], weights=[1, 1])
    assert with_nan.median() == without_nan.median()
    assert with_nan.quantile(0.5) == without_nan.quantile(0.5)

    q = [0.25, 0.5, 0.75]
    np.testing.assert_array_equal(
        mdf.MicroSeries([1.0, np.nan, 3.0, 5.0], weights=[1, 1, 1, 1]).quantile(q),
        mdf.MicroSeries([1.0, 3.0, 5.0], weights=[1, 1, 1]).quantile(q),
    )


def test_quantile_skipna_false_propagates_nan():
    """Skipna=False returns NaN when any value is NaN, like mean/var."""
    s = mdf.MicroSeries([1.0, np.nan, 3.0], weights=[1, 1, 1])
    assert np.isnan(s.quantile(0.5, skipna=False))
    assert np.isnan(s.median(skipna=False))
    assert s.quantile([0.25, 0.75], skipna=False).isna().all()


def test_quantile_all_nan_returns_nan():
    s = mdf.MicroSeries([np.nan, np.nan], weights=[1, 1])
    assert np.isnan(s.median())


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("q", [-0.1, 1.1, [0.5, 1.1]])
def test_quantile_validates_bounds_with_missing_values(q, skipna):
    series = mdf.MicroSeries([1.0, np.nan, 3.0], weights=[1, 7, 1])
    with pytest.raises(AssertionError, match="quantiles should be in"):
        series.quantile(q, skipna=skipna)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("multiple_keys", [False, True])
def test_grouped_quantiles_preserve_missing_groups(skipna, multiple_keys):
    series = mdf.MicroSeries(
        [1.0, np.nan, 3.0, 5.0, np.nan, np.nan], weights=[1, 1, 1, 1, 1, 1]
    )
    groups = ["a", "a", "b", "b", "c", "c"]
    keys = [groups, [1, 1, 2, 2, 3, 3]] if multiple_keys else groups
    grouped = series.groupby(keys)
    quantiles = [0.25, 0.75]
    result = grouped.quantile(quantiles, skipna=skipna)
    # Scalar calls retain every group. Vector calls must retain the same
    # groups, including the partial-NaN and all-NaN groups.
    for quantile in quantiles:
        pd.testing.assert_series_equal(
            result.xs(quantile, level=-1),
            grouped.quantile(quantile, skipna=skipna),
        )
    first_group = 1.0 if skipna else np.nan
    np.testing.assert_allclose(
        result.to_numpy(),
        [first_group, first_group, 3.0, 5.0, np.nan, np.nan],
        equal_nan=True,
    )


def test_grouped_quantiles_preserve_repeated_requests():
    series = mdf.MicroSeries([1.0, np.nan, 3.0, 5.0], weights=[1, 1, 1, 1])
    result = series.groupby(["a", "a", "b", "b"]).quantile([0.5, 0.5], skipna=False)
    assert result.index.tolist() == [("a", 0.5), ("a", 0.5), ("b", 0.5), ("b", 0.5)]
    np.testing.assert_allclose(
        result.to_numpy(), [np.nan, np.nan, 3.0, 3.0], equal_nan=True
    )
