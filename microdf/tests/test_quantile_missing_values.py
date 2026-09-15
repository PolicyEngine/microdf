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


@pytest.mark.parametrize("quantiles", [[0.75, 0.25], [0.5, 0.5], []])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_grouped_quantiles_preserve_missing_multiple_keys(quantiles, skipna, sort):
    """Missing group keys survive alongside missing values and repeated q."""
    frame = mdf.MicroDataFrame(
        {
            "region": ["north", "north", None, "south", "south"],
            "year": [2024, 2024, 2024, np.nan, 2025],
            "income": [10.0, np.nan, 20.0, 30.0, 40.0],
        },
        weights=[1, 4, 2, 3, 1],
    )
    grouped = frame.groupby(["region", "year"], dropna=False, sort=sort)["income"]
    result = grouped.quantile(quantiles, skipna=skipna)

    # Each retained group has one nonmissing value. With skipna=False,
    # the north group is NaN because it also contains a missing value.
    north = 10.0 if skipna else np.nan
    groups = [("north", 2024.0, north)]
    if sort:
        groups += [
            ("south", 2025.0, 40.0),
            ("south", np.nan, 30.0),
            (np.nan, 2024.0, 20.0),
        ]
    else:
        groups += [
            (np.nan, 2024.0, 20.0),
            ("south", np.nan, 30.0),
            ("south", 2025.0, 40.0),
        ]
    expected_index = pd.MultiIndex.from_tuples(
        [(region, year, q) for region, year, _ in groups for q in quantiles],
        names=["region", "year", None],
    )
    expected_values = [value for _, _, value in groups for _ in quantiles]
    assert result.index.names == expected_index.names
    if quantiles:
        for level in range(3):
            pd.testing.assert_index_equal(
                result.index.get_level_values(level),
                expected_index.get_level_values(level),
            )
    assert result.index.nlevels == 3
    np.testing.assert_allclose(result.to_numpy(), expected_values, equal_nan=True)
    for q in set(quantiles):
        if quantiles.count(q) == 1:
            selected = result.xs(q, level=-1)
            scalar = grouped.quantile(q, skipna=skipna)
            assert selected.index.equals(scalar.index)
            np.testing.assert_allclose(
                selected.to_numpy(), scalar.to_numpy(), equal_nan=True
            )
