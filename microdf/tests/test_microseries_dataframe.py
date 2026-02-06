import warnings

import numpy as np
import pandas as pd

import microdf as mdf
from microdf.microdataframe import MicroDataFrame
from microdf.microseries import MicroSeries


def test_df_init() -> None:
    arr = np.array([0, 1, 1])
    w = np.array([3, 0, 9])
    df = mdf.MicroDataFrame({"a": arr}, weights=w)
    assert df.a.mean() == np.average(arr, weights=w)

    df = mdf.MicroDataFrame()
    df["a"] = arr
    df.set_weights(w)
    assert df.a.mean() == np.average(arr, weights=w)

    df = mdf.MicroDataFrame()
    df["a"] = arr
    df["w"] = w
    df.set_weight_col("w")
    assert df.a.mean() == np.average(arr, weights=w)

    # Test set_weights with string (column name)
    df2 = mdf.MicroDataFrame()
    df2["a"] = arr
    df2["w"] = w
    df2.set_weights("w")  # Using string column name instead of set_weight_col
    assert df2.a.mean() == np.average(arr, weights=w)
    assert np.array_equal(df2.weights.values, w)


def test_handles_empty_index() -> None:
    arr = np.array([0, 1, 1])
    w = np.array([3, 0, 9])
    df = mdf.MicroDataFrame({"a": arr}, weights=w)

    empty_index = pd.Index([])
    df[empty_index]  # Implicit assert; checking for ValueError


def test_series_getitem() -> None:
    arr = np.array([0, 1, 1])
    w = np.array([3, 0, 9])
    s = mdf.MicroSeries(arr, weights=w)
    assert s[[1, 2]].sum() == np.sum(arr[[1, 2]] * w[[1, 2]])

    assert s[1:3].sum() == np.sum(arr[1:3] * w[1:3])


def test_sum() -> None:
    arr = np.array([0, 1, 1])
    w = np.array([3, 0, 9])
    series = mdf.MicroSeries(arr, weights=w)
    assert series.sum() == (arr * w).sum()

    arr = np.linspace(-20, 100, 100)
    w = np.linspace(1, 3, 100)
    series = mdf.MicroSeries(arr)
    series.set_weights(w)
    assert series.sum() == (arr * w).sum()

    # Verify that an error is thrown when passing weights of different size
    # from the values.
    w = np.linspace(1, 3, 101)
    series = mdf.MicroSeries(arr)
    try:
        series.set_weights(w)
        assert False
    except Exception:
        pass


def test_mean() -> None:
    arr = np.array([3, 0, 2])
    w = np.array([4, 1, 1])
    series = mdf.MicroSeries(arr, weights=w)
    assert series.mean() == np.average(arr, weights=w)

    arr = np.linspace(-20, 100, 100)
    w = np.linspace(1, 3, 100)
    series = mdf.MicroSeries(arr)
    series.set_weights(w)
    assert series.mean() == np.average(arr, weights=w)

    w = np.linspace(1, 3, 101)
    series = mdf.MicroSeries(arr)
    try:
        series.set_weights(w)
        assert False
    except Exception:
        pass


def test_mean_skipna() -> None:
    # Test skipna=True (default) - should skip NaN values
    arr = np.array([3.0, np.nan, 2.0])
    w = np.array([4.0, 1.0, 1.0])
    series = mdf.MicroSeries(arr, weights=w)

    # skipna=True should exclude NaN and its weight
    expected = np.average([3.0, 2.0], weights=[4.0, 1.0])
    assert series.mean(skipna=True) == expected
    assert series.mean() == expected  # Default should be skipna=True

    # Test skipna=False - should return NaN if any value is NaN
    assert np.isnan(series.mean(skipna=False))

    # Test with all NaN values
    arr_all_nan = np.array([np.nan, np.nan, np.nan])
    w_all_nan = np.array([1.0, 2.0, 3.0])
    series_all_nan = mdf.MicroSeries(arr_all_nan, weights=w_all_nan)
    assert np.isnan(series_all_nan.mean(skipna=True))
    assert np.isnan(series_all_nan.mean(skipna=False))

    # Test with no NaN values - skipna should not affect result
    arr_no_nan = np.array([3.0, 5.0, 2.0])
    w_no_nan = np.array([4.0, 1.0, 1.0])
    series_no_nan = mdf.MicroSeries(arr_no_nan, weights=w_no_nan)
    expected_no_nan = np.average(arr_no_nan, weights=w_no_nan)
    assert series_no_nan.mean(skipna=True) == expected_no_nan
    assert series_no_nan.mean(skipna=False) == expected_no_nan


def test_poverty_count() -> None:
    arr = np.array([10000, 20000, 50000])
    w = np.array([1123, 1144, 2211])
    df = pd.DataFrame()
    df["income"] = arr
    df["threshold"] = 16000
    df = MicroDataFrame(df, weights=w)
    assert df.poverty_count("income", "threshold") == w[0]


def test_median() -> None:
    # 1, 2, 3, 4, *4*, 4, 5, 5, 5
    arr = np.array([1, 2, 3, 4, 5])
    w = np.array([1, 1, 1, 3, 3])
    series = mdf.MicroSeries(arr, weights=w)
    assert series.median() == 4


def test_weighted_quantile_skewed() -> None:
    # 99% of the population has 0 income, 1% has 1M
    # The median should be 0, not an interpolated value
    series = mdf.MicroSeries([0, 1_000_000], weights=[99, 1])
    assert series.median() == 0
    assert series.quantile(0.5) == 0
    # 99th percentile is still 0 since exactly 99% have 0
    assert series.quantile(0.99) == 0
    # Only quantile > 0.99 gives 1M
    assert series.quantile(1.0) == 1_000_000
    # Test multiple quantiles
    result = series.quantile([0.1, 0.5, 0.99, 1.0])
    assert result[0.1] == 0
    assert result[0.5] == 0
    assert result[0.99] == 0
    assert result[1.0] == 1_000_000


def test_weighted_quantile_boundaries() -> None:
    # Test q=0 returns minimum, q=1 returns maximum
    series = mdf.MicroSeries([10, 20, 30], weights=[1, 1, 1])
    assert series.quantile(0.0) == 10
    assert series.quantile(1.0) == 30


def test_weighted_quantile_equal_weights() -> None:
    # With equal weights, should match "replicated" interpretation
    # Values: 1, 2, 3 each with weight 2 -> like [1,1,2,2,3,3]
    series = mdf.MicroSeries([1, 2, 3], weights=[2, 2, 2])
    # cumsum_normalized = [2/6, 4/6, 6/6] = [0.333, 0.667, 1.0]
    # median (0.5): smallest where cumsum >= 0.5 -> index 1 -> value 2
    assert series.median() == 2
    # 0.25 quantile: smallest where cumsum >= 0.25 -> index 0 -> value 1
    assert series.quantile(0.25) == 1
    # 0.75 quantile: smallest where cumsum >= 0.75 -> index 2 -> value 3
    assert series.quantile(0.75) == 3


def test_weighted_quantile_unsorted_input() -> None:
    # Ensure sorting works correctly
    series = mdf.MicroSeries([30, 10, 20], weights=[1, 2, 1])
    # Sorted: values [10, 20, 30], weights [2, 1, 1]
    # cumsum_normalized = [0.5, 0.75, 1.0]
    assert series.quantile(0.0) == 10
    assert series.quantile(0.5) == 10  # cumsum[0]=0.5 >= 0.5
    assert series.quantile(0.6) == 20  # cumsum[1]=0.75 >= 0.6
    assert series.quantile(1.0) == 30


def test_unweighted_groupby() -> None:
    df = mdf.MicroDataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert (df.groupby("x").z.sum().values == np.array([5.0, 6.0])).all()


def test_multiple_groupby() -> None:
    df = mdf.MicroDataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert (df.groupby(["x", "y"]).z.sum() == np.array([5, 6])).all()


def test_set_index() -> None:
    d = mdf.MicroDataFrame(dict(x=[1, 2, 3]), weights=[4, 5, 6])
    assert d.x.__class__ == MicroSeries
    d.index = [1, 2, 3]
    assert d.x.__class__ == MicroSeries


def test_reset_index() -> None:
    d = mdf.MicroDataFrame(dict(x=[1, 2, 3]), weights=[4, 5, 6])
    assert d.reset_index().__class__ == MicroDataFrame


def test_cumsum() -> None:
    s = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    assert np.array_equal(s.cumsum().values, [4, 14, 32])

    s = mdf.MicroSeries([2, 1, 3], weights=[5, 4, 6])
    assert np.array_equal(s.cumsum().values, [10, 14, 32])

    s = mdf.MicroSeries([3, 1, 2], weights=[6, 4, 5])
    assert np.array_equal(s.cumsum().values, [18, 22, 32])


def test_rank() -> None:
    s = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    assert np.array_equal(s.rank().values, [4, 9, 15])

    s = mdf.MicroSeries([3, 1, 2], weights=[6, 4, 5])
    assert np.array_equal(s.rank().values, [15, 4, 9])

    s = mdf.MicroSeries([2, 1, 3], weights=[5, 4, 6])
    assert np.array_equal(s.rank().values, [9, 4, 15])


def test_percentile_rank() -> None:
    s = mdf.MicroSeries([4, 2, 3, 1], weights=[20, 40, 20, 20])
    assert np.array_equal(s.percentile_rank().values, [100, 60, 80, 20])


def test_quartile_rank() -> None:
    s = mdf.MicroSeries([4, 2, 3], weights=[25, 50, 25])
    assert np.array_equal(s.quartile_rank().values, [4, 2, 3])


def test_quintile_rank() -> None:
    s = mdf.MicroSeries([4, 2, 3], weights=[20, 60, 20])
    assert np.array_equal(s.quintile_rank().values, [5, 3, 4])


def test_decile_rank() -> None:
    s = mdf.MicroSeries(
        [5, 4, 3, 2, 1, 6, 7, 8, 9],
        weights=[10, 20, 10, 10, 10, 10, 10, 10, 10],
    )
    assert np.array_equal(s.decile_rank().values, [6, 5, 3, 2, 1, 7, 8, 9, 10])


def test_copy_equals() -> None:
    d = mdf.MicroDataFrame(
        {"x": [1, 2], "y": [3, 4], "z": [5, 6]}, weights=[7, 8]
    )
    d_copy = d.copy()
    d_copy_diff_weights = d_copy.copy()
    d_copy_diff_weights.weights *= 2
    assert d.equals(d_copy)
    assert not d.equals(d_copy_diff_weights)
    # Same for a MicroSeries.
    assert d.x.equals(d_copy.x)
    assert not d.x.equals(d_copy_diff_weights.x)


def test_subset() -> None:
    df = mdf.MicroDataFrame(
        {"x": [1, 2], "y": [3, 4], "z": [5, 6]}, weights=[7, 8]
    )
    df_no_z = mdf.MicroDataFrame({"x": [1, 2], "y": [3, 4]}, weights=[7, 8])
    assert df[["x", "y"]].equals(df_no_z)
    df_no_z_diff_weights = df_no_z.copy()
    df_no_z_diff_weights.weights += 1
    assert not df[["x", "y"]].equals(df_no_z_diff_weights)


def test_value_subset() -> None:
    d = mdf.MicroDataFrame({"x": [1, 2, 3], "y": [1, 2, 2]}, weights=[4, 5, 6])
    d2 = d[d.y > 1]
    assert d2.y.shape == d2.weights.shape


def test_bitwise_ops_return_microseries() -> None:
    s1 = mdf.MicroSeries([True, False, True], weights=[1, 2, 3])
    s2 = mdf.MicroSeries([False, False, True], weights=[1, 2, 3])
    and_result = s1 & s2
    or_result = s1 | s2
    assert isinstance(and_result, mdf.MicroSeries)
    assert isinstance(or_result, mdf.MicroSeries)
    expected_and = mdf.MicroSeries([False, False, True], weights=[1, 2, 3])
    expected_or = mdf.MicroSeries([True, False, True], weights=[1, 2, 3])
    assert and_result.equals(expected_and)
    assert or_result.equals(expected_or)


def test_additional_ops_return_microseries() -> None:
    s = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    radd = 1 + s
    xor = s ^ mdf.MicroSeries([0, 1, 0], weights=[4, 5, 6])
    inv = ~mdf.MicroSeries([True, False], weights=[1, 1])
    assert isinstance(radd, mdf.MicroSeries)
    assert isinstance(xor, mdf.MicroSeries)
    assert isinstance(inv, mdf.MicroSeries)


def test_reset_index_inplace() -> None:
    df = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=["a", "b", "c", "d"]
    )
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    mdf = MicroDataFrame(df, weights=weights)

    # Test 1: reset_index with inplace=False (default)
    mdf_copy = mdf.copy()
    result = mdf_copy.reset_index()
    assert list(mdf_copy.index) == ["a", "b", "c", "d"]
    assert list(result.index) == [0, 1, 2, 3]
    assert "index" in result.columns
    assert list(result["index"]) == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(result.weights.values, weights)

    # Test 2: reset_index with inplace=True
    mdf_copy = mdf.copy()
    result = mdf_copy.reset_index(inplace=True)
    assert result is None
    assert list(mdf_copy.index) == [0, 1, 2, 3]
    assert "index" in mdf_copy.columns
    assert list(mdf_copy["index"]) == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(mdf_copy.weights.values, weights)
    assert isinstance(mdf_copy["A"], MicroSeries)
    assert isinstance(mdf_copy["B"], MicroSeries)
    assert isinstance(mdf_copy["index"], MicroSeries)

    # Test 3: reset_index with drop=True
    mdf_copy = mdf.copy()
    mdf_copy.reset_index(drop=True, inplace=True)
    assert list(mdf_copy.index) == [0, 1, 2, 3]
    assert "index" not in mdf_copy.columns
    assert list(mdf_copy.columns) == ["A", "B"]
    np.testing.assert_array_equal(mdf_copy.weights.values, weights)

    # Test 4: Multi-level index
    arrays = [["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"]]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
    df_multi = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=multi_index
    )
    mdf_multi = MicroDataFrame(df_multi, weights=weights)
    result = mdf_multi.reset_index(level="first")
    assert "first" in result.columns
    assert result.index.name == "second"
    np.testing.assert_array_equal(result.weights.values, weights)

    # Reset all levels in place
    mdf_multi.reset_index(inplace=True)
    assert "first" in mdf_multi.columns
    assert "second" in mdf_multi.columns
    assert list(mdf_multi.index) == [0, 1, 2, 3]
    np.testing.assert_array_equal(mdf_multi.weights.values, weights)


def test_loc_preserves_weights() -> None:
    """Test that .loc[] returns MicroDataFrame with proper weights (issue
    #265)."""
    df = mdf.MicroDataFrame(
        {"one": [1, 1, 1, 1, 1]}, weights=[10, 20, 30, 40, 50]
    )

    # Filter all rows (should get same weights)
    filtered = df.loc[df.one == 1]
    assert isinstance(filtered, MicroDataFrame)
    assert filtered.one.sum() == 150.0  # Weighted sum

    # Partial filter
    df2 = mdf.MicroDataFrame(
        {"x": [1, 2, 3, 4, 5]}, weights=[10, 20, 30, 40, 50]
    )
    subset = df2.loc[df2.x > 2]
    assert isinstance(subset, MicroDataFrame)
    assert subset.x.sum() == 500.0  # 3*30 + 4*40 + 5*50 = 500
    np.testing.assert_array_equal(subset.weights.values, [30.0, 40.0, 50.0])


def test_iloc_preserves_weights() -> None:
    """Test that .iloc[] returns MicroDataFrame with proper weights."""
    df = mdf.MicroDataFrame(
        {"x": [1, 2, 3, 4, 5]}, weights=[10, 20, 30, 40, 50]
    )

    # Select rows by position
    subset = df.iloc[2:5]
    assert isinstance(subset, MicroDataFrame)
    assert subset.x.sum() == 500.0  # 3*30 + 4*40 + 5*50 = 500
    np.testing.assert_array_equal(subset.weights.values, [30.0, 40.0, 50.0])


def test_groupby_column_selection() -> None:
    """Test that groupby column selection preserves weights (issue #193)."""
    d = mdf.MicroDataFrame(
        dict(g=["a", "a", "b"], y=[1, 2, 3]), weights=[4, 5, 6]
    )

    # Test single column string selection
    result_str = d.groupby("g")["y"].sum()
    assert result_str["a"] == 14.0  # 1*4 + 2*5 = 14
    assert result_str["b"] == 18.0  # 3*6 = 18

    # Test list column selection
    result_list = d.groupby("g")[["y"]].sum()
    assert result_list.loc["a", "y"] == 14.0
    assert result_list.loc["b", "y"] == 18.0

    # Aggregated results should be plain DataFrame (no spurious weight column)
    result_all = d.groupby("g").sum()
    assert "weight" not in result_all.columns
    assert list(result_all.columns) == ["y"]


def test_values_warns() -> None:
    """Accessing .values on a MicroSeries should emit a UserWarning."""
    ms = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = ms.values
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "weights" in str(w[0].message).lower()


def test_to_numpy_warns() -> None:
    """Calling .to_numpy() on a MicroSeries should emit a UserWarning."""
    ms = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = ms.to_numpy()
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "weights" in str(w[0].message).lower()


def test_mean_no_warning() -> None:
    """Internal .values usage in .mean() should NOT emit a warning."""
    ms = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = ms.mean()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0


def test_repr_no_warning() -> None:
    """Internal .values usage in __repr__ should NOT emit a warning."""
    ms = mdf.MicroSeries([1, 2, 3], weights=[4, 5, 6])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = repr(ms)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
