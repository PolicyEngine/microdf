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


def test_unweighted_groupby() -> None:
    df = mdf.MicroDataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert (df.groupby("x").z.sum().values == np.array([5.0, 6.0])).all()


def test_multiple_groupby() -> None:
    df = mdf.MicroDataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    assert (df.groupby(["x", "y"]).z.sum() == np.array([5, 6])).all()


def test_concat() -> None:
    df1 = mdf.MicroDataFrame({"x": [1, 2]}, weights=[3, 4])
    df2 = mdf.MicroDataFrame({"y": [5, 6]}, weights=[7, 8])
    # Verify that pd.concat returns DataFrame (probably no way to fix this).
    pd_long = pd.concat([df1, df2])
    assert isinstance(pd_long, pd.DataFrame)
    assert not isinstance(pd_long, mdf.MicroDataFrame)
    # Verify that mdf.concat works.
    mdf_long = mdf.concat([df1, df2])
    assert isinstance(mdf_long, mdf.MicroDataFrame)
    # Weights should be preserved.
    assert mdf_long.weights.equals(pd.concat([df1.weights, df2.weights]))
    # Verify it works horizontally too (take the first set of weights).
    mdf_wide = mdf.concat([df1, df2], axis=1)
    assert isinstance(mdf_wide, mdf.MicroDataFrame)
    assert mdf_wide.weights.equals(df1.weights)


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
