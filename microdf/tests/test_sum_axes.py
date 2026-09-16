import warnings

import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def test_sum_axis_1() -> None:
    # Test basic row-wise sum
    df = mdf.MicroDataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},
        weights=[0.5, 1.0, 2.0],
    )

    # Row-wise sum (axis=1) should not use weights
    row_sums = df.sum(axis=1)
    expected = pd.Series([12, 15, 18], index=df.index)  # 1+4+7, 2+5+8, 3+6+9
    pd.testing.assert_series_equal(pd.Series(row_sums), expected)

    # Column-wise sum (axis=0) should use weights
    col_sums = df.sum(axis=0)
    expected_weighted = pd.Series(
        {
            "A": 1 * 0.5 + 2 * 1.0 + 3 * 2.0,  # 8.5
            "B": 4 * 0.5 + 5 * 1.0 + 6 * 2.0,  # 19.0
            "C": 7 * 0.5 + 8 * 1.0 + 9 * 2.0,  # 29.5
        }
    )
    pd.testing.assert_series_equal(col_sums, expected_weighted)

    # Test with mixed types (non-numeric columns should be ignored)
    df_mixed = mdf.MicroDataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6], "text": ["a", "b", "c"]},
        weights=[1, 1, 1],
    )

    row_sums_mixed = df_mixed.sum(axis=1)
    expected_mixed = pd.Series([5, 7, 9], index=df_mixed.index)  # Only A+B
    pd.testing.assert_series_equal(pd.Series(row_sums_mixed), expected_mixed)

    # Test with axis='columns' (string form)
    row_sums_str = df.sum(axis="columns")
    pd.testing.assert_series_equal(pd.Series(row_sums_str), expected)

    # Test with additional parameters
    df_with_nan = mdf.MicroDataFrame(
        {"A": [1, np.nan, 3], "B": [4, 5, 6], "C": [7, 8, np.nan]},
        weights=[1, 1, 1],
    )

    # skipna=True (default)
    row_sums_skipna = df_with_nan.sum(axis=1)
    expected_skipna = pd.Series([12.0, 13.0, 9.0])  # NaN values skipped
    pd.testing.assert_series_equal(pd.Series(row_sums_skipna), expected_skipna)

    # skipna=False
    row_sums_no_skipna = df_with_nan.sum(axis=1, skipna=False)
    expected_no_skipna = pd.Series([12.0, np.nan, np.nan])  # NaN propagates
    pd.testing.assert_series_equal(pd.Series(row_sums_no_skipna), expected_no_skipna)

    # Test min_count parameter
    row_sums_min_count = df_with_nan.sum(axis=1, min_count=3)
    expected_min_count = pd.Series(
        [12.0, np.nan, np.nan]
    )  # Row 1 and 2 have < 3 non-NA values
    pd.testing.assert_series_equal(pd.Series(row_sums_min_count), expected_min_count)


@pytest.mark.parametrize("axis", [0, "index", 1, "columns"])
@pytest.mark.parametrize("positional", [False, True])
def test_sum_binds_positional_and_keyword_axes(axis, positional):
    frame = mdf.MicroDataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=[7, 8, 9], weights=[1, 2, 3]
    )
    result = frame.sum(axis) if positional else frame.sum(axis=axis)
    if axis in (0, "index"):
        assert type(result) is pd.Series
        pd.testing.assert_series_equal(result, pd.Series({"a": 14.0, "b": 32.0}))
    else:
        assert isinstance(result, mdf.MicroSeries)
        pd.testing.assert_series_equal(
            pd.Series(result), pd.Series([5, 7, 9], index=frame.index)
        )
        pd.testing.assert_series_equal(result.weights, frame.weights)
        # Row values are not weighted yet; subsequent aggregation is weighted.
        assert result.sum() == 5 * 1 + 7 * 2 + 9 * 3
        result.weights.iloc[0] = 100
        assert frame.weights.iloc[0] == 1


@pytest.mark.parametrize("skipna,min_count", [(True, 0), (False, 0), (True, 3)])
@pytest.mark.parametrize("axis", [0, "index", None])
def test_weighted_column_sum_options(axis, skipna, min_count):
    raw = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    weights = pd.Series([1.0, 2.0, 3.0])
    frame = mdf.MicroDataFrame(raw, weights=weights)
    # Native sum defines version-specific axis=None and missing-value behavior.
    # The independently weighted entries are [1, NaN, 9] and [4, 10, 18].
    expected_data = pd.DataFrame({"a": [1.0, np.nan, 9.0], "b": [4.0, 10.0, 18.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = expected_data.sum(axis=axis, skipna=skipna, min_count=min_count)
        actual = frame.sum(axis=axis, skipna=skipna, min_count=min_count)
    if isinstance(expected, pd.Series):
        assert type(actual) is pd.Series
        pd.testing.assert_series_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("axis", [None, 0, "index"])
@pytest.mark.parametrize("skipna,min_count", [(True, 0), (False, 0), (True, 3)])
def test_microseries_sum_options(axis, skipna, min_count):
    series = mdf.MicroSeries([1.0, np.nan, 3.0], index=[7, 8, 9], weights=[1, 2, 3])
    expected = pd.Series([1.0, np.nan, 9.0]).sum(
        axis=axis, skipna=skipna, min_count=min_count
    )
    np.testing.assert_allclose(
        series.sum(axis, skipna=skipna, min_count=min_count), expected, equal_nan=True
    )


@pytest.mark.parametrize("min_count,expected", [(0, 0.0), (1, np.nan)])
def test_empty_numeric_row_sum_identity(min_count, expected):
    frame = mdf.MicroDataFrame({"text": ["a", "b"]}, index=[7, 8], weights=[2, 3])
    actual = frame.sum(axis=1, min_count=min_count)
    assert isinstance(actual, mdf.MicroSeries)
    pd.testing.assert_series_equal(
        pd.Series(actual), pd.Series([expected, expected], index=frame.index)
    )
    pd.testing.assert_series_equal(actual.weights, frame.weights)


def test_sum_rejects_invalid_arguments():
    frame = mdf.MicroDataFrame({"a": [1, 2]}, weights=[1, 2])
    with pytest.raises(TypeError):
        frame.sum(1, axis=0)
    with pytest.raises(TypeError):
        frame.sum(bogus=True)
    with pytest.raises(ValueError):
        frame.sum(axis=2)
    with pytest.raises(ValueError):
        frame["a"].sum(axis=1)


def test_sum_handles_boolean_and_nullable_numeric_columns():
    raw = pd.DataFrame(
        {
            "count": pd.Series([1, None, 3], dtype="Int64"),
            "flag": pd.Series([True, False, True], dtype="boolean"),
            "text": ["a", "b", "c"],
        }
    )
    frame = mdf.MicroDataFrame(raw, weights=[1, 2, 3])
    expected = raw[["count", "flag"]].sum(axis=1)
    pd.testing.assert_series_equal(pd.Series(frame.sum(1)), expected)
    totals = frame.sum(numeric_only=True)
    assert list(totals.index) == ["count", "flag"]
    assert totals["count"] == 10
    assert totals["flag"] == 4


def test_sum_preserves_other_scalar_positional_arguments():
    frame = mdf.MicroDataFrame({"a": [-1.0, 2.0, 3.0]}, weights=[1, 2, 3])
    for method, argument in [
        ("gini", "shift"),
        ("top_x_pct_share", 0.25),
        ("mean", False),
        ("var", 0),
    ]:
        actual = getattr(frame, method)(argument)["a"]
        expected = getattr(frame["a"], method)(argument)
        assert actual == expected


@pytest.mark.parametrize("min_count,expected", [(0, 0.0), (1, np.nan)])
def test_sum_of_empty_inputs(min_count, expected):
    frame = mdf.MicroDataFrame(pd.DataFrame({"a": pd.Series([], dtype=float)}))
    row_sums = frame.sum(axis=1, min_count=min_count)
    assert isinstance(row_sums, mdf.MicroSeries)
    assert row_sums.empty
    pd.testing.assert_series_equal(row_sums.weights, pd.Series([], dtype=float))
    pd.testing.assert_series_equal(
        frame.sum(min_count=min_count), pd.Series({"a": expected})
    )
    series = mdf.MicroSeries([], dtype=float)
    np.testing.assert_allclose(
        series.sum(min_count=min_count), expected, equal_nan=True
    )
    with pytest.raises(ValueError):
        series.sum(axis=1)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("mixed_dtypes", [False, True])
def test_sum_preserves_duplicate_numeric_column_labels(axis, mixed_dtypes):
    if mixed_dtypes:
        raw = pd.DataFrame([[1.0, "x", 4.0], [2.0, "y", 5.0]], columns=["a", "a", "a"])
    else:
        raw = pd.DataFrame([[1.0, 4.0], [2.0, 5.0]], columns=["a", "a"])
    frame = mdf.MicroDataFrame(raw, weights=[2, 3])

    result = frame.sum(axis)

    if axis == 0:
        assert type(result) is pd.Series
        expected = pd.Series([8.0, 23.0], index=["a", "a"])
        pd.testing.assert_series_equal(result, expected)
    else:
        assert isinstance(result, mdf.MicroSeries)
        pd.testing.assert_series_equal(pd.Series(result), pd.Series([5.0, 7.0]))
        pd.testing.assert_series_equal(result.weights, frame.weights)
        assert result.sum() == 31.0
    pd.testing.assert_frame_equal(pd.DataFrame(frame), raw)
