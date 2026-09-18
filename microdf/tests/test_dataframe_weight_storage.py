import warnings

import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def test_weights_stay_a_series_after_nullify():
    """nullify_weights must leave weights as an index-aligned Series."""
    df = mdf.MicroDataFrame(pd.DataFrame({"x": [1, 2, 3]}), weights=[4, 5, 6])
    df.nullify_weights()
    assert isinstance(df.weights, pd.Series)
    assert list(df.weights.index) == list(df.index)
    assert df.equals(df)
    assert df.sum()["x"] == 6


def test_weights_stay_a_series_after_set_weight_col():
    """The deprecated set_weight_col must also produce a Series."""
    df = mdf.MicroDataFrame(pd.DataFrame({"x": [1, 2, 3], "w": [1.0, 2.0, 3.0]}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df.set_weight_col("w")
    assert isinstance(df.weights, pd.Series)
    assert df.weights_col == "w"
    assert df.equals(df)
    assert df.sum()["x"] == 14


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_weight_column_edits_do_not_change_stored_weights(dtype):
    """Selecting a weight column takes a copy of its current values."""
    df = mdf.MicroDataFrame(
        {"x": [10, 20], "w": np.array([1, 2], dtype=dtype)},
        index=[10, 20],
    )
    with pytest.warns(DeprecationWarning):
        df.set_weight_col("w")

    df.loc[10, "w"] = 100

    np.testing.assert_array_equal(df.weights, [1, 2])
    assert df.sum()["x"] == 10 * 1 + 20 * 2


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_stored_weight_edits_do_not_change_weight_column(dtype):
    """Changing stored weights leaves the source column values intact."""
    df = mdf.MicroDataFrame(
        {"x": [10, 20], "w": np.array([1, 2], dtype=dtype)},
        index=[10, 20],
    )
    with pytest.warns(DeprecationWarning):
        df.set_weight_col("w")

    df.weights.iloc[0] = 100

    np.testing.assert_array_equal(df["w"], [1, 2])
    assert df.sum()["x"] == 10 * 100 + 20 * 2


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "index,level",
    [
        (pd.Index(["b", "a", "a"], name="row"), None),
        (
            pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("a", 1)], names=["group", "row"]
            ),
            None,
        ),
        (
            pd.MultiIndex.from_tuples(
                [("b", 2), ("a", 1), ("a", 1)], names=["group", "row"]
            ),
            "group",
        ),
    ],
)
def test_reset_index_owns_independently_mutable_weights(index, level, drop, inplace):
    source = mdf.MicroDataFrame({"x": [10, 20, 30]}, index=index, weights=[1, 9, 3])
    original_weights = source.weights
    expected = pd.DataFrame(source).reset_index(level=level, drop=drop)

    result = source.reset_index(level=level, drop=drop, inplace=inplace)

    if inplace:
        assert result is None
        result = source
    assert isinstance(result, mdf.MicroDataFrame)
    pd.testing.assert_frame_equal(pd.DataFrame(result), expected)
    pd.testing.assert_series_equal(
        result.weights, pd.Series([1.0, 9.0, 3.0], index=expected.index)
    )
    assert result.x.sum() == 10 * 1 + 20 * 9 + 30 * 3
    assert result.weights is not original_weights
    result.weights.iloc[0] = 100
    np.testing.assert_array_equal(original_weights, [1, 9, 3])
    assert result.x.sum() == 10 * 100 + 20 * 9 + 30 * 3
    if not inplace:
        assert source.x.sum() == 10 * 1 + 20 * 9 + 30 * 3
    original_weights.iloc[1] = 200
    np.testing.assert_array_equal(result.weights, [100, 9, 3])
    assert result.x.sum() == 10 * 100 + 20 * 9 + 30 * 3
