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
