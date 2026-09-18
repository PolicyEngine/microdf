import microdf as mdf
import numpy as np
import pandas as pd
import pytest


def test_aggregation_surfaces_real_errors():
    """A genuine argument error must raise, not be swallowed per column."""
    df = mdf.MicroDataFrame(pd.DataFrame({"x": [1, 2, 3]}), weights=[1, 1, 1])
    with pytest.raises(ValueError, match="Unknown negatives option"):
        df.gini(negatives="bogus")


def test_aggregation_still_skips_non_numeric_columns():
    """Narrowing the guard must not change which columns aggregate."""
    df = mdf.MicroDataFrame(
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "s": ["a", "b", "c"],
                "dt": pd.to_datetime(["2020-01-01"] * 3),
            }
        ),
        weights=[1, 2, 3],
    )
    assert list(df.sum().index) == ["x"]
    assert df.sum()["x"] == 14
    assert list(df.mean().index) == ["x"]


def test_gini_shift_accepts_nonnegative_and_negative_columns():
    frame = mdf.MicroDataFrame(
        {"positive": [1, 2, 3], "negative": [-1, 1, 3]}, weights=[1, 1, 1]
    )
    # Shift leaves [1, 2, 3] unchanged and changes [-1, 1, 3] to [0, 2, 4].
    # The pairwise-difference Ginis are 2/9 and 4/9, respectively.
    expected = pd.Series({"positive": 2 / 9, "negative": 4 / 9})
    pd.testing.assert_series_equal(frame.gini(negatives="shift"), expected)


@pytest.mark.parametrize("selected", [False, True])
def test_grouped_gini_shift_accepts_positive_groups(selected):
    frame = mdf.MicroDataFrame(
        {"group": ["a", "a", "a", "b", "b", "b"], "value": [1, 2, 3, -1, 1, 3]},
        weights=[1, 1, 1, 1, 1, 1],
    )
    grouped = frame.groupby("group")
    if selected:
        grouped = grouped[["value"]]
    expected = pd.DataFrame(
        {"value": [2 / 9, 4 / 9]}, index=pd.Index(["a", "b"], name="group")
    )
    pd.testing.assert_frame_equal(grouped.gini(negatives="shift"), expected)


def test_gini_shift_accepts_empty_series():
    assert np.isnan(mdf.MicroSeries([], weights=[]).gini(negatives="shift"))
