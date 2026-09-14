import copy
import io
import pickle

import pandas as pd
import pytest

import microdf as mdf


def test_microseries_survives_pickling():
    """Weights must survive a pickle round-trip."""
    import pickle

    s = mdf.MicroSeries([1, 2, 3], index=[7, 8, 9], weights=[1, 2, 3])
    restored = pickle.loads(pickle.dumps(s))
    assert isinstance(restored, mdf.MicroSeries)
    assert restored.sum() == 14
    assert list(restored.weights) == [1.0, 2.0, 3.0]


def test_microdataframe_survives_pickling():
    """Weights and the weighted aggregations must survive a round-trip."""
    import pickle

    df = mdf.MicroDataFrame(
        pd.DataFrame({"x": [1, 2, 3]}, index=[7, 8, 9]), weights=[1, 2, 3]
    )
    restored = pickle.loads(pickle.dumps(df))
    assert isinstance(restored, mdf.MicroDataFrame)
    assert isinstance(restored.weights, pd.Series)
    # Would be 6 (unweighted) if the aggregation overrides were not
    # reinstalled after unpickling.
    assert restored.sum()["x"] == 14


def test_deepcopy_preserves_weights():
    df = mdf.MicroDataFrame(pd.DataFrame({"x": [1, 2, 3]}), weights=[1, 2, 3])
    assert copy.deepcopy(df).sum()["x"] == 14
    s = mdf.MicroSeries([1, 2, 3], weights=[1, 2, 3])
    assert copy.deepcopy(s).sum() == 14


@pytest.mark.parametrize("use_weight_column", [False, True])
@pytest.mark.parametrize("use_pandas_pickle", [False, True])
def test_serialization_preserves_weight_column_state(
    use_weight_column, use_pandas_pickle
):
    """Replacing restored weights can preserve the original weight column."""
    frame = mdf.MicroDataFrame(
        pd.DataFrame({"x": [1, 2, 3], "w": [1, 2, 3]}, index=[7, 8, 9]),
        weights="w" if use_weight_column else [1, 2, 3],
    )
    if use_pandas_pickle:
        buffer = io.BytesIO()
        frame.to_pickle(buffer)
        buffer.seek(0)
        restored = pd.read_pickle(buffer)
    else:
        restored = pickle.loads(pickle.dumps(frame))

    assert restored.weights_col == ("w" if use_weight_column else None)
    restored.set_weights([3, 2, 1], preserve_old=True)
    assert restored.sum()["x"] == 10
    assert restored.index.equals(frame.index)
    if use_weight_column:
        assert restored["old_w"].tolist() == [1, 2, 3]
    else:
        assert "old_w" not in restored.columns


@pytest.mark.parametrize("operation", ["pickle", "pandas_pickle", "deepcopy"])
def test_named_microseries_preserves_name_and_weights(operation):
    """Serialization retains pandas metadata as well as survey weights."""
    series = mdf.MicroSeries(
        [1, 2, 3], index=[7, 8, 9], name="group", weights=[1, 2, 3]
    )
    if operation == "deepcopy":
        restored = copy.deepcopy(series)
    elif operation == "pandas_pickle":
        buffer = io.BytesIO()
        series.to_pickle(buffer)
        buffer.seek(0)
        restored = pd.read_pickle(buffer)
    else:
        restored = pickle.loads(pickle.dumps(series))

    assert restored.name == "group"
    assert restored.index.equals(series.index)
    pd.testing.assert_series_equal(restored.weights, series.weights)
    assert restored.sum() == 14


@pytest.mark.parametrize("selected", [False, True])
def test_grouped_aggregation_retains_index_name(selected):
    """Copying internal grouped weights must retain the grouping label."""
    frame = mdf.MicroDataFrame(
        {"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]},
        weights=[1, 2, 3, 4],
    )
    grouped = frame.groupby("group")
    if selected:
        grouped = grouped[["value"]]
    expected = pd.DataFrame(
        {"value": [5.0, 25.0]}, index=pd.Index(["a", "b"], name="group")
    )
    pd.testing.assert_frame_equal(grouped.sum(), expected)
