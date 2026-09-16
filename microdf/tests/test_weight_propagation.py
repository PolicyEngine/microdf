"""Weighted pandas operations preserve row identity, not just row labels."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def make_data(kind):
    values = pd.Series([30.0, np.nan, 10.0, 20.0], index=[7, 7, 3, 7], name="x")
    weights = [2.0, 5.0, 11.0, 17.0]
    if kind == "series":
        return mdf.MicroSeries(values, weights=weights)
    return mdf.MicroDataFrame(values.to_frame(), weights=weights)


def assert_rows(result, original, positions, values=None):
    assert type(result) is type(original)
    expected = original.weights.to_numpy()[positions]
    np.testing.assert_array_equal(result.weights.to_numpy(), expected)
    assert result.weights.index.equals(result.index)
    assert result.weights is not original.weights
    data = np.asarray(result if isinstance(result, mdf.MicroSeries) else result["x"])
    if values is not None:
        np.testing.assert_allclose(data, values, equal_nan=True)
    total = np.nansum(data * expected)
    assert (
        result.sum() if isinstance(result, mdf.MicroSeries) else result.sum()["x"]
    ) == total


@pytest.mark.parametrize("kind", ["series", "frame"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_sort_values_preserves_positional_weights(kind, ascending, ignore_index):
    original = make_data(kind)
    args = () if kind == "series" else ("x",)
    result = original.sort_values(*args, ascending=ascending, ignore_index=ignore_index)
    positions = [2, 3, 0, 1] if ascending else [0, 3, 2, 1]
    assert_rows(result, original, positions)
    if ignore_index:
        assert result.index.equals(pd.RangeIndex(4))
    assert_rows(original, original.copy(), np.arange(4))


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_fillna_and_row_transforms_preserve_weights(kind):
    original = make_data(kind)
    filled = original.fillna(4)
    assert_rows(filled, original, np.arange(4), [30, 4, 10, 20])
    assert_rows(filled.replace(30, 40), filled, np.arange(4), [40, 4, 10, 20])
    assert_rows(filled.astype(float), filled, np.arange(4))
    assert_rows(filled + 1, filled, np.arange(4), [31, 5, 11, 21])
    filled.weights.iloc[0] = 100
    assert original.weights.iloc[0] == 2


@pytest.mark.parametrize("kind", ["series", "frame"])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_sample_with_replacement_and_duplicate_labels(kind, ignore_index):
    original = make_data(kind)
    positions = (
        pd.Series(np.arange(4)).sample(n=12, replace=True, random_state=17).to_numpy()
    )
    result = original.sample(
        n=12, replace=True, random_state=17, ignore_index=ignore_index
    )
    assert_rows(result, original, positions)
    assert len(set(positions)) < len(positions)


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_duplicate_label_selections_and_sort_index(kind):
    original = make_data(kind)
    assert_rows(original.iloc[[3, 0, 3]], original, [3, 0, 3])
    assert_rows(original.iloc[1:3], original, [1, 2])
    assert_rows(original.loc[[3, 7]], original, [2, 0, 1, 3])
    assert_rows(original.sort_index(kind="stable"), original, [2, 0, 1, 3])
    if kind == "frame":
        column = original.loc[[3, 7], "x"]
        assert isinstance(column, mdf.MicroSeries)
        np.testing.assert_array_equal(column.weights, [11, 2, 5, 17])
        row = original.iloc[0]
        assert type(row) is pd.Series


@pytest.mark.parametrize("kind", ["series", "frame"])
@pytest.mark.parametrize("options", [{}, {"ignore_index": True}, {"keys": ["a", "b"]}])
def test_concat_rows_preserves_duplicate_and_repeated_weights(kind, options):
    original = make_data(kind)
    pieces = [original.iloc[[3, 0]], original.iloc[[2, 3]]]
    result = pd.concat(pieces, **options)
    assert_rows(result, original, [3, 0, 2, 3])


def test_concat_columns_requires_consistent_row_weights():
    first = mdf.MicroDataFrame({"x": [1, 2]}, index=[9, 3], weights=[4, 5])
    second = mdf.MicroDataFrame({"y": [6, 7]}, index=[3, 9], weights=[5, 4])
    result = pd.concat([first, second], axis=1)
    assert isinstance(result, mdf.MicroDataFrame)
    np.testing.assert_array_equal(result.weights, [4, 5])
    assert result.sum().to_dict() == {"x": 14, "y": 58}
    second.set_weights([50, 40])
    with pytest.raises(ValueError, match="weights"):
        pd.concat([first, second], axis=1)


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_concat_with_unweighted_objects_is_explicit(kind):
    weighted = make_data(kind)
    plain = pd.Series([1, 2]) if kind == "series" else pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="weights"):
        pd.concat([weighted, plain])


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_reindex_new_rows_cannot_invent_weights(kind):
    weighted = make_data(kind).iloc[:1]
    with pytest.raises(ValueError, match="weights"):
        weighted.reindex([7, 99])


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_propagated_objects_keep_serialization_and_rename_isolation(kind):
    original = make_data(kind).fillna(4)
    renamed = (
        original.rename("income")
        if kind == "series"
        else original.rename(columns={"x": "income"})
    )
    renamed.weights.iloc[0] = 99
    assert original.weights.iloc[0] == 2
    for result in [copy.deepcopy(original), pickle.loads(pickle.dumps(original))]:
        assert_rows(result, original, np.arange(4))
    if kind == "series":
        assert original.name == "x"


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_inplace_sort_and_fillna_preserve_weights(kind):
    original = make_data(kind)
    expected = original.copy()
    args = () if kind == "series" else ("x",)
    assert original.sort_values(*args, inplace=True, ascending=False) is None
    assert_rows(original, expected, [0, 3, 2, 1])
    original.fillna(4, inplace=True)
    assert_rows(original, expected, [0, 3, 2, 1], [30, 20, 10, 4])


@pytest.mark.parametrize("kind", ["series", "frame"])
def test_boolean_selection_and_drop_with_duplicate_indexes(kind):
    original = make_data(kind)
    mask = np.array([True, False, True, False])
    assert_rows(original[mask], original, [0, 2])
    assert_rows(original.drop(index=7), original, [2])
    assert_rows(original.loc[mask], original, [0, 2])
    if kind == "series":
        assert_rows(original.repeat(2), original, [0, 0, 1, 1, 2, 2, 3, 3])


def test_column_access_uses_current_weights_without_mutating_earlier_series():
    frame = mdf.MicroDataFrame({"x": [1, 2]}, weights=[3, 4])
    earlier = frame["x"]
    assert earlier.sum() == 11
    frame.weights.iloc[0] = 10
    assert frame["x"].sum() == 18
    assert frame.loc[:, "x"].sum() == 18
    assert earlier.sum() == 11


def test_transpose_cannot_reinterpret_row_weights_as_column_weights():
    frame = mdf.MicroDataFrame(
        [[1, 2], [3, 4]], index=[0, 1], columns=[0, 1], weights=[5, 7]
    )
    with pytest.raises(ValueError, match="weights"):
        frame.transpose()


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "index,level",
    [
        (pd.Index(["a", "a", "b"], name="row"), None),
        (
            pd.MultiIndex.from_tuples(
                [("a", 1), ("a", 1), ("b", 2)], names=["group", "row"]
            ),
            None,
        ),
        (
            pd.MultiIndex.from_tuples(
                [("a", 1), ("a", 1), ("b", 2)], names=["group", "row"]
            ),
            "group",
        ),
    ],
)
def test_series_reset_index_drop_preserves_positional_weights(index, level, inplace):
    original = mdf.MicroSeries(
        [10, 20, 30], index=index, name="income", weights=[2, 3, 5]
    )
    source_weights = original.weights
    expected = pd.Series(original).reset_index(level=level, drop=True, name="ignored")
    result = original.reset_index(
        level=level, drop=True, name="ignored", inplace=inplace
    )
    if inplace:
        assert result is None
        result = original
    assert isinstance(result, mdf.MicroSeries)
    pd.testing.assert_series_equal(pd.Series(result), expected)
    pd.testing.assert_series_equal(
        result.weights, pd.Series([2.0, 3.0, 5.0], index=expected.index)
    )
    assert result.sum() == 230
    assert result.weights is not source_weights
    result.weights.iloc[0] = 99
    assert source_weights.iloc[0] == 2
    source_weights.iloc[1] = 88
    assert result.weights.iloc[1] == 3


def test_series_reset_index_preserves_dataframe_and_invalid_inplace_behavior():
    original = mdf.MicroSeries(
        [10, 20], index=pd.Index(["a", "b"], name="row"), name="income", weights=[2, 3]
    )
    result = original.reset_index(name="amount")
    assert isinstance(result, mdf.MicroDataFrame)
    pd.testing.assert_frame_equal(
        pd.DataFrame(result), pd.Series(original).reset_index(name="amount")
    )
    np.testing.assert_array_equal(result.weights, [2, 3])
    result.weights.iloc[0] = 99
    assert original.weights.iloc[0] == 2
    with pytest.raises(TypeError, match="inplace"):
        original.reset_index(inplace=True)
