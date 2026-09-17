"""Binary NumPy dispatch retains the weights of identifiable observations."""

import numpy as np
import pandas as pd
import pytest

from microdf import MicroSeries


def test_maximum_preserves_issue_322_weighted_total():
    weighted = MicroSeries([10, 20], weights=[2, 9])

    result = np.maximum(weighted, pd.Series([12, 10]))

    assert isinstance(result, MicroSeries)
    pd.testing.assert_series_equal(pd.Series(result), pd.Series([12, 20]))
    pd.testing.assert_series_equal(result.weights, pd.Series([2.0, 9.0]))
    assert result.sum() == 204  # 12 * 2 + 20 * 9.


def test_plain_series_left_divmod_preserves_issue_322_weighted_totals():
    weighted = MicroSeries([10, 20], weights=[2, 9])

    quotient, remainder = divmod(pd.Series([23, 41]), weighted)

    for result, values, total in [
        (quotient, [2, 2], 22),  # 2 * 2 + 2 * 9.
        (remainder, [3, 1], 15),  # 3 * 2 + 1 * 9.
    ]:
        assert isinstance(result, MicroSeries)
        pd.testing.assert_series_equal(pd.Series(result), pd.Series(values))
        pd.testing.assert_series_equal(result.weights, pd.Series([2.0, 9.0]))
        assert result.sum() == total


@pytest.mark.parametrize("ufunc", [np.maximum, np.minimum, np.fmax])
@pytest.mark.parametrize("weighted_first", [True, False])
@pytest.mark.parametrize(
    "source_labels,other_labels,other_values,other_name,expected_weights",
    [
        (["b", "a", "c"], ["b", "a", "c"], [12, 10, 30], "income", [2, 9, 5]),
        (["b", "a", "c"], ["c", "a", "b"], [30, 10, 12], "other", [9, 2, 5]),
        (["a", "a", "b"], ["a", "a", "b"], [12, 10, 30], "income", [2, 9, 5]),
    ],
    ids=["matching", "reordered", "equal-duplicates"],
)
def test_binary_ufunc_preserves_values_row_weights_and_independence(
    ufunc,
    weighted_first,
    source_labels,
    other_labels,
    other_values,
    other_name,
    expected_weights,
):
    source = pd.Series(
        [10.0, 20.0, np.nan],
        index=pd.Index(source_labels, name="person"),
        name="income",
    )
    other = pd.Series(
        other_values,
        index=pd.Index(other_labels, name="person"),
        name=other_name,
    )
    weighted = MicroSeries(source, weights=[2, 9, 5])
    if weighted_first:
        expected = ufunc(source, other)
        result = ufunc(weighted, other)
    else:
        expected = ufunc(other, source)
        result = ufunc(other, weighted)
    weights = pd.Series(expected_weights, index=expected.index, dtype=float)

    assert isinstance(result, MicroSeries)
    pd.testing.assert_series_equal(pd.Series(result), expected)
    pd.testing.assert_series_equal(result.weights, weights)
    assert result.sum() == expected.multiply(weights).sum()

    result.weights.iloc[0] = 100
    pd.testing.assert_series_equal(
        weighted.weights, pd.Series([2.0, 9.0, 5.0], index=source.index)
    )
    weighted.weights.iloc[-1] = 200
    assert result.weights.iloc[-1] == expected_weights[-1]


@pytest.mark.parametrize("operation", [divmod, np.divmod])
@pytest.mark.parametrize(
    "source_labels,other_labels,other_values,other_name,expected_weights",
    [
        (["b", "a", "c"], ["b", "a", "c"], [23, 41, 95], "income", [2, 9, 5]),
        (["b", "a", "c"], ["c", "a", "b"], [95, 41, 23], "other", [9, 2, 5]),
        (["a", "a", "b"], ["a", "a", "b"], [23, 41, 95], "income", [2, 9, 5]),
    ],
    ids=["matching", "reordered", "equal-duplicates"],
)
def test_reverse_divmod_retains_each_members_values_and_independent_weights(
    operation,
    source_labels,
    other_labels,
    other_values,
    other_name,
    expected_weights,
):
    source = pd.Series(
        [10, 20, 30],
        index=pd.Index(source_labels, name="person"),
        name="income",
    )
    other = pd.Series(
        other_values,
        index=pd.Index(other_labels, name="person"),
        name=other_name,
    )
    weighted = MicroSeries(source, weights=[2, 9, 5])
    expected = operation(other, source)

    result = operation(other, weighted)

    assert isinstance(result, tuple)
    assert len(result) == 2
    for member, plain in zip(result, expected):
        weights = pd.Series(expected_weights, index=plain.index, dtype=float)
        assert isinstance(member, MicroSeries)
        pd.testing.assert_series_equal(pd.Series(member), plain)
        pd.testing.assert_series_equal(member.weights, weights)
        assert member.sum() == plain.multiply(weights).sum()

    quotient, remainder = result
    quotient.weights.iloc[0] = 100
    assert remainder.weights.iloc[0] == expected_weights[0]
    remainder.weights.iloc[1] = 300
    assert quotient.weights.iloc[1] == expected_weights[1]
    pd.testing.assert_series_equal(
        weighted.weights, pd.Series([2.0, 9.0, 5.0], index=source.index)
    )
    weighted.weights.iloc[-1] = 200
    assert quotient.weights.iloc[-1] == expected_weights[-1]
    assert remainder.weights.iloc[-1] == expected_weights[-1]


@pytest.mark.parametrize(
    "operation,source_labels,other_labels",
    [
        (np.maximum, ["a", "b", "c"], ["a", "b", "unknown"]),
        (divmod, ["a", "b", "c"], ["a", "b", "unknown"]),
        (np.divmod, ["a", "b", "c"], ["a", "b", "unknown"]),
        (divmod, ["b", "a", "a"], ["a", "a", "b"]),
        (np.divmod, ["b", "a", "a"], ["a", "a", "b"]),
    ],
    ids=[
        "maximum-unknown-row",
        "divmod-unknown-row",
        "numpy-divmod-unknown-row",
        "divmod-ambiguous-duplicate-rows",
        "numpy-divmod-ambiguous-duplicate-rows",
    ],
)
def test_binary_dispatch_rejects_rows_without_unambiguous_weights(
    operation, source_labels, other_labels
):
    source = pd.Series([10, 20, 30], index=source_labels)
    other = pd.Series([23, 41, 95], index=other_labels)
    weighted = MicroSeries(source, weights=[2, 9, 5])
    # Pandas can produce values, but these rows have no unique weight assignment.
    operation(other, source)

    with pytest.raises(ValueError, match="weights"):
        operation(other, weighted)


@pytest.mark.parametrize("ufunc", [np.maximum, np.minimum, np.fmax])
def test_binary_ufunc_preserves_pandas_duplicate_alignment_errors(ufunc):
    source = pd.Series([10, 20, 30], index=["b", "a", "a"])
    other = pd.Series([23, 41, 95], index=["a", "a", "b"])
    weighted = MicroSeries(source, weights=[2, 9, 5])

    with pytest.raises(ValueError) as pandas_error:
        ufunc(other, source)
    with pytest.raises(type(pandas_error.value)) as microdf_error:
        ufunc(other, weighted)

    assert str(microdf_error.value) == str(pandas_error.value)


@pytest.mark.parametrize("operation", [np.maximum, divmod, np.divmod])
def test_binary_dispatch_preserves_pandas_invalid_dtype_errors(operation):
    source = pd.Series([10, 20], index=["a", "b"], name="income")
    other = pd.Series(["invalid", "data"], index=source.index, dtype=object)
    weighted = MicroSeries(source, weights=[2, 9])

    with pytest.raises(TypeError) as pandas_error:
        operation(other, source)
    with pytest.raises(type(pandas_error.value)) as microdf_error:
        operation(other, weighted)

    assert str(microdf_error.value) == str(pandas_error.value)


@pytest.mark.parametrize("weighted_first", [True, False])
def test_binary_ufunc_preserves_pandas_ndarray_out_and_where(weighted_first):
    source = pd.Series([10.0, 20.0, 30.0], index=["b", "a", "c"], name="income")
    other = pd.Series([12.0, 10.0, 40.0], index=source.index, name="income")
    weighted = MicroSeries(source, weights=[2, 9, 5])
    out = np.full(3, -99.0)
    plain_out = out.copy()
    plain_inputs = (source, other) if weighted_first else (other, source)
    inputs = (weighted, other) if weighted_first else (other, weighted)
    where = np.array([True, False, True])
    expected = np.maximum(*plain_inputs, out=plain_out, where=where)

    result = np.maximum(*inputs, out=out, where=where)

    assert (result is out) == (expected is plain_out)
    np.testing.assert_array_equal(out, [12.0, -99.0, 40.0])
    np.testing.assert_array_equal(out, plain_out)
    pd.testing.assert_series_equal(pd.Series(result), expected)
    assert np.shares_memory(np.asarray(result), out) == np.shares_memory(
        np.asarray(expected), plain_out
    )


@pytest.mark.parametrize("weighted_first", [True, False])
def test_binary_ufunc_explicit_none_out_retains_weights(weighted_first):
    source = pd.Series([10, 20], index=["b", "a"], name="income")
    other = pd.Series([12, 10], index=source.index, name="income")
    weighted = MicroSeries(source, weights=[2, 9])
    plain_inputs = (source, other) if weighted_first else (other, source)
    inputs = (weighted, other) if weighted_first else (other, weighted)
    expected = np.maximum(*plain_inputs, out=None)

    result = np.maximum(*inputs, out=None)

    assert isinstance(result, MicroSeries)
    pd.testing.assert_series_equal(pd.Series(result), expected)
    pd.testing.assert_series_equal(
        result.weights, pd.Series([2.0, 9.0], index=source.index)
    )
    assert result.sum() == 204  # 12 * 2 + 20 * 9.


@pytest.mark.parametrize("ufunc", [np.negative, np.modf])
def test_unary_ufunc_and_tuple_outputs_keep_existing_weight_behavior(ufunc):
    source = pd.Series([10.25, -20.5], index=["b", "a"], name="income")
    weighted = MicroSeries(source, weights=[2, 9])
    expected = ufunc(source)

    result = ufunc(weighted)

    if isinstance(expected, tuple):
        assert isinstance(result, tuple)
        assert len(result) == len(expected)
    else:
        result, expected = (result,), (expected,)
    weights = pd.Series([2.0, 9.0], index=source.index)
    for member, plain in zip(result, expected):
        assert isinstance(member, MicroSeries)
        pd.testing.assert_series_equal(pd.Series(member), plain)
        pd.testing.assert_series_equal(member.weights, weights)
        assert member.sum() == plain.multiply(weights).sum()


def test_add_ufunc_reduction_keeps_weighted_sum_behavior():
    weighted = MicroSeries([10, 20], index=["b", "a"], weights=[2, 9])

    assert np.add.reduce(weighted) == 200  # 10 * 2 + 20 * 9.


@pytest.mark.parametrize("base", [pd.Series, pd.DataFrame])
@pytest.mark.parametrize("weighted_first", [True, False])
def test_binary_ufunc_defers_to_foreign_pandas_handlers(base, weighted_first):
    sentinel = object()
    calls = []

    class ForeignPandasObject(base):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            calls.append((ufunc, method, inputs, kwargs))
            return sentinel

    weighted = MicroSeries([10, 20], weights=[2, 9])
    foreign = ForeignPandasObject([12, 10])
    inputs = (weighted, foreign) if weighted_first else (foreign, weighted)

    result = np.maximum(*inputs)

    assert result is sentinel
    assert len(calls) == 1
    ufunc, method, received_inputs, kwargs = calls[0]
    assert ufunc is np.maximum
    assert method == "__call__"
    assert len(received_inputs) == len(inputs)
    assert all(
        received is original for received, original in zip(received_inputs, inputs)
    )
    assert kwargs == {}


@pytest.mark.parametrize("base", [pd.Series, pd.DataFrame])
@pytest.mark.parametrize("weighted_first", [True, False])
def test_binary_ufunc_defers_to_higher_priority_pandas_subclasses(base, weighted_first):
    class HigherPriorityObject(base):
        __array_priority__ = MicroSeries.__array_priority__ + 1

    source = pd.Series([10, 20])
    weighted = MicroSeries(source, weights=[2, 9])
    higher = HigherPriorityObject([12, 10])
    plain_inputs = (source, higher) if weighted_first else (higher, source)
    inputs = (weighted, higher) if weighted_first else (higher, weighted)

    # Test the deferral protocol directly: the foreign handler decides how
    # to handle the operation after MicroSeries returns NotImplemented.
    assert (
        source.__array_ufunc__(np.maximum, "__call__", *plain_inputs) is NotImplemented
    )
    assert weighted.__array_ufunc__(np.maximum, "__call__", *inputs) is NotImplemented
