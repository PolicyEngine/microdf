"""Binary operations retain the calling Series' observation weights."""

import inspect

import numpy as np
import pandas as pd
import pytest

from microdf import MicroSeries


ARITHMETIC = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow"]
LOGICAL = ["and", "or", "xor"]
COMPARISONS = ["lt", "le", "eq", "ne", "ge", "gt"]


def assert_weighted_result(result, expected, source):
    assert isinstance(result, MicroSeries)
    pd.testing.assert_series_equal(pd.Series(result), expected)
    expected_weights = (
        source.weights
        if source.index.equals(expected.index)
        else source.weights.reindex(expected.index)
    )
    pd.testing.assert_series_equal(result.weights, expected_weights)
    assert result.weights is not source.weights
    assert result.sum() == expected.multiply(expected_weights).sum()


@pytest.mark.parametrize(
    "method",
    [f"__{prefix}{op}__" for op in ARITHMETIC + LOGICAL for prefix in ["", "r"]],
)
@pytest.mark.parametrize("weighted_other", [False, True])
def test_binary_operators_align_weights_with_labels(method, weighted_other):
    source = MicroSeries([10, 20], index=["b", "a"], weights=[1, 9], name="x")
    other = pd.Series([2, 1], index=["a", "b"], name="x")
    if weighted_other:
        other = MicroSeries(other, weights=[9, 1])
    expected = getattr(pd.Series(source), method)(pd.Series(other))

    result = getattr(source, method)(other)

    assert_weighted_result(result, expected, source)
    # Addition is 22 * 9 + 11 * 1 = 209, rather than the positional 121.
    if method in ["__add__", "__radd__"]:
        assert result.sum() == 209
    result.weights.iloc[0] = 100
    np.testing.assert_array_equal(source.weights, [1, 9])
    if weighted_other:
        np.testing.assert_array_equal(other.weights, [9, 1])
    source.weights.iloc[1] = 200
    assert result.weights.iloc[0] == 100


@pytest.mark.parametrize(
    "method",
    ARITHMETIC + [f"r{op}" for op in ARITHMETIC] + COMPARISONS + ["div", "rdiv"],
)
@pytest.mark.parametrize("permuted", [False, True])
def test_named_binary_methods_use_calling_series_weights(method, permuted):
    source = MicroSeries([10, 20], index=["b", "a"], weights=[1, 9], name="x")
    other = MicroSeries(
        [2, 1], index=["a", "b"] if permuted else ["b", "a"], weights=[5, 7], name="x"
    )
    expected = getattr(pd.Series(source), method)(pd.Series(other))

    result = getattr(source, method)(other)

    assert_weighted_result(result, expected, source)
    # Inherited public methods keep the installed pandas API signatures.
    assert inspect.signature(getattr(MicroSeries, method)) == inspect.signature(
        getattr(pd.Series, method)
    )


@pytest.mark.parametrize("method", [f"__{op}__" for op in COMPARISONS])
def test_comparison_operators_keep_calling_series_weights_and_pandas_errors(method):
    source = MicroSeries([10, 20], index=["b", "a"], weights=[1, 9])
    other = MicroSeries([20, 10], index=source.index, weights=[5, 7])
    expected = getattr(pd.Series(source), method)(pd.Series(other))
    assert_weighted_result(getattr(source, method)(other), expected, source)

    other.index = ["a", "b"]
    with pytest.raises(ValueError) as pandas_error:
        getattr(pd.Series(source), method)(pd.Series(other))
    with pytest.raises(ValueError) as microdf_error:
        getattr(source, method)(other)
    assert str(microdf_error.value) == str(pandas_error.value)


@pytest.mark.parametrize("method", ["__add__", "__rsub__", "add", "rsub", "lt"])
@pytest.mark.parametrize("operand", [3, [2, 1], np.array([2, 1])])
def test_scalar_and_array_binary_operands_keep_weights(method, operand):
    source = MicroSeries([10, 20], index=["b", "a"], weights=[1, 9])
    expected = getattr(pd.Series(source), method)(operand)
    assert_weighted_result(getattr(source, method)(operand), expected, source)


@pytest.mark.parametrize("method", ["__add__", "__rsub__", "add", "rsub", "lt"])
def test_matching_duplicate_indexes_keep_positional_weights(method):
    source = MicroSeries([10, 20, 30], index=["a", "a", "b"], weights=[1, 9, 3])
    other = MicroSeries([2, 1, 4], index=source.index, weights=[5, 7, 11])
    expected = getattr(pd.Series(source), method)(pd.Series(other))
    assert_weighted_result(getattr(source, method)(other), expected, source)


@pytest.mark.parametrize("method", ["__divmod__", "__rdivmod__", "divmod", "rdivmod"])
def test_divmod_results_keep_calling_series_weights(method):
    source = MicroSeries([10, 20], index=["b", "a"], weights=[1, 9])
    other = MicroSeries([3, 4], index=["a", "b"], weights=[5, 7])
    expected = getattr(pd.Series(source), method)(pd.Series(other))
    result = getattr(source, method)(other)
    assert isinstance(result, tuple)
    for actual, plain in zip(result, expected):
        assert_weighted_result(actual, plain, source)


@pytest.mark.parametrize("method", ["__add__", "__rsub__", "add", "rsub", "lt"])
@pytest.mark.parametrize(
    "left_index,right_index",
    [
        (["b", "a"], ["a", "c"]),
        (["b", "a", "b"], ["a", "b", "b"]),
    ],
    ids=["new-rows", "ambiguous-duplicates"],
)
def test_binary_operations_reject_unknown_row_weights(method, left_index, right_index):
    source = MicroSeries(
        range(len(left_index)), index=left_index, weights=range(1, len(left_index) + 1)
    )
    other = MicroSeries(
        range(len(right_index)),
        index=right_index,
        weights=range(4, len(right_index) + 4),
    )
    with pytest.raises(ValueError, match="weights"):
        getattr(source, method)(other)


@pytest.mark.parametrize("method", ["add", "rsub", "lt"])
def test_named_binary_arguments_preserve_pandas_values_and_errors(method):
    index = pd.MultiIndex.from_tuples([("b", 2), ("a", 1)], names=["group", "row"])
    source = MicroSeries([np.nan, 20], index=index, weights=[1, 9])
    other = pd.Series([2, 1], index=pd.Index(["a", "b"], name="group"))
    kwargs = {"level": "group", "fill_value": 0, "axis": "index"}
    expected = getattr(pd.Series(source), method)(other, **kwargs)
    assert_weighted_result(getattr(source, method)(other, **kwargs), expected, source)
    for args, options in [
        ((other,), {"axis": 1}),
        (([1],), {}),
        ((other,), {"unknown": True}),
    ]:
        with pytest.raises((TypeError, ValueError)) as pandas_error:
            getattr(pd.Series(source), method)(*args, **options)
        with pytest.raises(type(pandas_error.value)) as microdf_error:
            getattr(source, method)(*args, **options)
        # pandas identifies the concrete subclass in invalid-axis messages.
        expected_error = str(pandas_error.value).replace(
            "object type Series", "object type MicroSeries"
        )
        assert str(microdf_error.value) == expected_error


@pytest.mark.parametrize(
    "operation",
    [
        lambda plain, weighted: plain + weighted,
        lambda plain, weighted: plain - weighted,
        lambda plain, weighted: plain * weighted,
        lambda plain, weighted: plain / weighted,
        lambda plain, weighted: plain // weighted,
        lambda plain, weighted: plain % weighted,
        lambda plain, weighted: plain**weighted,
        lambda plain, weighted: plain & weighted,
        lambda plain, weighted: plain | weighted,
        lambda plain, weighted: plain ^ weighted,
    ],
    ids=ARITHMETIC + LOGICAL,
)
@pytest.mark.parametrize("indexes", ["matching", "permuted", "duplicates"])
def test_plain_series_left_expressions_preserve_weighted_dispatch(operation, indexes):
    index = ["a", "a"] if indexes == "duplicates" else ["b", "a"]
    source = MicroSeries([10, 20], index=index, weights=[1, 9], name="x")
    other_index = ["a", "b"] if indexes == "permuted" else index
    other = pd.Series([2, 1], index=other_index, name="x")
    expected = operation(other, pd.Series(source))

    result = operation(other, source)

    assert_weighted_result(result, expected, source)
    result.weights.iloc[0] = 100
    np.testing.assert_array_equal(source.weights, [1, 9])
    source.weights.iloc[1] = 200
    assert result.weights.iloc[0] == 100
