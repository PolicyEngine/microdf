"""Regression cases for issue #333, also used to build the support matrix."""

import operator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def survey():
    return mdf.MicroDataFrame({"x": [10.0, 100.0], "g": ["a", "a"]}, weights=[9, 1])


# Each supported spelling has a numerical expectation, not only a type check.
WEIGHTED_CASES = [
    ("groupby column mean", lambda d: d.groupby("g").x.mean().iloc[0], 19),
    ("groupby dict agg", lambda d: d.groupby("g").agg({"x": "mean"}).iloc[0, 0], 19),
    ("groupby named agg", lambda d: d.groupby("g").agg(m=("x", "mean")).iloc[0, 0], 19),
    (
        "groupby callable agg",
        lambda d: d.groupby("g").agg({"x": lambda s: s.mean()}).iloc[0, 0],
        19,
    ),
    (
        "SeriesGroupBy callable agg",
        lambda d: d.groupby("g").x.agg(lambda s: s.mean()).iloc[0],
        19,
    ),
    (
        "callable pivot_table",
        lambda d: d.pivot_table(index="g", values="x", aggfunc=lambda s: s.mean()).iloc[
            0, 0
        ],
        19,
    ),
    ("row apply", lambda d: d.apply(lambda r: r.x, axis=1).mean(), 19),
    ("frame numeric mean", lambda d: d.mean(numeric_only=True)["x"], 19),
    ("frame numeric median", lambda d: d.median(numeric_only=True)["x"], 10),
    (
        "groupby numeric mean",
        lambda d: d.groupby("g").mean(numeric_only=True).loc["a", "x"],
        19,
    ),
    (
        "groupby numeric median",
        lambda d: d.groupby("g").median(numeric_only=True).loc["a", "x"],
        10,
    ),
    ("frame constructor", lambda d: mdf.MicroDataFrame(d).x.mean(), 19),
    ("series constructor", lambda d: mdf.MicroSeries(d.x).mean(), 19),
    ("pd.cut preserves weights", lambda d: pd.cut(d.x, 2, labels=False).sum(), 1),
    ("pd.qcut preserves weights", lambda d: pd.qcut(d.x, 2, labels=False).sum(), 1),
    (
        "pd.to_numeric preserves weights",
        lambda d: pd.to_numeric(d.x.astype(str)).mean(),
        19,
    ),
    (
        "series explode",
        lambda d: (
            mdf.MicroSeries([[10.0, 10.0], [100.0]], weights=d.weights).explode().sum()
        ),
        280,
    ),
    ("np.average", lambda d: np.average(d.x), 19),
    ("np.mean", lambda d: np.mean(d.x), 19),
    ("np.median", lambda d: np.median(d.x), 10),
    ("weighted value_counts", lambda d: d.x.value_counts().loc[10.0], 9),
    ("weighted mode", lambda d: d.x.mode().iloc[0], 10),
    ("numeric frame dropna", lambda d: d[["x"]].dropna().x.mean(), 19),
]


@pytest.mark.parametrize(
    "label,operation,expected", WEIGHTED_CASES, ids=[c[0] for c in WEIGHTED_CASES]
)
def test_supported_operations(label, operation, expected):
    assert operation(survey()) == pytest.approx(expected)


REJECTED = [
    "rolling",
    "expanding",
    "ewm",
    "sem",
    "skew",
    "kurt",
    "kurtosis",
    "prod",
    "product",
    "idxmax",
    "idxmin",
]


@pytest.mark.parametrize("method", REJECTED)
def test_unsupported_series_operations_name_plain_escape(method):
    args = (2,) if method in {"rolling", "ewm"} else ()
    with pytest.raises(NotImplementedError, match=r"pd\.Series\(s\)"):
        getattr(survey().x, method)(*args)


@pytest.mark.parametrize(
    "operation",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        np.add,
        np.maximum,
        lambda a, b: a.add(b),
        lambda a, b: a.rsub(b),
    ],
)
@pytest.mark.parametrize("kind", [mdf.MicroSeries, mdf.MicroDataFrame])
def test_conflicting_weights_raise_in_both_orders(operation, kind):
    a = kind([10.0, 100.0], weights=[9, 1])
    b = kind([1.0, 2.0], weights=[1, 9])
    for left, right in [(a, b), (b, a)]:
        with pytest.raises(ValueError, match="weights"):
            operation(left, right)


@pytest.mark.parametrize("kind", [mdf.MicroSeries, mdf.MicroDataFrame])
def test_constructor_aligns_and_copies_source_weights(kind):
    source = kind([10.0, 100.0], index=["b", "a"], weights=[9, 1])
    result = kind(data=source, index=["a", "b"])
    assert result.weights.tolist() == [1, 9]
    result.weights.iloc[0] = 100
    assert source.weights.tolist() == [9, 1]
    assert kind(source, weights=[2, 3]).weights.tolist() == [2, 3]


@pytest.mark.parametrize(
    "operation", [lambda d: d.dropna(), lambda d: d.dropna(ignore_index=True)]
)
def test_dropna_tracks_positions_with_duplicate_labels(operation):
    d = mdf.MicroDataFrame(
        {"x": [10.0, np.nan, 100.0]}, index=[4, 4, 4], weights=[9, 5, 1]
    )
    result = operation(d)
    assert result.weights.tolist() == [9, 1]
    assert result.x.mean() == 19


def test_groupby_multiple_functions_and_named_callables():
    d = survey()
    result = d.groupby("g").agg({"x": ["mean", "sum"]})
    assert result.loc["a", ("x", "mean")] == 19
    assert result.loc["a", ("x", "sum")] == 190
    named = d.groupby("g").agg(m=("x", lambda s: s.mean()), total=("x", "sum"))
    assert named.loc["a"].tolist() == [19, 190]
    assert d.groupby("g").x.aggregate(["mean", "sum"]).loc["a"].tolist() == [19, 190]


def test_callable_groups_keep_duplicate_row_weights():
    d = mdf.MicroDataFrame(
        {"x": [10.0, 100.0, 30.0], "g": ["a", "a", "b"]},
        index=[2, 2, 2],
        weights=[9, 1, 4],
    )
    result = d.groupby("g").agg(m=("x", lambda s: s.mean()))
    assert result.m.tolist() == [19, 30]


def test_weighted_value_counts_and_mode():
    s = mdf.MicroSeries([1, 2, 2, np.nan, 3], weights=[9, 1, 1, 3, 0])
    assert s.value_counts().loc[1] == 9
    assert s.value_counts(normalize=True).loc[2] == pytest.approx(2 / 11)
    assert s.value_counts(dropna=False).loc[np.nan] == 3
    assert s.mode().tolist() == [1]
    assert type(s.value_counts()) is pd.Series
    assert type(s.mode()) is pd.Series


def test_numpy_options_cannot_silently_override_weights():
    s = survey().x
    assert np.average(s, returned=True) == (19, 10)
    with pytest.raises((ValueError, NotImplementedError), match="weights"):
        np.average(s, weights=[1, 9])
    with pytest.raises(NotImplementedError):
        np.median(s, overwrite_input=True)


def support_matrix():
    rows = [
        "# Supported operations",
        "",
        "Generated from `microdf/tests/test_fail_closed.py` by `uv run python docs/build_support.py`.",
        "",
        "The regression suite checks these contracts on pandas 2 and 3. Aggregated results carry no row weights; transformations retain independent copies of the input weights.",
        "",
        "| Operation | Behaviour |",
        "|---|---|",
    ]
    rows += [
        f"| {label} | Weighted result / preserved weights |"
        for label, _, _ in WEIGHTED_CASES
    ]
    rows += [
        f"| `{name}` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |"
        for name in REJECTED
    ]
    rows += [
        "| Arithmetic with conflicting weights | Raises `ValueError` in either operand order |",
        "",
        "`pd.cut` and `pd.qcut` retain row weights, but choose bin edges using pandas' unweighted rules. Supply explicit bin edges for weighted quantile bins.",
        "",
    ]
    rows += [
        "`pivot_table` grouping keys must name columns; external Series, callable",
        "groupers and index-level groupers raise. Weighted `Series.value_counts` and",
        "`Series.mode` return plain summary Series; frame and grouped variants raise.",
        "",
        "`microdf.concat` rejects mixed weighted/plain inputs in either order. Direct",
        "`pd.concat` still bypasses microdf when its first input is plain pandas; the",
        "regression suite records this upstream dispatch limitation as an expected failure.",
        "Use Micro objects for every input to `pd.concat`, or use `microdf.concat`.",
        "",
    ]
    return "\n".join(rows)


def test_support_matrix_is_current():
    path = Path(__file__).resolve().parents[2] / "docs" / "support.md"
    if not path.exists():
        pytest.skip("support page not installed")
    assert path.read_text() == support_matrix()


@pytest.mark.parametrize(
    "convert", [lambda s: pd.cut(s, 2), lambda s: pd.qcut(s, 2), pd.to_numeric]
)
def test_pandas_conversions_preserve_independent_weights(convert):
    source = mdf.MicroSeries([10.0, 100.0], index=[5, 5], weights=[9, 1], name="x")
    result = convert(source)
    assert isinstance(result, mdf.MicroSeries)
    pd.testing.assert_series_equal(result.weights, source.weights)
    result.weights.iloc[0] = 25
    assert source.weights.iloc[0] == 9


@pytest.mark.parametrize("weighted_first", [False, True])
@pytest.mark.parametrize("mapping", [False, True])
def test_checked_concat_rejects_plain_inputs_in_any_order(weighted_first, mapping):
    inputs = [survey(), pd.DataFrame({"x": [1.0]})]
    if not weighted_first:
        inputs.reverse()
    if mapping:
        inputs = dict(zip(["a", "b"], inputs))
    with pytest.raises(ValueError, match="weights"):
        mdf.concat(inputs)


def test_checked_concat_keeps_weighted_rows():
    result = mdf.concat([survey(), survey()], ignore_index=True)
    assert result.weights.tolist() == [9, 1, 9, 1]
    assert result.x.mean() == 19


@pytest.mark.xfail(
    strict=True,
    reason="pandas chooses the plain first input's constructor without a subclass dispatch hook; use microdf.concat",
)
def test_plain_first_pandas_concat_dispatch_boundary():
    with pytest.raises(ValueError, match="weights"):
        pd.concat([pd.DataFrame({"x": [1.0]}), survey()])


@pytest.mark.parametrize("method", ["sem", "skew", "prod", "idxmax", "idxmin"])
@pytest.mark.parametrize(
    "selection", [lambda d: d.groupby("g"), lambda d: d.groupby("g").x]
)
def test_groupby_unsupported_reductions_raise(method, selection):
    with pytest.raises(NotImplementedError, match=r"pd\."):
        getattr(selection(survey()), method)()


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("select", [lambda g: g, lambda g: g[["x"]]])
def test_groupby_selection_numeric_only_and_result_index(as_index, select):
    grouped = select(survey().groupby("g", as_index=as_index))
    for result in [grouped.mean(numeric_only=True), grouped.agg({"x": "mean"})]:
        assert result.x.tolist() == [19]
        if as_index:
            assert result.index.name == "g"
        else:
            assert result.g.tolist() == ["a"]


@pytest.mark.parametrize("kind", [mdf.MicroSeries, mdf.MicroDataFrame])
def test_equal_weight_arithmetic_retains_weighted_answer(kind):
    left = kind([10.0, 100.0], weights=[9, 1])
    right = kind([1.0, 2.0], weights=[9, 1])
    result = left + right
    assert result.weights.tolist() == [9, 1]
    value = result.mean()
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    assert value == pytest.approx(20.1)


def test_covariance_documentation_example():
    pair = mdf.MicroDataFrame(
        {"x": [1.0, 3.0, 5.0], "y": [2.0, 5.0, 4.0]}, weights=[1, 2, 1]
    )
    expected = np.array([[8 / 3, 4 / 3], [4 / 3, 2]])
    np.testing.assert_allclose(pair.cov(), expected)
    np.testing.assert_allclose(
        np.cov([[1.0, 3.0, 5.0], [2.0, 5.0, 4.0]], fweights=[1, 2, 1]), expected
    )
    replicated = pd.DataFrame(pair).iloc[[0, 1, 1, 2]]
    np.testing.assert_allclose(pair.corr(), replicated.corr())
    assert type(pair.cov()) is pd.DataFrame
    assert type(pair.corr()) is pd.DataFrame


@pytest.mark.parametrize(
    "operation", [np.maximum, np.minimum, np.add, lambda a, b: a + b]
)
def test_frame_arithmetic_with_equal_weights_keeps_weights(operation):
    left = survey()[["x"]]
    right = mdf.MicroDataFrame({"x": [12.0, 80.0]}, weights=[9, 1])
    result = operation(left, right)
    expected = operation(pd.DataFrame(left), pd.DataFrame(right))
    pd.testing.assert_frame_equal(pd.DataFrame(result), expected)
    assert result.weights.tolist() == [9, 1]
    assert result.x.mean() == pytest.approx((expected.x * [9, 1]).sum() / 10)


def test_numeric_group_keys_are_excluded_from_reductions():
    d = mdf.MicroDataFrame({"x": [10.0, 100.0], "g": [1, 1]}, weights=[9, 1])
    result = d.groupby("g").mean(numeric_only=True)
    assert list(result.columns) == ["x"]
    assert result.x.tolist() == [19]


@pytest.mark.parametrize("method", REJECTED + ["mode", "value_counts"])
def test_frame_unsupported_reductions_name_plain_escape(method):
    args = (2,) if method in {"rolling", "ewm"} else ()
    with pytest.raises(NotImplementedError, match=r"pd\.DataFrame\(df\)"):
        getattr(survey(), method)(*args)


def test_named_aggregation_accepts_keyword_like_output_names():
    result = survey().groupby("g").agg(numeric_only=("x", "mean"), engine=("x", "sum"))
    assert result.loc["a"].tolist() == [19, 190]


def test_grouped_callables_with_missing_group_keys_and_multiple_keys():
    d = mdf.MicroDataFrame(
        {"x": [10.0, 100.0, 30.0], "g": ["a", "a", None], "h": [1, 1, 2]},
        weights=[9, 1, 4],
    )
    result = d.groupby(["g", "h"], dropna=False).agg(m=("x", lambda s: s.mean()))
    assert result.m.tolist() == [19, 30]


def test_dropna_inplace_and_columns():
    d = mdf.MicroDataFrame({"x": [10.0, 100.0], "y": [None, 2.0]}, weights=[9, 1])
    assert d.dropna(axis=1).x.mean() == 19
    assert d.dropna(inplace=True) is None
    assert d.weights.tolist() == [1]
    assert d.x.mean() == 100


def test_row_apply_expansion_retains_weights():
    result = survey().apply(lambda r: [r.x, 2 * r.x], axis=1, result_type="expand")
    assert isinstance(result, mdf.MicroDataFrame)
    assert result.mean().tolist() == [19, 38]


@pytest.mark.parametrize("keys", [pd.Series(["a", "a"], index=[5, 6]), lambda row: row])
def test_pivot_rejects_groupers_without_positional_provenance(keys):
    d = mdf.MicroDataFrame({"x": [10.0, 100.0]}, index=[5, 6], weights=[9, 1])
    with pytest.raises(NotImplementedError, match="grouping keys must name columns"):
        d.pivot_table(values="x", index=keys, aggfunc=lambda s: s.mean())
