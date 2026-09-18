#!/usr/bin/env python
"""Regenerate ``docs/api.md`` from the live package.

Run it from anywhere:

    uv run --with . --no-project python docs/build_api.py

The page used to be produced by a script that lived outside the repo, which is
how the source and the page drifted apart four separate times in review.  This
is that script, committed, so the page can always be rebuilt and a stale page is
a visible diff.

Signatures are rendered through ``microdf._docs``, the same module the test
imports, so the page reads the same under pandas 2 and pandas 3 and the test
cannot disagree with the generator.

The prose is held verbatim here; only the tables are generated.
"""

import inspect
from pathlib import Path

import microdf as mdf
from microdf._docs import markdown_signature

PAGE = Path(__file__).resolve().parent / "api.md"

# Descriptions that are not the method's own docstring: methods inherited or
# overridden from pandas keep pandas' docstring, which describes pandas'
# behaviour rather than what microdf does with the weights.  These are written
# by hand and must survive regeneration.
OVERRIDES = {
    ("MicroSeries", "groupby"): (
        "Group into `MicroSeriesGroupBy`, carrying weights into each group."
    ),
    ("MicroSeries", "cumsum"): (
        "Cumulative sum of value times weight. Returns a plain `pandas.Series`: "
        "the weights have been applied and are not carried forward, so this is "
        "the one method here that does not preserve them."
    ),
    ("MicroSeries", "clip"): "Trim values at the given thresholds, preserving weights.",
    ("MicroSeries", "round"): "Round each value, preserving weights.",
    ("MicroSeries", "repeat"): "Repeat elements, repeating their weights alongside.",
    ("MicroSeries", "sqrt"): "Element-wise square root, preserving weights.",
    ("MicroSeries", "copy"): "Copy the series and its weights.",
    ("MicroSeries", "equals"): "True when both the values and the weights are equal.",
    ("MicroDataFrame", "cov"): "Pairwise frequency-weighted covariance of the columns.",
    ("MicroDataFrame", "corr"): (
        "Pairwise frequency-weighted Pearson correlation of the columns."
    ),
    ("MicroDataFrame", "merge"): (
        "Database-style join that carries the weight column through."
    ),
    ("MicroDataFrame", "reset_index"): (
        "Reset the index, keeping weights aligned to their rows."
    ),
    ("MicroDataFrame", "drop"): (
        "Drop rows or columns, keeping weights aligned to the remaining rows."
    ),
    ("MicroDataFrame", "copy"): "Copy the frame and its weights.",
    ("MicroDataFrame", "equals"): (
        "True when both the values and the weights are equal."
    ),
}

HEADER = "| Method | Signature | Description |\n|---|---|---|"


def describe(cls, name):
    key = (cls.__name__, name)
    if key in OVERRIDES:
        return OVERRIDES[key]
    attr = inspect.getattr_static(cls, name)
    if isinstance(attr, property):
        attr = attr.fget
    doc = inspect.getdoc(attr) or ""
    first = " ".join(doc.split("\n\n")[0].split()).split(":param")[0].strip()
    if not first:
        raise SystemExit(
            f"{cls.__name__}.{name} has no docstring and no description override"
        )
    return first


def row(cls, name):
    attr = inspect.getattr_static(cls, name)
    if isinstance(attr, property):
        signature = "*attribute*"
    else:
        signature = f"`{markdown_signature(attr)}`"
    return f"| `{name}` | {signature} | {describe(cls, name)} |"


def table(cls, names):
    return "\n".join([HEADER] + [row(cls, name) for name in names])


SERIES = mdf.MicroSeries
FRAME = mdf.MicroDataFrame


def build():
    return f"""# API reference

`microdf` exposes two classes. `MicroSeries` is a `pandas.Series` carrying a weight
vector; `MicroDataFrame` is a `pandas.DataFrame` carrying a weight column. Both
behave like their pandas counterparts, and the methods below either add a
weighted estimator or preserve weights through an operation that would otherwise
drop them.

```python
import microdf as mdf

df = mdf.MicroDataFrame({{"income": [10_000, 30_000, 120_000]}}, weights=[800, 1_200, 50])
df.income.gini()
```

## MicroSeries

### Weighted aggregation

These have the same names as their pandas equivalents and return weighted results.

{
        table(
            SERIES,
            [
                "sum",
                "count",
                "mean",
                "median",
                "quantile",
                "var",
                "std",
                "cov",
                "corr",
                "rank",
            ],
        )
    }

### Weight-preserving operations

Operations that change the shape or type of the data, overridden so weights stay aligned with their rows.

{
        table(
            SERIES,
            [
                "groupby",
                "cumsum",
                "astype",
                "clip",
                "round",
                "repeat",
                "sqrt",
                "copy",
                "equals",
                "values",
                "to_numpy",
            ],
        )
    }

### Inequality and distribution

{
        table(
            SERIES,
            [
                "gini",
                "top_1_pct_share",
                "top_10_pct_share",
                "top_50_pct_share",
                "bottom_50_pct_share",
                "top_0_1_pct_share",
                "top_x_pct_share",
                "bottom_x_pct_share",
                "t10_b50",
            ],
        )
    }

### Ranking

{table(SERIES, ["decile_rank", "quintile_rank", "quartile_rank", "percentile_rank"])}

### Variance from replicate weights

{table(SERIES, ["replicate_standard_error"])}

`replicate_standard_error` accepts `method` of `jackknife`, `brr`, `bootstrap`,
`successive-difference`, or `fay` (which also requires `fay_k`). Because it
resamples rather than applying an analytic formula, it works for any statistic
the series can compute, including the Gini coefficient and quantiles.

### Weights

{table(SERIES, ["set_weights", "nullify_weights", "weight"])}

## MicroDataFrame

### Weighted aggregation

{table(FRAME, ["sum", "cov", "corr"])}

### Weight-preserving operations

{
        table(
            FRAME,
            ["groupby", "merge", "reset_index", "drop", "astype", "copy", "equals"],
        )
    }

### Poverty

{
        table(
            FRAME,
            [
                "poverty_rate",
                "poverty_gap",
                "poverty_count",
                "deep_poverty_rate",
                "deep_poverty_gap",
                "squared_poverty_gap",
            ],
        )
    }

### Weights

{table(FRAME, ["set_weights", "set_weight_col", "nullify_weights"])}

## Module-level functions

| Function | Description |
|---|---|
| `microdf.replicate_variance` | Variance of a statistic from replicate weights. |
| `microdf.replicate_standard_error` | Square root of the above. |

## A note on estimator conventions

Quantiles follow the inverse cumulative distribution function, so results can be
checked against `survey::svyquantile` in R. Weighted variance treats weights as
frequency weights, so integer weights agree with `numpy` computed on the
replicated sample. Top-share cutoffs split a record that straddles the boundary
in proportion, rather than assigning it wholly to one side.
"""


if __name__ == "__main__":
    PAGE.write_text(build())
    print(f"wrote {PAGE}")
