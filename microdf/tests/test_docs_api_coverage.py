"""The API reference must list every public method, in both directions.

A one-way check lets the page fall behind the code silently, which is how the
weighted estimators came to be missing from it.
"""

import inspect
import re
from pathlib import Path

import pytest

import microdf as mdf

DOCS = Path(__file__).resolve().parents[2] / "docs" / "api.md"

# docs/ is not shipped in the sdist or the wheel, so these cannot run against an
# installed copy of the package.
pytestmark = pytest.mark.skipif(
    not DOCS.exists(), reason="docs/api.md is not present in the installed package"
)

# Public names that read as internals rather than API a user would call.
INTERNAL = {
    "scalar_function",
    "vector_function",
    "override_df_functions",
    "catch_series_relapse",
    "get_args_as_micro_series",
}


def documented_names():
    return set(re.findall(r"^\| `(\w+)` \|", DOCS.read_text(), re.M))


def public_methods(cls):
    """Public names this class defines itself.

    Anything inherited unchanged from pandas is pandas' to document; what
    matters here is what microdf adds or overrides.
    """
    names = set()
    for name, attr in vars(cls).items():
        if name.startswith("_") or name in INTERNAL:
            continue
        if callable(attr) or isinstance(attr, property):
            names.add(name)
    return names


def test_every_documented_method_exists():
    documented = documented_names()
    real = (
        public_methods(mdf.MicroSeries)
        | public_methods(mdf.MicroDataFrame)
        | {"replicate_variance", "replicate_standard_error"}
    )
    assert not (documented - real), f"documented but absent: {documented - real}"


def test_every_public_method_is_documented():
    documented = documented_names()
    for cls in (mdf.MicroSeries, mdf.MicroDataFrame):
        missing = public_methods(cls) - documented
        assert not missing, (
            f"{cls.__name__} methods missing from docs/api.md: {sorted(missing)}"
        )


def test_documented_weight_behaviour_holds():
    """Pin the claims the page makes that a docstring does not enforce.

    Every error found in review was in a description written by hand rather
    than taken from a docstring, so the ones that remain are asserted here.
    """
    import numpy as np
    import pandas as pd

    frame = mdf.MicroDataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 2.0, 8.0]},
        weights=[1.0, 1.0, 1.0, 5.0],
    )
    replicated = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0] + [4.0] * 5, "y": [1.0, 4.0, 2.0] + [8.0] * 5}
    )

    # The page says these are unweighted, and points at #327.
    plain = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 2.0, 8.0]})
    assert frame.cov().loc["x", "y"] == pytest.approx(plain.cov().loc["x", "y"])
    assert frame.corr().loc["x", "y"] == pytest.approx(plain.corr().loc["x", "y"])

    # The page says the MicroSeries versions are frequency-weighted.
    assert frame.x.cov(frame.y) == pytest.approx(replicated.cov().loc["x", "y"])

    # The page says equals compares weights.
    light = mdf.MicroSeries([1, 2, 3], weights=[1, 1, 1])
    heavy = mdf.MicroSeries([1, 2, 3], weights=[9, 9, 9])
    assert not light.equals(heavy)

    # The page says cumsum drops the weights.
    assert not hasattr(frame.x.cumsum(), "weights")

    # The page says repeat repeats the weights alongside the values.
    repeated = mdf.MicroSeries([1.0, 2.0], weights=[3.0, 4.0]).repeat(2)
    assert list(np.asarray(repeated.weights)) == [3.0, 3.0, 4.0, 4.0]


def test_no_row_is_missing_its_description():
    """Every documented method needs a description.

    A row whose description cell is blank is the signature of a source change
    that the page was never regenerated for, which happened three times in
    review before this test existed.
    """
    blank = re.findall(
        r"^\| `(\w+)` \| (?:`[^`]*`|\*attribute\*) \|\s*\|$", DOCS.read_text(), re.M
    )
    assert not blank, f"rows with no description: {blank}"


def test_signatures_match_the_live_ones():
    """The rendered signature must be the one the code actually has.

    Rows are attributed to the class whose `## ` heading they fall under, since
    several names exist on both.
    """
    current, mismatches = None, []
    for line in DOCS.read_text().split("\n"):
        if line.startswith("## MicroSeries"):
            current = mdf.MicroSeries
        elif line.startswith("## MicroDataFrame"):
            current = mdf.MicroDataFrame
        elif line.startswith("## "):
            current = None
        row = re.match(r"^\| `(\w+)` \| `([^`]*)` \|", line)
        if not row or current is None:
            continue
        name, rendered = row.groups()
        func = inspect.getattr_static(current, name, None)
        if func is None or isinstance(func, property):
            continue
        try:
            live = str(inspect.signature(func))
        except (TypeError, ValueError):
            continue
        live = live.replace("(self, ", "(").replace("(self)", "()").replace("|", "\\|")
        if live != rendered:
            mismatches.append((current.__name__, name, rendered, live))
    assert not mismatches, f"page is out of date with the code: {mismatches}"
