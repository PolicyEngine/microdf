"""The API reference must list every public method, in both directions.

A one-way check lets the page fall behind the code silently, which is how the
weighted estimators came to be missing from it.
"""

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
