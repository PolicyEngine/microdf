from importlib.metadata import PackageNotFoundError, version

from .microdataframe import MicroDataFrame, MicroDataFrameGroupBy
from .microseries import MicroSeries, MicroSeriesGroupBy

name = "microdf"

# Read the version from package metadata so it can't drift from
# pyproject.toml (the automated bump only touches pyproject).
try:
    __version__ = version("microdf-python")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "unknown"

__all__ = [
    # microseries.py
    "MicroSeries",
    "MicroSeriesGroupBy",
    # microdataframe.py
    "MicroDataFrame",
    "MicroDataFrameGroupBy",
]
