from importlib.metadata import PackageNotFoundError, version

from .concat import concat
from .microdataframe import MicroDataFrame, MicroDataFrameGroupBy
from .microseries import MicroSeries, MicroSeriesGroupBy
from .replication import replicate_standard_error, replicate_variance

name = "microdf"

# Read the version from package metadata so it can't drift from
# pyproject.toml (the automated bump only touches pyproject).
try:
    __version__ = version("microdf-python")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "unknown"

__all__ = [
    "concat",
    # microseries.py
    "MicroSeries",
    "MicroSeriesGroupBy",
    # microdataframe.py
    "MicroDataFrame",
    "MicroDataFrameGroupBy",
    # replication.py
    "replicate_variance",
    "replicate_standard_error",
]
