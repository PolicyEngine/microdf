import distutils.version
import importlib
import types
import warnings
from typing import Optional, Union

# Adapted from:
# https://github.com/pandas-dev/pandas/blob/master/pandas/compat/_optional.py

VERSIONS = {
    "taxcalc": "2.0.0",
}


def _get_version(module: types.ModuleType) -> str:
    """

    :param module: types.ModuleType:
    :param module: types.ModuleType:

    """
    version = getattr(module, "__version__", None)
    if version is None:
        # xlrd uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)

    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    return version


def import_optional_dependency(
    name: str,
    extra: Optional[str] = "",
    raise_on_missing: Optional[bool] = True,
    on_version: Optional[str] = "raise",
) -> Union[types.ModuleType, None]:
    """Import an optional dependency. By default, if a dependency is missing an
    ImportError with a nice message will be raised. If a dependency is present,
    but too old, we raise.

    :param name: The module name. This should be top-level only, so that the
        version may be checked.
    :type name: str
    :param extra: Additional text to include in the ImportError message.
    :type extra: str
    :param raise_on_missing: Whether to raise if the optional dependency is
        not found. When False and the module is not present, None is returned.
    :type raise_on_missing: bool, default True
    :param on_version: What to do when a dependency's version is too old.
        * raise : Raise an ImportError
        * warn : Warn that the version is too old. Returns None
        * ignore: Return the module, even if the version is too old.
          It's expected that users validate the version locally when
    :type on_version: str {'raise', 'warn'}
    """
    msg = (
        f"Missing optional dependency '{name}'. {extra} "
        f"Use pip or conda to install {name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_on_missing:
            raise ImportError(msg) from None
        else:
            return None

    minimum_version = VERSIONS.get(name)
    if minimum_version:
        version = _get_version(module)
        if distutils.version.LooseVersion(version) < minimum_version:
            assert on_version in {"warn", "raise", "ignore"}
            msg = (
                f"microdf requires version '{minimum_version}' or newer of "
                f"'{name}' "
                f"(version '{version}' currently installed)."
            )
            if on_version == "warn":
                warnings.warn(msg, UserWarning)
                return None
            elif on_version == "raise":
                raise ImportError(msg)

    return module
