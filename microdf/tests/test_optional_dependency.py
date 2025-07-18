import sys
import types

import pytest

from microdf._optional import VERSIONS, import_optional_dependency


def test_import_optional() -> None:
    """"""
    match = "Missing .*notapackage.* pip .* conda .* notapackage"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("notapackage")

    result = import_optional_dependency("notapackage", raise_on_missing=False)
    assert result is None


def test_xlrd_version_fallback() -> None:
    """"""
    pytest.importorskip("xlrd")
    import_optional_dependency("xlrd")


def test_bad_version() -> None:
    """"""
    name = "fakemodule"
    module = types.ModuleType(name)
    module.__version__ = "0.9.0"
    sys.modules[name] = module
    VERSIONS[name] = "1.0.0"

    match = "microdf requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency("fakemodule")

    with pytest.warns(UserWarning):
        result = import_optional_dependency("fakemodule", on_version="warn")
    assert result is None

    module.__version__ = "1.0.0"  # exact match is OK
    result = import_optional_dependency("fakemodule")
    assert result is module


def test_no_version_raises() -> None:
    """"""
    name = "fakemodule"
    module = types.ModuleType(name)
    sys.modules[name] = module
    VERSIONS[name] = "1.0.0"

    with pytest.raises(ImportError, match="Can't determine .* fakemodule"):
        import_optional_dependency(name)
