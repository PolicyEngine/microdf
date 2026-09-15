import microdf as mdf


def test_version_matches_package_metadata():
    """__version__ must not drift from pyproject.toml."""
    from importlib.metadata import version

    assert mdf.__version__ == version("microdf-python")
