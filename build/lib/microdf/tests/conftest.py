import os

import pytest


@pytest.fixture(scope="session")
def tests_path() -> str:
    """"""
    return os.path.abspath(os.path.dirname(__file__))
