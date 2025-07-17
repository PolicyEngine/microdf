import os

import pytest


@pytest.fixture(scope="session")
def tests_path():
    """"""
    return os.path.abspath(os.path.dirname(__file__))
