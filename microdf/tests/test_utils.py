import pandas as pd

import microdf as mdf


def test_cartesian_product() -> None:
    """"""
    res = mdf.cartesian_product(
        {"a": [1, 2, 3], "b": ["val1", "val2"], "c": [100, 101]}
    )
    EXPECTED = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "b": [
                "val1",
                "val1",
                "val2",
                "val2",
                "val1",
                "val1",
                "val2",
                "val2",
                "val1",
                "val1",
                "val2",
                "val2",
            ],
            "c": [100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101],
        }
    )
    pd.testing.assert_frame_equal(res, EXPECTED)


def test_flatten() -> None:
    """"""
    L = [[[1, 2, 3], [4, 5]], 6]
    res = list(mdf.flatten(L))
    EXPECTED = [1, 2, 3, 4, 5, 6]
    assert res == EXPECTED
