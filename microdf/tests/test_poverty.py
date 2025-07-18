import numpy as np
import pandas as pd

import microdf as mdf

df = pd.DataFrame(
    {
        "income": [-10, 0, 10, 20],
        "threshold": [15, 10, 15, 10],
        "weight": [1, 2, 3, 4],
    }
)
md = mdf.MicroDataFrame(df[["income", "threshold"]], weights=df.weight)


def test_poverty_rate() -> None:
    # Unweighted
    assert np.allclose(mdf.poverty_rate(df, "income", "threshold"), 3 / 4)
    # Weighted
    assert np.allclose(
        mdf.poverty_rate(df, "income", "threshold", "weight"), 6 / 10
    )
    assert np.allclose(md.poverty_rate("income", "threshold"), 6 / 10)


def test_deep_poverty_rate() -> None:
    # Unweighted
    assert np.allclose(mdf.deep_poverty_rate(df, "income", "threshold"), 2 / 4)
    # Weighted
    assert np.allclose(
        mdf.deep_poverty_rate(df, "income", "threshold", "weight"), 3 / 10
    )
    assert np.allclose(md.deep_poverty_rate("income", "threshold"), 3 / 10)


def test_poverty_gap() -> None:
    # Unweighted
    assert np.allclose(mdf.poverty_gap(df, "income", "threshold"), 25 + 10 + 5)
    # Weighted
    RES = 25 * 1 + 10 * 2 + 5 * 3
    assert np.allclose(
        mdf.poverty_gap(df, "income", "threshold", "weight"), RES
    )
    assert np.allclose(md.poverty_gap("income", "threshold"), RES)


def test_squared_poverty_gap() -> None:
    # Unweighted
    assert np.allclose(
        mdf.squared_poverty_gap(df, "income", "threshold"),
        25**2 + 10**2 + 5**2,
    )
    # Weighted
    RES = 1 * (25**2) + 2 * (10**2) + 3 * (5**2)
    assert np.allclose(
        mdf.squared_poverty_gap(df, "income", "threshold", "weight"),
        RES,
    )
    assert np.allclose(md.squared_poverty_gap("income", "threshold"), RES)


def test_deep_poverty_gap() -> None:
    # Unweighted
    assert np.allclose(
        mdf.deep_poverty_gap(df, "income", "threshold"), 17.5 + 5 + 0 + 0
    )
    # Weighted
    RES = 17.5 * 1 + 5 * 2 + 0 * 3 + 0 * 4
    assert np.allclose(
        mdf.deep_poverty_gap(df, "income", "threshold", "weight"), RES
    )
    # Same in MicroDataFrame.
    assert np.allclose(md.deep_poverty_gap("income", "threshold"), RES)
