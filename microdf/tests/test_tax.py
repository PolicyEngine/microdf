import numpy as np
import pandas as pd
import pytest

import microdf as mdf


def test_tax() -> None:
    """"""
    # Consider a MTR schedule of 0% up to 10,000, then 10% after that.
    BRACKETS = [0, 10e3]
    RATES = [0, 0.1]
    INCOME = [0, 5e3, 10e3, 10e3 + 1, 20e3]
    EXPECTED = [0, 0, 0, 0.1, 1e3]
    res = mdf.tax_from_mtrs(INCOME, BRACKETS, RATES)
    pd.testing.assert_series_equal(res, pd.Series(EXPECTED))
    # Try with 10% avoidance.
    EXPECTED_10PCT_AVOIDANCE = [0, 0, 0, 0, 800.0]
    res_10pct_avoidance = mdf.tax_from_mtrs(INCOME, BRACKETS, RATES, 0.1)
    pd.testing.assert_series_equal(
        res_10pct_avoidance, pd.Series(EXPECTED_10PCT_AVOIDANCE)
    )
    # Try with avoidance elasticity of 2.
    EXPECTED_E2_AVOIDANCE = [
        0,
        0,
        0,
        0,  # Taxable base becomes (10e3 + 1) * (1 - 2 * 0.1)
        # Taxable base becomes 20e3 * (exp(-2 * 0.1)).
        0.1 * (20e3 * np.exp(-0.2) - 10e3),
    ]
    res_e2_avoidance = mdf.tax_from_mtrs(
        INCOME, BRACKETS, RATES, avoidance_elasticity=2
    )
    pd.testing.assert_series_equal(
        res_e2_avoidance, pd.Series(EXPECTED_E2_AVOIDANCE)
    )
    # Try with flat avoidance elasticity of 2.
    EXPECTED_E2_AVOIDANCE_FLAT = [
        0,
        0,
        0,
        0,  # Taxable base becomes (10e3 + 1) * (1 - 2 * 0.1)
        600.0,
    ]  # Taxable base becomes 20e3 * (1 - 2 * 0.1) = 16e3.
    res_e2_avoidance_flat = mdf.tax_from_mtrs(
        INCOME, BRACKETS, RATES, avoidance_elasticity_flat=2
    )
    pd.testing.assert_series_equal(
        res_e2_avoidance_flat, pd.Series(EXPECTED_E2_AVOIDANCE_FLAT)
    )
    # Ensure error when passing both rate and elasticity.
    with pytest.raises(Exception):
        mdf.tax_from_mtrs(INCOME, BRACKETS, RATES, 0.1, 2)
