import pytest

import microdf as mdf

try:
    import taxcalc as tc

    _HAVE_TAXCALC = True
except ImportError:
    _HAVE_TAXCALC = False


def test_calc_df() -> None:
    """"""
    if not _HAVE_TAXCALC:
        pytest.skip("taxcalc is not installed")
    mdf.calc_df()


def test_static_baseline_calc() -> None:
    """"""
    if not _HAVE_TAXCALC:
        pytest.skip("taxcalc is not installed")
    recs = tc.Records.cps_constructor()
    mdf.static_baseline_calc(recs, 2020)
