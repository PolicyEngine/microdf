import microdf as mdf


def test_read_stata_zip() -> None:
    """"""
    SCF2016 = "https://www.federalreserve.gov/econres/files/scfp2016s.zip"
    COLS = ["wgt", "networth"]
    df = mdf.read_stata_zip(SCF2016, columns=COLS)
    assert df.columns.tolist() == COLS
    assert df.shape[0] > 0
