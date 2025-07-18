import pandas as pd

import microdf as mdf


def test_top_pct() -> None:
    x = list(range(1, 11))  # 1 to 10. Sum = 10 * 11 / 2 = 55.
    df = pd.DataFrame({"x": x})
    ms = mdf.MicroSeries(x)
    RES = 10 / 55
    assert mdf.top_10_pct_share(df, "x") == RES
    assert ms.top_10_pct_share() == RES
    x = list(range(1, 4))
    df = pd.DataFrame({"x": x, "w": x})
    ms = mdf.MicroSeries(x, weights=x)
    # This is equivalent to [1, 2, 2, 3, 3, 3]
    # Sum = 14, top half is 9.
    RES = 9 / 14
    assert mdf.top_50_pct_share(df, "x", "w") == RES
    assert ms.top_50_pct_share() == RES
