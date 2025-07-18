import pandas as pd

import microdf as mdf

V1 = [1, 2, 3]
V2 = [4, 5, 6]
W1 = [7, 8, 9]
W2 = [10, 11, 12]
DF1 = pd.DataFrame({"v": V1, "w": W1})
DF2 = pd.DataFrame({"v": V2, "w": W2})


def test_quantile_chg() -> None:
    mdf.quantile_chg(DF1, DF2, "v", "w", "v", "w")
