import os

import numpy as np
import pandas as pd

import microdf as mdf


def differences(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    f_actual: str,
    f_expected: str,
) -> None:
    """Check for differences between results in afilename and efilename files.

    :param actual: Actual DataFrame.
    :param expected: Expected DataFrame.
    :param f_actual: Filename of the actual CSV.
    :param f_expected: Filename of the expected CSV.
    """
    if not np.allclose(actual, expected):
        msg = "COMPARE RESULTS DIFFER\n"
        msg += "-------------------------------------------------\n"
        msg += "--- NEW RESULTS IN {} FILE ---\n"
        msg += "--- if new OK, copy {} to  ---\n"
        msg += "---                 {}     ---\n"
        msg += "---            and rerun test.                ---\n"
        msg += "-------------------------------------------------\n"
        raise ValueError(msg.format(f_actual, f_actual, f_expected))


def test_percentile_agg_compare(tests_path: str) -> None:
    """
    :param tests_path: Folder path to write test results.
    """
    N = 1000
    np.random.seed(0)
    df = pd.DataFrame({"val": np.random.rand(N), "w": np.random.rand(N)})
    mdf.add_weighted_quantiles(df, "val", "w")
    percentile_sum = df.groupby("val_percentile")[["val", "w"]].sum()
    F_ACTUAL = "test_percentile_actual.csv"
    F_EXPECTED = "test_percentile_expected.csv"
    percentile_sum.to_csv(os.path.join(tests_path, F_ACTUAL))
    # Re-read as CSV to remove index and ensure CSVs are equal.
    actual = pd.read_csv(os.path.join(tests_path, F_ACTUAL))
    expected = pd.read_csv(os.path.join(tests_path, F_EXPECTED))
    differences(actual, expected, F_ACTUAL, F_EXPECTED)
