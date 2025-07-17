import io
import zipfile

import pandas as pd
import requests

HEADER = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "
    + "AppleWebKit/537.36 (KHTML, like Gecko) "
    + "Chrome/50.0.2661.102 Safari/537.36"
}


def read_stata_zip(url: str, **kwargs) -> pd.DataFrame:
    """Reads zipped Stata file by URL.

    From https://stackoverflow.com/a/59122689/1840471

    Pending native support in
    https://github.com/pandas-dev/pandas/issues/26599.

    :param url: URL string of .zip file containing a single .dta file. :param
        **kwargs: Arguments passed to pandas.read_stata().
    :returns: DataFrame.
    """
    r = requests.get(url, headers=HEADER)
    data = io.BytesIO(r.content)
    with zipfile.ZipFile(data) as archive:
        with archive.open(archive.namelist()[0]) as stata:
            return pd.read_stata(stata, **kwargs)
