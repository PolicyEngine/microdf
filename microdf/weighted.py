import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import microdf as mdf


def weight(
    df: pd.DataFrame, col: Union[str, List[str]], w: Optional[str] = None
) -> pd.Series:
    """Calculates the weighted value of a column in a DataFrame.

    :param df: A pandas DataFrame.
    :param col: A string indicating the column in the DataFrame to weight. Can
        also be a list of column strings.
    :param w: Weight column.
    :returns: A pandas Series multiplying the column by its weight.
    """
    if w is None:
        return df[col]
    return df[col].multiply(df[w], axis="index")


def weighted_sum(
    df: pd.DataFrame,
    col: Union[str, List[str]],
    w: Optional[str] = None,
    groupby: Optional[str] = None,
) -> pd.Series:
    """Calculates the weighted sum of a column in a DataFrame.

    :param df: A pandas DataFrame.
    :param col: A string indicating the column in the DataFrame. Can also be a
        list of column strings.
    :param w: Weight column.
    :param groupby: Groupby column.
    :returns: The weighted sum of a DataFrame's column.
    """

    def _weighted_sum(
        df: pd.DataFrame, col: Union[str, List[str]], w: Optional[str]
    ) -> float:
        """For weighted sum with provided weight."""
        return weight(df, col, w).sum()

    if groupby is None:
        if w is None:
            return df[col].sum()
        return _weighted_sum(df, col, w)
    # If grouping.
    if w is None:
        return df.groupby(groupby)[col].sum()
    return df.groupby(groupby).apply(lambda x: _weighted_sum(x, col, w))


def weighted_mean(
    df: pd.DataFrame,
    col: Union[str, List[str]],
    w: Optional[str] = None,
    groupby: Optional[str] = None,
) -> pd.Series:
    """Calculates the weighted mean of a column in a DataFrame.

    :param df: A pandas DataFrame.
    :param col: A string indicating the column in the DataFrame. Can also be a
        list of column strings.
    :param w: Weight column.
    :param groupby: Groupby column.
    :returns: The weighted mean of a DataFrame's column.
    """

    def _weighted_mean(
        df: pd.DataFrame, col: Union[str, List[str]], w: Optional[str]
    ) -> float:
        """For weighted mean with provided weight."""
        return weighted_sum(df, col, w) / df[w].sum()

    if groupby is None:
        if w is None:
            return df[col].mean()
        return _weighted_mean(df, col, w)
    # Group.
    if w is None:
        return df.groupby(groupby)[col].mean()
    return df.groupby(groupby).apply(lambda x: _weighted_mean(x, col, w))


def weighted_quantile(
    df: pd.DataFrame, col: str, w: str, quantiles: np.array
) -> np.array:
    """Calculates weighted quantiles of a set of values.

    Doesn't exactly match unweighted quantiles of stacked values.
    See stackoverflow.com/q/21844024#comment102342137_29677616.

    :param df: DataFrame to calculate weighted quantiles from.
    :type df: pd.DataFrame
    :param col: Name of numeric column in df to calculate weighted quantiles
        from.
    :type col: str
    :param w: Name of weight column in df.
    :type w: str
    :param quantiles: Array of quantiles to calculate.
    :type quantiles: np.array
    :return: Array of weighted quantiles.
    :rtype: np.array
    """
    values = np.array(df[col])
    quantiles = np.array(quantiles)
    if w is None:
        sample_weight = np.ones(len(values))
    else:
        sample_weight = np.array(df[w])
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def weighted_median(
    df: pd.DataFrame,
    col: Union[str, List[str]],
    w: Optional[str] = None,
    groupby: Optional[str] = None,
) -> pd.Series:
    """Calculates the weighted median of a column in a DataFrame.

    :param df: A pandas DataFrame containing Tax-Calculator data.
    :param col: A string indicating the column in the DataFrame.
    :param w: Weight column.
    :returns: The weighted median of a DataFrame's column.
    """

    def _weighted_median(
        df: pd.DataFrame, col: Union[str, List[str]], w: Optional[str]
    ) -> float:
        """For weighted median with provided weight."""
        return weighted_quantile(df, col, w, 0.5)

    if groupby is None:
        if w is None:
            return df[col].median()
        return _weighted_median(df, col, w)
    # Group.
    if w is None:
        return df.groupby(groupby)[col].median()
    return df.groupby(groupby).apply(lambda x: _weighted_median(x, col, w))


def add_weighted_quantiles(
    df: pd.DataFrame, col: Union[str, List[str]], w: Optional[str] = None
) -> None:
    """Adds weighted quantiles of a column to a DataFrame. This will be
    deprecated in the next minor release. Please use MicroSeries.rank instead.

    Adds columns for each of these types of quantiles to a DataFrame:
    * *_percentile_exact: Exact percentile.
    * *_percentile: Integer percentile (ceiling).
    * *_2percentile: Integer percentile (ceiling, for each two percentiles).
    * *_ventile: Integer percentile (ceiling, for each five percentiles).
    * *_decile: Integer decile.
    * *_quintile: Integer quintile.
    * *_quartile: Integer quartile.

    Negative values are assigned -1.

    :param df: A pandas DataFrame.
    :param col: A string indicating the column in the DataFrame to calculate.
    :param w: Weight column.
    :returns: Nothing. Columns are added in place. Also sorts df by col.
    """
    warnings.warn(
        "This will be deprecated in the next minor release. "
        "Please use MicroSeries.rank instead.",
        DeprecationWarning,
    )
    df.sort_values(by=col, inplace=True)
    col_pctile = col + "_percentile_exact"
    df[col_pctile] = 100 * df[w].cumsum() / df[w].sum()
    # "Null out" negatives using -1, since integer arrays can't be NaN.
    df[col_pctile] = np.where(df[col] >= 0, df[col_pctile], 0)
    # Reduce top record, otherwise it's incorrectly rounded up.
    df[col_pctile] = np.where(
        df[col_pctile] >= 99.99999, 99.99999, df[col_pctile]
    )
    df[col + "_percentile"] = np.ceil(df[col_pctile]).astype(int)
    df[col + "_2percentile"] = 2 * np.ceil(df[col_pctile] / 2).astype(int)
    df[col + "_ventile"] = 5 * np.ceil(df[col_pctile] / 5).astype(int)
    df[col + "_decile"] = np.ceil(df[col_pctile] / 10).astype(int)
    df[col + "_quintile"] = np.ceil(df[col_pctile] / 20).astype(int)
    df[col + "_quartile"] = np.ceil(df[col_pctile] / 25).astype(int)


def quantile_chg(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    col1: str,
    col2: str,
    w1: Optional[str] = None,
    w2: Optional[str] = None,
    q: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Create table with two sets of quantiles.

    :param df1: DataFrame with first set of values.
    :param df2: DataFrame with second set of values.
    :param col1: Name of columns with values in df1.
    :param col2: Name of columns with values in df2.
    :param w1: Name of weight column in df1.
    :param w2: Name of weight column in df2.
    :param q: Quantiles. Defaults to decile boundaries.
    :returns: DataFrame with two rows and a column for each quantile. Column
        labels are "xth percentile" and a label is added to the median.
    """
    if q is None:
        q = np.arange(0.1, 1, 0.1)
    q1 = weighted_quantile(df1, col1, w1, q)
    q2 = weighted_quantile(df2, col2, w2, q)
    qdf = pd.DataFrame([q1, q2])
    # Set decile labels.
    q_print = [mdf.ordinal_label((i * 100)) for i in q]
    try:  # List index throws an error if the value is not found.
        median_index = q.tolist().index(0.5)
        q_print[median_index] += " (median)"
    except ValueError:
        pass  # Don't assign median to any label.
    qdf.columns = q_print
    return qdf
