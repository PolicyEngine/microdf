from typing import Optional

import pandas as pd

import microdf as mdf


def combine_base_reform(
    base: pd.DataFrame,
    reform: pd.DataFrame,
    base_cols: Optional[list] = None,
    cols: Optional[list] = None,
    reform_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Combine base and reform with certain columns.

    :param base: Base DataFrame. Index must match reform.
    :type base: pd.DataFrame
    :param reform: Reform DataFrame. Index must match base.
    :type reform: pd.DataFrame
    :param base_cols: Columns in base to keep.
    :type base_cols: list, optional
    :param cols: Columns to keep from both base and reform.
    :type cols: list, optional
    :param reform_cols: Columns in reform to keep.
    :type reform_cols: list, optional
    :returns: DataFrame with columns for base ("_base") and reform ("_reform").
    :rtype: pd.DataFrame
    """
    all_base_cols = mdf.listify([base_cols] + [cols])
    all_reform_cols = mdf.listify([reform_cols] + [cols])
    return base[all_base_cols].join(
        reform[all_reform_cols], lsuffix="_base", rsuffix="_reform"
    )


def pctchg_base_reform(combined: pd.DataFrame, metric: str) -> pd.Series:
    """Calculates the percentage change in a metric for a combined dataset.

    :param combined: Combined DataFrame with _base and _reform columns.
    :type combined: pd.DataFrame
    :param metric: String of the column to calculate the difference. Must exist
        as metric_m_base and metric_m_reform in combined.
    :type metric: str
    :returns: Series with percentage change.
    :rtype: pd.Series
    """
    return combined[metric + "_m_reform"] / combined[metric + "_m_base"] - 1


def agg(
    base: pd.DataFrame,
    reform: pd.DataFrame,
    groupby: str,
    metrics: list,
    base_metrics: Optional[list] = None,
    reform_metrics: Optional[list] = None,
) -> pd.DataFrame:
    """Aggregates differences between base and reform.

    :param base: Base DataFrame. Index must match reform.
    :type base: pd.DataFrame
    :param reform: Reform DataFrame. Index must match base.
    :type reform: pd.DataFrame
    :param groupby: Variable in base to group on.
    :type groupby: str
    :param metrics: List of variables to agg and calculate the % change of.
        These should have associated weighted columns ending in _m in base and
        reform.
    :type metrics: list
    :param base_metrics: List of variables from base to sum.
    :type base_metrics: Optional[list]
    :param reform_metrics: List of variables from reform to sum.
    :type reform_metrics: Optional[list]
    :returns: DataFrame with groupby and metrics, and _pctchg metrics.
    :rtype: pd.DataFrame
    """
    metrics = mdf.listify(metrics)
    metrics_m = [i + "_m" for i in metrics]
    combined = combine_base_reform(
        base,
        reform,
        base_cols=mdf.listify([groupby, base_metrics]),
        cols=mdf.listify(metrics_m),
        reform_cols=mdf.listify(reform_metrics),
    )
    grouped = combined.groupby(groupby).sum()
    for metric in metrics:
        grouped[metric + "_pctchg"] = pctchg_base_reform(grouped, metric)
    return grouped
