import numpy as np
import pandas as pd


def fpl(people: int):
    """Calculates the federal poverty guideline for a household of a certain
       size.

    :param XTOT: The number of people in the household.
    :param people: returns: The federal poverty guideline for the contiguous
      48 states.
    :returns: The federal poverty guideline for the contiguous 48 states.

    """
    return 7820 + 4320 * people


def poverty_rate(
    df: pd.DataFrame, income: str, threshold: str, w: str = None
) -> float:
    """Calculate poverty rate, i.e., the population share with income
       below their poverty threshold.

    :param df: DataFrame with income, threshold, and possibly weight columns
        for each person/household.
    :type df: pd.DataFrame
    :param income: Column indicating income.
    :type income: str
    :param threshold: Column indicating threshold.
    :type threshold: str
    :param w: Column indicating weight, defaults to None (unweighted).
    :type w: str, optional
    :return: Poverty rate between zero and one.
    :rtype: float
    """
    pov = df[income] < df[threshold]
    if w is None:
        return pov.mean()
    return (pov * df[w]).sum() / df[w].sum()


def deep_poverty_rate(
    df: pd.DataFrame, income: str, threshold: str, w: str = None
) -> float:
    """Calculate deep poverty rate, i.e., the population share with income
       below half their poverty threshold.

    :param df: DataFrame with income, threshold, and possibly weight columns
        for each person/household.
    :type df: pd.DataFrame
    :param income: Column indicating income.
    :type income: str
    :param threshold: Column indicating threshold.
    :type threshold: str
    :param w: Column indicating weight, defaults to None (unweighted).
    :type w: str, optional
    :return: Deep poverty rate between zero and one.
    :rtype: float
    """
    pov = df[income] < df[threshold] / 2
    if w is None:
        return pov.mean()
    return (pov * df[w]).sum() / df[w].sum()


def poverty_gap(
    df: pd.DataFrame, income: str, threshold: str, w: str = None
) -> float:
    """Calculate poverty gap, i.e., the total gap between income and poverty
       thresholds for all people in poverty.

    :param df: DataFrame with income, threshold, and possibly weight columns
        for each household (data should represent households, not persons).
    :type df: pd.DataFrame
    :param income: Column indicating income.
    :type income: str
    :param threshold: Column indicating threshold.
    :type threshold: str
    :param w: Column indicating weight, defaults to None (unweighted).
    :type w: str, optional
    :return: Poverty gap.
    :rtype: float
    """
    gap = np.maximum(df[threshold] - df[income], 0)
    if w is None:
        return gap.sum()
    return (gap * df[w]).sum()


def squared_poverty_gap(
    df: pd.DataFrame, income: str, threshold: str, w: str = None
) -> float:
    """Calculate squared poverty gap, i.e., the total squared gap between
       income and poverty thresholds for all people in poverty.
       Also known as poverty severity index.

    :param df: DataFrame with income, threshold, and possibly weight columns
        for each household (data should represent households, not persons).
    :type df: pd.DataFrame
    :param income: Column indicating income.
    :type income: str
    :param threshold: Column indicating threshold.
    :type threshold: str
    :param w: Column indicating weight, defaults to None (unweighted).
    :type w: str, optional
    :return: Squared poverty gap.
    :rtype: float
    """
    gap = np.maximum(df[threshold] - df[income], 0)
    sq_gap = np.power(gap, 2)
    if w is None:
        return sq_gap.sum()
    return (sq_gap * df[w]).sum()


def deep_poverty_gap(
    df: pd.DataFrame, income: str, threshold: str, w: str = None
) -> float:
    """Calculate deep poverty gap, i.e., the total gap between income and
       halved poverty thresholds for all people in deep poverty.

    :param df: DataFrame with income, threshold, and possibly weight columns
        for each household (data should represent households, not persons).
    :type df: pd.DataFrame
    :param income: Column indicating income.
    :type income: str
    :param threshold: Column indicating threshold.
    :type threshold: str
    :param w: Column indicating weight, defaults to None (unweighted).
    :type w: str, optional
    :return: Deep poverty gap.
    :rtype: float
    """
    gap = np.maximum((df[threshold] / 2) - df[income], 0)
    if w is None:
        return gap.sum()
    return (gap * df[w]).sum()
