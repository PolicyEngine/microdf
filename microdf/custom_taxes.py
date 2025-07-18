"""Functions and data for estimating taxes outside the income tax system.

Examples include value added tax, financial transaction tax, and carbon tax.
"""

from typing import Optional

import numpy as np
import pandas as pd

import microdf as mdf

# Source:
# https://www.taxpolicycenter.org/briefing-book/who-would-bear-burden-vat
VAT_INCIDENCE = pd.Series(
    index=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9],
    data=[3.9, 3.9, 3.6, 3.6, 3.6, 3.6, 3.6, 3.4, 3.4, 3.2, 2.8, 2.5, 2.5],
)
VAT_INCIDENCE /= 100

# Source: Table 5 in
# https://www.treasury.gov/resource-center/tax-policy/tax-analysis/Documents/WP-115.pdf
CARBON_TAX_INCIDENCE = pd.Series(
    index=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9],
    data=[0.8, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.8, 1.8, 1.8, 1.6, 1.4, 0.7],
)
CARBON_TAX_INCIDENCE /= 100

# Source: Figure 1 in
# https://www.taxpolicycenter.org/sites/default/files/alfresco/publication-pdfs/2000587-financial-transaction-taxes.pdf
FTT_INCIDENCE = pd.Series(
    index=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9],
    data=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.8, 1.0],
)
FTT_INCIDENCE /= 100


def add_custom_tax(
    df: pd.DataFrame,
    segment_income: str,
    w: str,
    base_income: str,
    incidence: pd.Series,
    name: str,
    total: Optional[float] = None,
    ratio: Optional[float] = None,
    verbose: Optional[bool] = True,
) -> None:
    """Add a custom tax based on incidence analysis driven by percentiles.

    :param df: DataFrame.
    :param segment_income: Income measure used to segment tax units into
        quantiles.
    :param w: Weight used to segment into quantiles (either s006 or XTOT_m).
    :param base_income: Income measure by which incidence is multiplied to
        estimate liability.
    :param incidence: pandas Series indexed on the floor of an income
        percentile, with values for the tax rate.
    :param name: Name of the column to add.
    :param total: Total amount the tax should generate. If not provided,
        liabilities are calculated only based on the incidence schedule.
        (Default value = None)
    :param ratio: Ratio to adjust the tax by, compared to the original tax.
        This acts as a multiplier for the incidence argument. (Default value =
        None)
    :param verbose: Whether to print the tax adjustment factor if needed.
        Defaults to True.
    :returns: Nothing. Adds the column name to df representing the tax
        liability. df is also sorted by segment_income.
    """
    if ratio is not None:
        incidence = incidence * ratio
        assert total is None, "ratio and total cannot both be provided."
    df.sort_values(segment_income, inplace=True)
    income_percentile = 100 * df[w].cumsum() / df[w].sum()
    tu_incidence = incidence.iloc[
        pd.cut(
            income_percentile,
            # Add a right endpoint. Should be 100 but sometimes a decimal
            # gets added.
            bins=incidence.index.tolist() + [101],
            labels=False,
        )
    ].values
    df[name] = np.maximum(0, tu_incidence * df[base_income])
    if total is not None:
        initial_total = mdf.weighted_sum(df, name, "s006")
        if verbose:
            print(
                "Multiplying tax by "
                + str(round(total / initial_total, 2))
                + "."
            )
        df[name] *= total / initial_total


def add_vat(
    df: pd.DataFrame,
    segment_income: Optional[str] = "tpc_eci",
    w: Optional[str] = "XTOT_m",
    base_income: Optional[str] = "aftertax_income",
    incidence: Optional[pd.Series] = VAT_INCIDENCE,
    name: Optional[str] = "vat",
    **kwargs,
) -> None:
    """Add value added tax based on incidence estimate from Tax Policy Center.

    :param df: DataFrame with columns for tpc_eci, XTOT_m, and aftertax_income.
    :param Other: arguments: Args to add_custom_tax with VAT defaults.
    :param segment_income: Default value = "tpc_eci")
    :param w: Default value = "XTOT_m")
    :param base_income: Default value = "aftertax_income")
    :param incidence: Default value = VAT_INCIDENCE)
    :param name: Default value = "vat") :param **kwargs: Other arguments passed
        to add_custom_tax().
    :returns: Nothing. Adds vat to df. df is also sorted by tpc_eci.
    """
    add_custom_tax(
        df, segment_income, w, base_income, incidence, name, **kwargs
    )


def add_carbon_tax(
    df: pd.DataFrame,
    segment_income: Optional[str] = "tpc_eci",
    w: Optional[str] = "XTOT_m",
    base_income: Optional[str] = "aftertax_income",
    incidence: Optional[pd.Series] = CARBON_TAX_INCIDENCE,
    name: Optional[str] = "carbon_tax",
    **kwargs,
) -> None:
    """Add carbon tax based on incidence estimate from the US Treasury
    Department.

    :param df: DataFrame with columns for tpc_eci, XTOT_m, and aftertax_income.
    :param Other: arguments: Args to add_custom_tax with carbon tax defaults.
    :param segment_income: Default value = "tpc_eci")
    :param w: Default value = "XTOT_m")
    :param base_income: Default value = "aftertax_income")
    :param incidence: Default value = CARBON_TAX_INCIDENCE)
    :param name: Default value = "carbon_tax") :param **kwargs: Other arguments
        passed to add_custom_tax().
    :returns: Nothing. Adds carbon_tax to df. df is also sorted by tpc_eci.
    """
    add_custom_tax(
        df, segment_income, w, base_income, incidence, name, **kwargs
    )


def add_ftt(
    df: pd.DataFrame,
    segment_income: Optional[str] = "tpc_eci",
    w: Optional[str] = "XTOT_m",
    base_income: Optional[str] = "aftertax_income",
    incidence: Optional[pd.Series] = FTT_INCIDENCE,
    name: Optional[str] = "ftt",
    **kwargs,
) -> None:
    """Add financial transaction tax based on incidence estimate from Tax
    Policy Center.

    :param df: DataFrame with columns for tpc_eci, XTOT_m, and aftertax_income.
    :param Other: arguments: Args to add_custom_tax with FTT defaults.
    :param segment_income: Default value = "tpc_eci")
    :param w: Default value = "XTOT_m")
    :param base_income: Default value = "aftertax_income")
    :param incidence: Default value = FTT_INCIDENCE)
    :param name: Default value = "ftt") :param **kwargs: Other arguments passed
        to add_custom_tax().
    :returns: Nothing. Adds ftt to df. df is also sorted by tpc_eci.
    """
    add_custom_tax(
        df, segment_income, w, base_income, incidence, name, **kwargs
    )
