import numpy as np
import pandas as pd


def mtr(val, brackets, rates):
    """Calculates the marginal tax rate applied to a value depending on a
    tax schedule.

    :param val: Value to assess tax on, e.g. wealth or income (list or Series).
    :param brackets: Left side of each bracket (list or Series).
    :param rates: Rate corresponding to each bracket.
    :returns: Series of the size of val representing the marginal tax rate.

    """
    df_tax = pd.DataFrame({"brackets": brackets, "rates": rates})
    df_tax["base_tax"] = (
        df_tax.brackets.sub(df_tax.brackets.shift(fill_value=0))
        .mul(df_tax.rates.shift(fill_value=0))
        .cumsum()
    )
    rows = df_tax.brackets.searchsorted(val, side="right") - 1
    income_bracket_df = df_tax.loc[rows].reset_index(drop=True)
    return income_bracket_df.rates


def tax_from_mtrs(
    val,
    brackets,
    rates,
    avoidance_rate=0,
    avoidance_elasticity=0,
    avoidance_elasticity_flat=0,
):
    """Calculates tax liability based on a marginal tax rate schedule.

    :param val: Value to assess tax on, e.g. wealth or income (list or Series).
    :param brackets: Left side of each bracket (list or Series).
    :param rates: Rate corresponding to each bracket.
    :param avoidance_rate: Constant avoidance/evasion rate in percentage terms.
                        Defaults to zero.
    :param avoidance_elasticity: Avoidance/evasion elasticity.
                              Response of log taxable value with respect
                              to tax rate.
                              Defaults to zero. Should be positive.
    :param avoidance_elasticity_flat: Response of taxable value with respect
                                   to tax rate.
                                   Use avoidance_elasticity in most cases.
                                   Defaults to zero. Should be positive.
    :returns: Series of tax liabilities with the same size as val.

    """
    assert (
        avoidance_rate == 0
        or avoidance_elasticity == 0
        or avoidance_elasticity_flat == 0
    ), "Cannot supply multiple avoidance parameters."
    assert (
        avoidance_elasticity >= 0
    ), "Provide nonnegative avoidance_elasticity."
    df_tax = pd.DataFrame({"brackets": brackets, "rates": rates})
    df_tax["base_tax"] = (
        df_tax.brackets.sub(df_tax.brackets.shift(fill_value=0))
        .mul(df_tax.rates.shift(fill_value=0))
        .cumsum()
    )
    if avoidance_rate == 0:  # Only need MTRs if elasticity is supplied.
        mtrs = mtr(val, brackets, rates)
    if avoidance_elasticity > 0:
        avoidance_rate = 1 - np.exp(-avoidance_elasticity * mtrs)
    if avoidance_elasticity_flat > 0:
        avoidance_rate = avoidance_elasticity_flat * mtrs
    taxable = pd.Series(val) * (1 - avoidance_rate)
    rows = df_tax.brackets.searchsorted(taxable, side="right") - 1
    income_bracket_df = df_tax.loc[rows].reset_index(drop=True)
    return (
        pd.Series(taxable)
        .sub(income_bracket_df.brackets)
        .mul(income_bracket_df.rates)
        .add(income_bracket_df.base_tax)
    )
