from typing import Optional

import pandas as pd
import taxcalc

import microdf as mdf
from microdf._optional import import_optional_dependency


def static_baseline_calc(
    recs: pd.DataFrame, year: int
) -> "taxcalc.Calculator":
    """Creates a static Calculator object.

    :param recs: Records object.
    :param year: Year to advance to.
    :returns: Calculator object.
    """
    tc = import_optional_dependency("taxcalc")
    calc = tc.Calculator(records=recs, policy=tc.Policy())
    calc.advance_to_year(year)
    calc.calc_all()
    return calc


def add_weighted_metrics(
    df: pd.DataFrame,
    metric_vars: list,
    w: str = "s006",
    divisor: float = 1e6,
    suffix: str = "_m",
) -> None:
    """Adds weighted metrics in millions to a Tax-Calculator pandas DataFrame.

    Columns are renamed to *_m.

    :param df: A pandas DataFrame containing Tax-Calculator data.
    :param metric_vars: A list of column names to weight, or a single column
        name.
    :param w: Weight column. Defaults to s006.
    :param divisor: Number by which the product is divided. Defaults to 1e6.
    :param suffix: Suffix to add to each weighted total. Defaults to '_m' to
        match divisor default of 1e6.
    :returns: Nothing. Weighted columns are added in place.
    """
    df[w + suffix] = df[w] / divisor
    metric_vars = mdf.listify(metric_vars)
    for metric_var in metric_vars:
        df[metric_var + suffix] = df[metric_var] * df[w + suffix]


def n65(
    age_head: pd.Series, age_spouse: pd.Series, elderly_dependents: pd.Series
) -> pd.Series:
    """Calculates number of people in the tax unit age 65 or older.

    :param age_head: Series representing age_head from taxcalc data.
    :param age_spouse: Series representing age_spouse from taxcalc data.
    :param elderly_dependents: Series representing elderly_dependents from
        taxcalc data.
    :returns: Series representing the number of people age 65 or older.
    """
    return (
        (age_head >= 65).astype(int)
        + (age_spouse >= 65).astype(int)
        + elderly_dependents
    )


def calc_df(
    records: Optional[pd.DataFrame] = None,
    policy: Optional[taxcalc.Policy] = None,
    year: int = 2020,
    reform: Optional[dict] = None,
    group_vars: Optional[list] = None,
    metric_vars: Optional[list] = None,
    group_n65: Optional[bool] = False,
) -> pd.DataFrame:
    """Creates a pandas DataFrame for given Tax-Calculator data.

    s006 is always included, and RECID is used as an index.

    :param records: An optional Records object. If not provided, uses CPS
        records. (Default value = None)
    :param policy: An optional Policy object. If not provided, uses default
        Policy.
    :param year: An optional year to advance to. If not provided, defaults to
        2020.
    :param reform: An optional reform to implement for the Policy object.
        (Default value = None)
    :param group_vars: An optional list of column names to include in the
        DataFrame. (Default value = None)
    :param metric_vars: An optional list of column names to include and
        calculate weighted sums of (in millions named as *_m) in the DataFrame.
        (Default value = None)
    :param group_n65: Whether to calculate and group by n65. Defaults to False.
    :returns: A pandas DataFrame. market_income is also always calculated.
    """
    tc = import_optional_dependency("taxcalc")
    # Assign defaults.
    if records is None:
        records = tc.Records.cps_constructor()
    if policy is None:
        policy = tc.Policy()
    if reform is not None:
        policy.implement_reform(reform)
    # Calculate.
    calc = tc.Calculator(records=records, policy=policy, verbose=False)
    calc.advance_to_year(year)
    calc.calc_all()
    # Get a deduplicated list of all columns.
    if group_n65:
        group_vars = group_vars + [
            "age_head",
            "age_spouse",
            "elderly_dependents",
        ]
    # Include expanded_income and benefits to produce market_income.
    all_cols = mdf.listify(
        [
            "RECID",
            "s006",
            "expanded_income",
            "aftertax_income",
            mdf.BENS,
            group_vars,
            metric_vars,
        ]
    )
    df = calc.dataframe(all_cols)
    # Create core elements.
    df["market_income"] = mdf.market_income(df)
    df["bens"] = df[mdf.BENS].sum(axis=1)
    df["tax"] = df.expanded_income - df.aftertax_income
    if group_n65:
        df["n65"] = n65(df.age_head, df.age_spouse, df.elderly_dependents)
        df.drop(
            ["age_head", "age_spouse", "elderly_dependents"],
            axis=1,
            inplace=True,
        )
    # Add calculated columns for metrics.
    mdf.add_weighted_metrics(df, metric_vars)
    # Set RECID to int and set it as index before returning.
    df["RECID"] = df.RECID.map(int)
    return df.set_index("RECID")


def recalculate(df: pd.DataFrame) -> None:
    """Recalculates fields in the DataFrame for after components have changed.

    :param df: DataFrame for use in microdf.
    :returns: Nothing. Updates the DataFrame in place.
    """
    # Recalculate TPC's Expanded Cash Income measure.
    cols = df.columns
    if "tpc_eci" in cols:
        df.tpc_eci = mdf.tpc_eci(df)
    # Recalculate weighted metrics (anything ending in _m).
    mcols = cols[cols.str.endswith("_m")]
    mdf.add_weighted_metrics(df, mcols)
