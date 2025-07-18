from .agg import agg, combine_base_reform, pctchg_base_reform
from .concat import concat
from .constants import (
    BENS,
    ECI_REMOVE_COLS,
    HOUSING_CASH_SHARE,
    MCAID_CASH_SHARE,
    MCARE_CASH_SHARE,
    MED_BENS,
    OTHER_CASH_SHARE,
    SNAP_CASH_SHARE,
    SSI_CASH_SHARE,
    TANF_CASH_SHARE,
    VET_CASH_SHARE,
    WIC_CASH_SHARE,
)
from .custom_taxes import (
    CARBON_TAX_INCIDENCE,
    FTT_INCIDENCE,
    VAT_INCIDENCE,
    add_carbon_tax,
    add_custom_tax,
    add_ftt,
    add_vat,
)
from .income_measures import cash_income, market_income, tpc_eci
from .inequality import (
    bottom_50_pct_share,
    bottom_x_pct_share,
    gini,
    t10_b50,
    top_0_1_pct_share,
    top_1_pct_share,
    top_10_pct_share,
    top_50_pct_share,
    top_x_pct_share,
)
from .io import read_stata_zip
from .microdataframe import MicroDataFrame, MicroDataFrameGroupBy
from .microseries import MicroSeries, MicroSeriesGroupBy
from .poverty import (
    deep_poverty_gap,
    deep_poverty_rate,
    fpl,
    poverty_gap,
    poverty_rate,
    squared_poverty_gap,
)
from .tax import mtr, tax_from_mtrs
from .taxcalc import (
    add_weighted_metrics,
    calc_df,
    n65,
    recalculate,
    static_baseline_calc,
)
from .ubi import ubi_or_bens
from .utils import (
    cartesian_product,
    dedup_list,
    flatten,
    listify,
    ordinal_label,
)
from .weighted import (
    add_weighted_quantiles,
    quantile_chg,
    weight,
    weighted_mean,
    weighted_median,
    weighted_quantile,
    weighted_sum,
)

name = "microdf"
__version__ = "0.1.0"

__all__ = [
    # agg.py
    "combine_base_reform",
    "pctchg_base_reform",
    "agg",
    # concat.py
    "concat",
    # constants.py
    "BENS",
    "ECI_REMOVE_COLS",
    "HOUSING_CASH_SHARE",
    "MCAID_CASH_SHARE",
    "MCARE_CASH_SHARE",
    "MED_BENS",
    "OTHER_CASH_SHARE",
    "SNAP_CASH_SHARE",
    "SSI_CASH_SHARE",
    "TANF_CASH_SHARE",
    "VET_CASH_SHARE",
    "WIC_CASH_SHARE",
    # custom_taxes.py
    "CARBON_TAX_INCIDENCE",
    "FTT_INCIDENCE",
    "VAT_INCIDENCE",
    "add_custom_tax",
    "add_vat",
    "add_carbon_tax",
    "add_ftt",
    # income_measures.py
    "cash_income",
    "tpc_eci",
    "market_income",
    # inequality.py
    "gini",
    "top_x_pct_share",
    "bottom_x_pct_share",
    "bottom_50_pct_share",
    "top_10_pct_share",
    "top_1_pct_share",
    "top_0_1_pct_share",
    "top_50_pct_share",
    "t10_b50",
    # io.py
    "read_stata_zip",
    # poverty.py
    "fpl",
    "poverty_rate",
    "deep_poverty_rate",
    "poverty_gap",
    "squared_poverty_gap",
    "deep_poverty_gap",
    # tax.py
    "mtr",
    "tax_from_mtrs",
    # taxcalc.py
    "static_baseline_calc",
    "add_weighted_metrics",
    "n65",
    "calc_df",
    "recalculate",
    # ubi.py
    "ubi_or_bens",
    # utils.py
    "ordinal_label",
    "dedup_list",
    "listify",
    "flatten",
    "cartesian_product",
    # weighted.py
    "weight",
    "weighted_sum",
    "weighted_mean",
    "weighted_quantile",
    "weighted_median",
    "add_weighted_quantiles",
    "quantile_chg",
    # microseries.py
    "MicroSeries",
    "MicroSeriesGroupBy",
    # microdataframe.py
    "MicroDataFrame",
    "MicroDataFrameGroupBy",
]
