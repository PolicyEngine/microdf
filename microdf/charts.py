import numpy as np

import microdf as mdf


def quantile_pct_chg_plot(df1, df2, col1, col2, w1=None, w2=None, q=None):
    """Create stem plot with percent change in decile boundaries.

    :param df1: DataFrame with first set of values.
    :param df2: DataFrame with second set of values.
    :param col1: Name of columns with values in df1.
    :param col2: Name of columns with values in df2.
    :param w1: Name of weight column in df1.
    :param w2: Name of weight column in df2.
    :param q: Quantiles. Defaults to decile boundaries.
    :returns: Axis.
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "The function you've called requires extra dependencies. "
            + "Please install microdf with the 'charts' extra by running "
            + "'pip install microdf[charts]'"
        )

    if q is None:
        q = np.arange(0.1, 1, 0.1)
    # Calculate weighted quantiles.
    df = mdf.quantile_chg(df1, df2, col1, col2, w1, w2, q).transpose()
    # Prepare dataset for plotting.
    df.columns = ["base", "reform"]
    df["pct_chg"] = df.reform / df.base - 1
    # Multiply by 100 pending github.com/matplotlib/matplotlib/issues/17113
    df.pct_chg *= 100
    df["index_newline"] = np.where(
        df.index == "50th (median)", "50th\n(median)", df.index
    )
    # Plot.
    fig, ax = plt.subplots()
    markerline, stemlines, baseline = ax.stem(df.index_newline, df.pct_chg)
    plt.setp(baseline, color="gray", linewidth=0)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100))
    plt.title("Change to percentiles", loc="left")
    plt.ylabel("Change at the percentile boundary")
    plt.xlabel("Percentile")
    sns.despine(left=True, bottom=True)
    ax.grid(color=mdf.GRID_COLOR, axis="y")
    plt.xticks(rotation=0)
    return ax
