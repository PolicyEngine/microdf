def dollar_format(suffix=""):
    """Dollar formatter for matplotlib.

    :param suffix: Suffix to append, e.g. 'B'. Defaults to ''.
    :returns: FuncFormatter.

    """
    return currency_format(currency="USD", suffix=suffix)


def currency_format(currency="USD", suffix=""):
    """Currency formatter for matplotlib.

    :param currency: Name of the currency, e.g. 'USD', 'GBP'.
    :param suffix: Suffix to append, e.g. 'B'. Defaults to ''.
    :returns: FuncFormatter.

    """
    try:
        import matplotlib as mpl
    except ImportError:
        raise ImportError(
            "The function you've called requires extra dependencies. "
            + "Please install microdf with the 'charts' extra by running "
            + "'pip install microdf[charts]'"
        )

    prefix = {"USD": "$", "GBP": "£"}[currency]

    return mpl.ticker.FuncFormatter(lambda x, _: prefix + format(int(x), ",") + suffix)
