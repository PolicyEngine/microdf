import collections
from typing import Dict, List, Optional, Union

import pandas as pd


def ordinal_label(n: int) -> str:
    """Creates ordinal label from number.

    Adapted from https://stackoverflow.com/a/20007730/1840471.

    :param n: Number.
    :returns: Ordinal label, e.g., 1st, 3rd, 24th, etc.
    """
    n = int(n)
    ix = (n / 10 % 10 != 1) * (n % 10 < 4) * n % 10
    return "%d%s" % (n, "tsnrhtdd"[ix::4])


def dedup_list(lst: List) -> List:
    """Remove duplicate items from a list.

    :param lst: List.
    :returns: List with duplicate items removed from lst.
    """
    return list(set(lst))


def listify(
    x: Union[str, List[str]], dedup: Optional[bool] = True
) -> List[str]:
    """Return x as a list, if it isn't one already.

    :param x: A single item or a list
    :param dedup: Default value = True)
    :returns: x if x is a list, otherwise [x]. Also flattens the list and
        removes Nones.
    """
    if not isinstance(x, list):
        x = [x]
    res = flatten(x)
    res = [x for x in res if x is not None]
    if dedup:
        return dedup_list(res)
    return res


def flatten(lst: List) -> None:
    """Flatten list. From https://stackoverflow.com/a/2158532/1840471.

    :param lst: List.
    :returns: Flattened version.

    """
    for el in lst:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        else:
            yield el


def cartesian_product(d: Dict) -> pd.DataFrame:
    """Produces a DataFrame as a Cartesian product of dictionary keys and
    values.

    :param d: Dictionary where each item's key corresponds to a column name,
        and each value is a list of values.
    :returns: DataFrame with a Cartesian product of each dictionary item.
    """
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()
