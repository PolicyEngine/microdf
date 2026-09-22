"""Concatenation with validation before pandas chooses an output class."""

from collections.abc import Mapping

import pandas as pd

from ._weights import WeightPropagationMixin


def concat(objs, **kwargs):
    """Concatenate weighted objects, rejecting plain inputs in either order.

    Accept pandas.concat keyword arguments. Every non-None input must carry
    weights; construct Micro objects with explicit weights before calling.
    """
    objs = dict(objs) if isinstance(objs, Mapping) else list(objs)
    keys = kwargs.get("keys")
    if keys is not None:
        keys = list(keys)
        kwargs["keys"] = keys
    selected = (
        [objs[key] for key in (objs if keys is None else keys)]
        if isinstance(objs, Mapping)
        else objs
    )
    if any(
        obj is not None and not isinstance(obj, WeightPropagationMixin)
        for obj in selected
    ):
        raise ValueError(
            "Cannot concatenate weighted and unweighted objects: provide explicit weights for every input."
        )
    return pd.concat(objs, **kwargs)
