import inspect

import pandas as pd

import microdf as mdf
from microdf.microdataframe import MicroDataFrame


def concat(*args, **kwargs) -> "MicroDataFrame":
    """Concatenates MicroDataFrame objects, preserving weights. If
    concatenating horizontally, the first set of weights are used. All args and
    kwargs are passed to pd.concat.

    :return: MicroDataFrame with concatenated weights.
    :rtype: mdf.MicroDataFrame
    """
    # Extract args with respect to pd.concat.
    pd_args = inspect.getcallargs(pd.concat, *args, **kwargs)
    objs = pd_args["objs"]
    axis = pd_args["axis"]
    # Create result, starting with pd.concat.
    res = mdf.MicroDataFrame(pd.concat(*args, **kwargs))
    # Assign weights depending on axis.
    if axis == 0:
        res.weights = pd.concat([obj.weights for obj in objs])
    else:
        # If concatenating horizontally, use the first set of weights.
        res.weights = objs[0].weights
    return res
