"""Variance estimation from replicate weights.

Many survey products publish a set of replicate weight vectors alongside the
main weight. Recomputing a statistic once per replicate and measuring the
spread gives a variance estimate that requires no analytic formula, which is
what makes it usable for statistics such as the Gini coefficient or a
quantile where the analytic variance is awkward.

The scale factor depends on how the replicates were constructed, so the
method must be named rather than guessed.

Note that this is only valid for replicate weights as published with a
survey. Weights that have been calibrated or reweighted to external targets
no longer correspond to the original replication scheme, and applying these
estimators to them does not describe the variance of the resulting estimator.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Scale applied to the sum of squared deviations from the full-sample
# estimate. R is the number of replicates.
METHOD_FACTORS = {
    # Delete-a-group jackknife: (R - 1) / R.
    "jackknife": lambda r: (r - 1) / r,
    # Balanced repeated replication: 1 / R.
    "brr": lambda r: 1 / r,
    # Bootstrap replicates: 1 / R.
    "bootstrap": lambda r: 1 / r,
    # Successive difference replication, as used for the ACS and CPS: 4 / R.
    "successive-difference": lambda r: 4 / r,
}


def _fay_factor(r: int, fay_k: float) -> float:
    """Scale for Fay's variant of BRR, which perturbs rather than deletes."""
    if not 0 <= fay_k < 1:
        raise ValueError(f"fay_k must be in [0, 1), got {fay_k}")
    return 1 / (r * (1 - fay_k) ** 2)


def replicate_variance(
    series,
    statistic: Callable,
    replicate_weights: np.ndarray | pd.DataFrame,
    method: str = "jackknife",
    fay_k: float | None = None,
) -> float:
    """Variance of ``statistic`` estimated from replicate weights.

    :param series: A MicroSeries. Its own weights give the point estimate.
    :param statistic: Callable taking a MicroSeries and returning a float,
        for example ``lambda s: s.gini()``.
    :param replicate_weights: Array or frame of shape ``(len(series), R)``.
    :param method: One of ``jackknife``, ``brr``, ``bootstrap``,
        ``successive-difference``, or ``fay`` (which requires ``fay_k``).
    :param fay_k: Fay's perturbation constant, required when
        ``method="fay"``.
    :returns: The estimated variance of the statistic.
    """
    from microdf.microseries import MicroSeries

    weights = np.asarray(replicate_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError(
            f"replicate_weights must be 2-dimensional, got shape {weights.shape}"
        )
    if weights.shape[0] != len(series):
        raise ValueError(
            f"replicate_weights has {weights.shape[0]} rows but the series has "
            f"{len(series)}"
        )

    n_replicates = weights.shape[1]
    if n_replicates < 2:
        raise ValueError("At least two replicate weights are required")

    if method == "fay":
        if fay_k is None:
            raise ValueError("method='fay' requires fay_k")
        factor = _fay_factor(n_replicates, fay_k)
    elif method in METHOD_FACTORS:
        if fay_k is not None:
            raise ValueError("fay_k applies only to method='fay'")
        factor = METHOD_FACTORS[method](n_replicates)
    else:
        known = ", ".join(sorted([*METHOD_FACTORS, "fay"]))
        raise ValueError(f"Unknown method {method!r}; expected one of {known}")

    point = float(statistic(series))
    values = np.asarray(series, dtype=float)
    index = series.index

    deviations = []
    for column in range(n_replicates):
        replicate = MicroSeries(values, weights=weights[:, column], index=index)
        deviations.append(float(statistic(replicate)) - point)

    return factor * float(np.sum(np.square(deviations)))


def replicate_standard_error(
    series,
    statistic: Callable,
    replicate_weights: np.ndarray | pd.DataFrame,
    method: str = "jackknife",
    fay_k: float | None = None,
) -> float:
    """Standard error of ``statistic``, the square root of its variance.

    Takes the same arguments as :func:`replicate_variance`.
    """
    return float(
        np.sqrt(replicate_variance(series, statistic, replicate_weights, method, fay_k))
    )
