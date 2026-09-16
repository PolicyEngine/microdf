"""Variance estimation from replicate weights.

Many survey products publish a set of replicate weight vectors alongside the
main weight. Recomputing a statistic once per replicate and measuring the
spread gives a variance estimate without an analytic variance formula. Its
statistical validity depends on both the statistic and the replication design.
Nonsmooth statistics such as quantiles can require an appropriate replication
method or smoothing; these functions apply the supplied statistic directly.

The scale factor and centering convention must match the survey's replication
design. The default centers on the full-sample estimate; ``replicate-mean``
centering is also available. These estimators support common-factor replication
schemes, not arbitrary stratified jackknife or averaged-bootstrap designs that
require additional or replicate-specific factors.

Changing the main weights without corresponding design-consistent adjustments
to the replicate weights invalidates the original replicates. Calibration can
be valid when the required calibration is repeated appropriately for every
replicate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Scale applied to the sum of squared deviations from the selected center.
# R is the number of replicates.
METHOD_FACTORS = {
    # Unstratified JK1 / common-factor delete-group jackknife: (R - 1) / R.
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
    *,
    centering: str = "full-sample",
) -> float:
    """Variance of ``statistic`` estimated from replicate weights.

    Variance is the method's scale factor times the sum of squared deviations
    of replicate estimates from the selected center. The supported factors
    are ``(R - 1) / R`` for unstratified JK1 or common-factor delete-group
    jackknife, ``1 / R`` for BRR and bootstrap, ``4 / R`` for successive
    difference, and ``1 / (R * (1 - fay_k)**2)`` for Fay's BRR. Select the
    factor and center specified by the survey; arbitrary stratified jackknife
    and averaged-bootstrap schemes requiring other factors are unsupported.
    Validity also depends on the statistic: nonsmooth quantiles can require
    an appropriate replication method or smoothing of replicate estimates.

    :param series: A MicroSeries. Its own weights give the point estimate.
    :param statistic: Callable taking a MicroSeries and returning a float,
        for example ``lambda s: s.gini()``.
    :param replicate_weights: Array or frame of shape ``(len(series), R)``
        in the same row order as ``series``. Rows are matched by position;
        DataFrame index labels are ignored.
    :param method: One of ``jackknife``, ``brr``, ``bootstrap``,
        ``successive-difference``, or ``fay`` (which requires ``fay_k``).
    :param fay_k: Fay's perturbation constant, required when
        ``method="fay"``.
    :param centering: ``full-sample`` (default) centers on ``statistic(series)``;
        ``replicate-mean`` centers on the mean of the replicate estimates.
        This choice does not change the method's scale factor.
    :returns: The estimated variance of the statistic.
    """
    from microdf.microseries import MicroSeries

    if centering not in ("full-sample", "replicate-mean"):
        raise ValueError(
            f"centering must be 'full-sample' or 'replicate-mean', got {centering!r}"
        )

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

    center = float(statistic(series)) if centering == "full-sample" else None
    values = series.array
    index = series.index

    estimates = []
    for column in range(n_replicates):
        replicate = MicroSeries(
            values.copy(),
            weights=weights[:, column].copy(),
            index=index.copy(),
            name=series.name,
            dtype=series.dtype,
        )
        estimates.append(float(statistic(replicate)))

    estimates = np.asarray(estimates)
    if not np.all(np.isfinite(estimates)) or (
        center is not None and not np.isfinite(center)
    ):
        # Retain the original infinity/NaN propagation for nonfinite callbacks.
        if centering == "replicate-mean":
            center = float(np.mean(estimates))
        return factor * float(np.sum(np.square(estimates - center)))

    with np.errstate(over="ignore"):
        if centering == "replicate-mean":
            # Keep the common offset out of the mean so small differences are
            # preserved even when the absolute mean is not representable.
            deviations = estimates - estimates[0]
            if not np.all(np.isfinite(deviations)):
                return float("inf")
            deviations -= np.mean(deviations)
        else:
            deviations = estimates - center

    scale = np.max(np.abs(deviations))
    if scale == 0:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")

    scaled_squares = np.sum(np.square(deviations / scale))
    # Restore the scale by its binary exponent after applying the factor.
    # Squaring scale first could overflow, or underflow before Fay's factor
    # brings a tiny squared deviation back into the representable range.
    significand, exponent = np.frexp(scale)
    with np.errstate(over="ignore"):
        return float(np.ldexp(significand**2 * factor * scaled_squares, 2 * exponent))


def replicate_standard_error(
    series,
    statistic: Callable,
    replicate_weights: np.ndarray | pd.DataFrame,
    method: str = "jackknife",
    fay_k: float | None = None,
    *,
    centering: str = "full-sample",
) -> float:
    """Standard error of ``statistic``, the square root of its variance.

    Takes the same arguments as :func:`replicate_variance`.
    """
    return float(
        np.sqrt(
            replicate_variance(
                series, statistic, replicate_weights, method, fay_k, centering=centering
            )
        )
    )
