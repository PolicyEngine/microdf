import logging
import warnings
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _weighted_centered_vector(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return scaled sqrt-weighted deviations and their power-of-two
    exponent."""
    # Center relative to a maximum-weight observation: shifting by a low-weight
    # extreme could erase differences among the influential observations.
    # A relative mean also preserves nearby values at a large common offset.
    reference = values[np.argmax(weights)]
    with np.errstate(over="ignore"):
        shifted = values - reference
    exponent = 0
    if np.isinf(shifted).any():
        # Opposite finite extremes can overflow their difference. Halving is
        # exact for those values; restore that factor in the final exponent.
        shifted = values / 2 - reference / 2
        exponent = 1
    magnitude = np.max(np.abs(shifted))
    if magnitude == 0:
        return shifted, 0
    _, shift = np.frexp(magnitude)
    shifted = np.ldexp(shifted, -shift)
    # Raise tiny mean weights by an exact common power of two so products
    # with the scaled deviations do not underflow. Never scale down: that
    # could discard small weights when frequencies span a wide range.
    _, mean_weight_exponent = np.frexp(np.max(weights))
    mean_weights = np.ldexp(weights, -min(int(mean_weight_exponent), 0))
    shifted -= np.average(shifted, weights=mean_weights)
    # Weight each vector before taking products, then scale again so squared
    # deviations never accumulate raw frequencies at the original value scale.
    shifted *= np.sqrt(weights)
    _, weight_shift = np.frexp(np.max(np.abs(shifted)))
    return np.ldexp(shifted, -weight_shift), exponent + int(shift) + int(weight_shift)


def _weighted_top_share(
    values: np.ndarray, weights: np.ndarray, top_x_pct: float
) -> float:
    """Share of the sum held by the top ``top_x_pct`` of weight.

    Sort by value ascending, cumulate the weight, pick the slice from the top
    that covers exactly ``top_x_pct`` of total weight, and distribute the tied-
    at-cutoff row proportionally so constant values return exactly
    ``top_x_pct`` rather than 1.0.
    """
    if top_x_pct <= 0:
        return 0.0
    if top_x_pct >= 1:
        return 1.0
    total_weight = weights.sum()
    total_sum = float((values * weights).sum())
    if total_weight == 0 or total_sum == 0:
        return np.nan
    # Ascending sort; the "top" cutoff is the final ``top_x_pct`` of
    # cumulative weight.
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    w = weights[order]
    # Cumulative weight from the bottom up.
    cum_w = np.cumsum(w)
    target_bottom_weight = total_weight * (1.0 - top_x_pct)
    # searchsorted(cum_w, target, side="right") gives the first index
    # whose cumulative weight exceeds the bottom cutoff.
    k = int(np.searchsorted(cum_w, target_bottom_weight, side="right"))
    # Rows strictly above the cutoff contribute all of their weight.
    if k >= len(v):
        return 0.0
    top_sum = float((v[k + 1 :] * w[k + 1 :]).sum())
    # Row k straddles the cutoff; include the fraction of its weight
    # that lies above the cutoff so ties don't double-count.
    partial_weight = cum_w[k] - target_bottom_weight
    top_sum += float(v[k] * partial_weight)
    return top_sum / total_sum


class MicroSeries(pd.Series):
    # Declare ``weights`` as pandas metadata. pandas includes
    # _metadata attributes in the pickle state, so weights now survive
    # pickling, to_pickle/read_pickle and copy.deepcopy instead of
    # vanishing and leaving an AttributeError on the next aggregation.
    # Keep pandas' own metadata, including the Series name.
    _metadata = pd.Series._metadata + ["weights"]

    def __init__(self, *args, weights: np.array = None, **kwargs):
        """A Series-inheriting class for weighted microdata.

        Weights can be provided at initialisation, or using set_weights.

        :param weights: Array of weights.
        :type weights: np.array
        """
        super().__init__(*args, **kwargs)
        self.set_weights(weights)

    def __finalize__(self, other, method=None, **kwargs) -> "MicroSeries":
        """Retain copied weights when pandas finalizes a renamed result."""
        copied_weights = getattr(self, "weights", None) if method == "rename" else None
        super().__finalize__(other, method=method, **kwargs)
        if copied_weights is not None:
            # rename already called copy(); metadata propagation must not
            # replace those weights with the source's mutable Series.
            self.weights = copied_weights
        return self

    @property
    def _values(self):
        """Internal access to underlying numpy array without warning."""
        return super().values

    @property
    def values(self):
        """Access underlying numpy array.

        .. warning::
            Returns a plain numpy array without weights. Operations
            like ``.mean()`` on the result will be unweighted. Use
            MicroSeries methods directly for weighted calculations
            (e.g., ``ms.mean()`` instead of ``ms.values.mean()``).
        """
        warnings.warn(
            "Accessing .values on a MicroSeries returns a plain numpy "
            "array without weights. Operations like .mean() on the "
            "result will be unweighted. Use MicroSeries methods "
            "directly for weighted calculations (e.g., ms.mean() "
            "instead of ms.values.mean()).",
            UserWarning,
            stacklevel=2,
        )
        return super().values

    def to_numpy(self, *args, **kwargs):
        """Convert to numpy array.

        .. warning::
            Returns a plain numpy array without weights. Operations
            like ``.mean()`` on the result will be unweighted. Use
            MicroSeries methods directly for weighted calculations.
        """
        warnings.warn(
            "Calling .to_numpy() on a MicroSeries returns a plain "
            "numpy array without weights. Operations like .mean() on "
            "the result will be unweighted. Use MicroSeries methods "
            "directly for weighted calculations.",
            UserWarning,
            stacklevel=2,
        )
        return super().to_numpy(*args, **kwargs)

    def scalar_function(fn: Callable) -> Callable:
        """Decorator marking ``fn`` as returning a scalar (float)."""
        fn._rtype = float
        return fn

    def vector_function(fn: Callable) -> Callable:
        """Decorator marking ``fn`` as returning a pandas Series."""
        fn._rtype = pd.Series
        return fn

    def set_weights(
        self, weights: np.array, preserve_old: Optional[bool] = False
    ) -> None:
        """Sets the weight values.

        :param weights: Array of weights.
        :param preserve_old: If True, keeps the old weights as a column when
            new weights are provided.
        :type weights: np.array.
        """
        if weights is None:
            if len(self) > 0:
                self.weights = pd.Series(
                    np.ones_like(self._values),
                    index=self.index,
                    dtype=float,
                )
        else:
            if len(weights) != len(self):
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match "
                    f"length of DataFrame ({len(self)})."
                )

            if preserve_old and self.weights is not None:
                self["old_weights"] = self.weights

            # Align weights to self.index so element-wise operations such
            # as self.multiply(self.weights) (used by .sum(), .weight())
            # don't silently produce all-NaN when the caller uses a
            # non-default index. If a pandas Series is passed in, strip
            # its index first so we position-align rather than label-align.
            if isinstance(weights, pd.Series):
                weights = weights.values
            self.weights = pd.Series(np.asarray(weights), index=self.index, dtype=float)

    def nullify_weights(self) -> None:
        """Set all weights to 1, effectively making the Series unweighted.

        This is useful for comparing weighted and unweighted statistics or when
        you want to temporarily ignore weights.
        """
        # Index the ones against self.index: weighted ops are label-aligned
        # (self.multiply(self.weights) in .sum()/.weight()), so a default
        # RangeIndex here silently produces all-NaN and collapses every
        # aggregation to 0 whenever the caller uses a non-default index.
        self.weights = pd.Series(np.ones(len(self)), index=self.index, dtype=float)

    @vector_function
    def weight(self) -> pd.Series:
        """Calculates the weighted value of the MicroSeries.

        :returns: A Series multiplying the MicroSeries by its weight.
        :rtype: pd.Series
        """
        return pd.Series(self, copy=False).multiply(self.weights)

    @scalar_function
    def sum(
        self,
        axis: Optional[Union[int, str]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ) -> float:
        """Calculates the weighted sum of the MicroSeries.

        axis may be 0, 'index' or None, as for pandas Series.sum. skipna,
        numeric_only and min_count are applied to the weighted values;
        min_count counts valid observations, not the sum of their weights.

        :returns: The weighted sum.
        :rtype: float
        """
        # Keep the intermediate unweighted so subclass constructors cannot
        # apply observation weights a second time during the final reduction.
        values = pd.Series(self)
        if not self.empty:
            values = values.multiply(self.weights)
        return values.sum(
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    @scalar_function
    def count(self, skipna: bool = True) -> float:
        """Calculates the weighted count of the MicroSeries.

        By default skips NaN values (matching pandas ``Series.count``).

        :param skipna: Exclude NaN values (default True). If False, the
            weighted count of every row is returned.
        :type skipna: bool
        :returns: The weighted count.
        :rtype: float
        """
        weights = np.asarray(self.weights.values, dtype=float)
        if not skipna:
            return float(weights.sum())
        mask = ~pd.isna(self._values)
        return float(weights[mask].sum())

    @scalar_function
    def mean(self, skipna: bool = True) -> float:
        """Calculates the weighted mean of the MicroSeries.

        :param skipna: Exclude NA/null values. If True (default), NaN values
            are excluded. If False, returns NaN if any value is NaN.
        :type skipna: bool
        :returns: The weighted mean.
        :rtype: float
        """
        values = self._values
        weights = self.weights

        if skipna:
            # Create mask for non-NaN values
            mask = ~pd.isna(values)
            if not mask.any():
                # All values are NaN
                return np.nan
            values = values[mask]
            weights = weights[mask]

        # If skipna=False and there are any NaN values, return NaN
        if not skipna and pd.isna(values).any():
            return np.nan

        return np.average(values, weights=weights)

    def _weighted_variance(self, ddof: int = 1, skipna: bool = True) -> float:
        """Frequency-weighted variance.

        Uses ``sum(w * (x - wmean)**2) / (sum(w) - ddof)``. With
        ``ddof=0`` this is the population variance; with ``ddof=1`` it
        is Bessel-corrected assuming the weights are frequency counts —
        matching ``np.var(..., ddof=ddof)`` on a replicated sample.
        """
        values = np.asarray(self._values, dtype=float)
        weights = np.asarray(self.weights.values, dtype=float)
        if skipna:
            mask = ~np.isnan(values)
            values = values[mask]
            weights = weights[mask]
        elif np.isnan(values).any():
            return np.nan
        total_w = weights.sum()
        if total_w == 0 or total_w - ddof <= 0:
            return np.nan
        mean = np.average(values, weights=weights)
        return float((weights * (values - mean) ** 2).sum() / (total_w - ddof))

    @scalar_function
    def var(self, ddof: int = 1, skipna: bool = True) -> float:
        """Calculates the weighted variance of the MicroSeries.

        Treats weights as frequency counts (``sum(w) - ddof`` in the
        denominator) so that with integer weights the result matches
        ``np.var`` on the replicated sample.

        :param ddof: Delta degrees of freedom (default 1).
        :param skipna: Exclude NaN values (default True).
        :returns: The weighted variance.
        :rtype: float
        """
        return self._weighted_variance(ddof=ddof, skipna=skipna)

    @scalar_function
    def std(self, ddof: int = 1, skipna: bool = True) -> float:
        """Calculates the weighted standard deviation of the MicroSeries.

        :param ddof: Delta degrees of freedom (default 1).
        :param skipna: Exclude NaN values (default True).
        :returns: The weighted standard deviation.
        :rtype: float
        """
        v = self._weighted_variance(ddof=ddof, skipna=skipna)
        return float(np.sqrt(v)) if np.isfinite(v) else v

    def _weighted_pair(
        self,
        other: pd.Series,
        min_periods: Optional[int],
        ddof: int,
        skipna: bool,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Align usable paired observations with their left weights."""
        if not isinstance(other, pd.Series):
            raise TypeError("other must be a pandas Series or MicroSeries")
        if not isinstance(ddof, (int, np.integer)):
            raise TypeError("ddof must be an integer")
        if min_periods is None:
            min_periods = 1
        if not isinstance(min_periods, (int, np.integer)) or min_periods < 0:
            raise ValueError("min_periods must be a nonnegative integer")
        if len(self) == 0 or len(other) == 0:
            return None

        # Align left row positions so values and weights undergo exactly the
        # same join, including pandas' expansion of duplicate index labels.
        positions = pd.Series(np.arange(len(self)), index=self.index)
        positions, right = positions.align(pd.Series(other), join="inner")
        positions = positions.to_numpy(dtype=int)
        x = (
            pd.Series(self._values)
            .iloc[positions]
            .to_numpy(dtype=float, na_value=np.nan)
        )
        y = right.to_numpy(dtype=float, na_value=np.nan)
        weights = np.asarray(self.weights, dtype=float)[positions]
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("frequency weights must be finite and nonnegative")

        # Zero frequency means the row is absent, including for skipna=False.
        positive = weights > 0
        x, y, weights = x[positive], y[positive], weights[positive]
        missing = np.isnan(x) | np.isnan(y)
        if not skipna and missing.any():
            return None
        x, y, weights = x[~missing], y[~missing], weights[~missing]
        total_weight = weights.sum()
        if not np.isfinite(total_weight):
            raise ValueError("the sum of frequency weights must be finite")
        if len(x) < min_periods or total_weight == 0 or total_weight <= ddof:
            return None
        return (
            x,
            y,
            weights,
            float(total_weight - ddof),
        )

    def cov(
        self,
        other: pd.Series,
        min_periods: Optional[int] = None,
        ddof: int = 1,
        *,
        skipna: bool = True,
    ) -> float:
        """Calculate frequency-weighted covariance with another Series.

        Observations align by index as in pandas, including its duplicate-
        label join behavior. Only this Series' weights are used; weights on
        another MicroSeries are ignored. Each aligned left weight must be
        finite and nonnegative. Zero-weight rows are omitted.

        Uses ``sum(w * (x - xmean) * (y - ymean)) / (sum(w) - ddof)``.
        Integer weights therefore match covariance on the replicated sample.
        Missing values are removed pairwise before computing both means.

        :param other: A pandas Series or MicroSeries to align by index.
        :param min_periods: Minimum usable aligned row pairs, not the sum of
            frequency weights. Defaults to 1.
        :param ddof: Degrees of freedom subtracted from the weight total.
        :param skipna: Drop pairs with a missing value. If False, any missing
            value in a positive-weight aligned pair produces NaN.
        :returns: Weighted covariance, or NaN for an empty or insufficient
            sample (including a weight total no greater than ddof).
        """
        pair = self._weighted_pair(other, min_periods, ddof, skipna)
        if pair is None:
            return np.nan
        x, y, weights, denominator = pair
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return np.nan
        x, x_exponent = _weighted_centered_vector(x, weights)
        y, y_exponent = _weighted_centered_vector(y, weights)
        # Combine exponents only after dividing out sum(weights) - ddof.
        # Neither the original squared scale nor raw weighted sum need fit.
        denominator, denominator_exponent = np.frexp(denominator)
        return float(
            np.ldexp(
                np.sum(x * y) / denominator,
                x_exponent + y_exponent - int(denominator_exponent),
            )
        )

    def corr(
        self,
        other: pd.Series,
        method: str = "pearson",
        min_periods: Optional[int] = None,
        *,
        ddof: int = 1,
        skipna: bool = True,
    ) -> float:
        """Calculate frequency-weighted Pearson correlation.

        Uses the same aligned pairs and left Series weights for covariance and
        both variances. Weights on another MicroSeries are ignored. Weights
        must be finite and nonnegative; zero-weight rows are omitted. Other
        correlation methods, including callables, are unsupported.

        :param other: A pandas Series or MicroSeries to align by index.
        :param method: Only "pearson" is supported.
        :param min_periods: Minimum usable aligned row pairs, not frequency
            weight total. Defaults to 1.
        :param ddof: Degrees of freedom for all three moments. It cancels from
            the correlation but the weight total must exceed it.
        :param skipna: Drop pairs with a missing value. If False, any missing
            value in a positive-weight aligned pair produces NaN.
        :returns: Weighted correlation, or NaN for an empty, insufficient, or
            constant sample.
        """
        if method != "pearson":
            raise ValueError("weighted correlation only supports method='pearson'")
        pair = self._weighted_pair(other, min_periods, ddof, skipna)
        if pair is None:
            return np.nan
        x, y, weights, _ = pair
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return np.nan
        # A weighted mean can round away from identical decimal inputs.
        # Check the retained observations exactly before subtracting it.
        if (x == x[0]).all() or (y == y[0]).all():
            return np.nan
        x, _ = _weighted_centered_vector(x, weights)
        y, _ = _weighted_centered_vector(y, weights)
        x_ss = np.sum(x * x)
        y_ss = np.sum(y * y)
        if x_ss == 0 or y_ss == 0:
            return np.nan
        result = np.sum(x * y) / (np.sqrt(x_ss) * np.sqrt(y_ss))
        return float(np.clip(result, -1.0, 1.0))

    def quantile(self, q: np.array, skipna: bool = True) -> pd.Series:
        """Calculates weighted quantiles of the MicroSeries.

        Uses the inverse CDF method: the q-th quantile is the smallest
        value where the cumulative weight proportion >= q. This matches
        the default behavior of R's survey::svyquantile.

        :param q: Quantile(s) to calculate, must be in [0, 1].
        :type q: float or np.array
        :param skipna: Exclude NaN values (default True). NaN sorts to the
            end of the array, so leaving NaN rows in would let their weight
            inflate the cumulative distribution and push the cutoff upward.
            If False, NaN is returned whenever any value is NaN.
        :type skipna: bool

        :return: Weighted quantile value(s).
        :rtype: float or pd.Series
        """
        values = np.array(self._values)
        quantiles = np.atleast_1d(q)
        sample_weight = np.array(self.weights)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), (
            "quantiles should be in [0, 1]"
        )
        na_mask = pd.isna(values)
        if not skipna and na_mask.any():
            return (
                np.nan
                if np.array(q).shape == ()
                else pd.Series(np.full(len(quantiles), np.nan), index=quantiles)
            )
        # Drop zero-weight rows before sorting. Without this, q=0 (and
        # internal plateaus of zero weight) picked a value with 0 weight
        # that should have been skipped by the inverse CDF. E.g.
        # MicroSeries([10, 20, 30], weights=[0, 1, 1]).quantile(0)
        # returned 10 instead of 20.
        # Drop NaN rows for the same reason: NaN sorts last, so its weight
        # would inflate the cumulative distribution and push the cutoff up
        # (median of [1, nan, 3] returned 3.0 instead of 1.0).
        nonzero = (sample_weight > 0) & ~na_mask
        if not nonzero.any():
            return (
                np.nan
                if np.array(q).shape == ()
                else pd.Series(np.full(len(quantiles), np.nan), index=quantiles)
            )
        values = values[nonzero]
        sample_weight = sample_weight[nonzero]
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        cumsum = np.cumsum(sample_weight)
        cumsum_normalized = cumsum / cumsum[-1]
        result = np.array(
            [
                values[min(np.searchsorted(cumsum_normalized, qi), len(values) - 1)]
                for qi in quantiles
            ]
        )
        if np.array(q).shape == ():
            return result[0]
        return pd.Series(result, index=quantiles)

    @scalar_function
    def median(self, skipna: bool = True) -> float:
        """Calculates the weighted median of the MicroSeries.

        :param skipna: Exclude NaN values (default True).
        :type skipna: bool
        :returns: The weighted median of a DataFrame's column.
        :rtype: float
        """
        return self.quantile(0.5, skipna=skipna)

    @scalar_function
    def gini(self, negatives: Optional[str] = None) -> float:
        """Calculates Gini index.

        :param negatives: An optional string indicating how to treat
            negative values of x:
            'zero' replaces negative values with zeroes.
            'shift' subtracts the minimum value from all values of x,
            when this minimum is negative. That is, it adds the absolute
            minimum value.
            Defaults to None, which leaves negative values as they are.
        :type negatives: str
        :returns: Gini index.
        :rtype: float
        """
        x = np.array(self).astype("float")
        w = np.asarray(self.weights.values, dtype=float)
        if negatives == "zero":
            x = np.where(x < 0, 0.0, x)
        elif negatives == "shift":
            if len(x) > 0 and np.amin(x) < 0:
                x = x - np.amin(x)
        elif negatives is not None:
            raise ValueError(
                f"Unknown negatives option {negatives!r}; expected "
                "'zero', 'shift', or None."
            )

        if len(x) == 0:
            return np.nan
        if np.any(x < 0):
            # The Lorenz-based formula assumes non-negative values; with
            # negatives it can return values outside [0, 1].
            warnings.warn(
                "gini() called on data containing negative values; the "
                "result is not guaranteed to lie in [0, 1]. Pass "
                "negatives='zero' or negatives='shift' to handle them.",
                UserWarning,
                stacklevel=2,
            )

        # Short-circuit degenerate cases so we don't divide by zero.
        total = float((x * w).sum())
        if total == 0:
            return 0.0

        sorter = np.argsort(x, kind="mergesort")
        sorted_x = x[sorter]
        sorted_w = w[sorter]
        cumw = np.cumsum(sorted_w)
        cumxw = np.cumsum(sorted_x * sorted_w)
        # Trapezoidal approximation of the area under the Lorenz curve.
        return float(
            np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:])
            / (cumxw[-1] * cumw[-1])
        )

    @scalar_function
    def top_x_pct_share(self, top_x_pct: float) -> float:
        """Calculates top x% share.

        Uses a cumulative-weight sort so that rows tied at the cutoff
        contribute proportionally rather than all-or-nothing. With
        constant values this correctly returns ``top_x_pct`` itself.

        :param top_x_pct: Decimal between 0 and 1 of the top %, e.g. 0.1,
            0.001.
        :type top_x_pct: float
        :returns: The weighted share held by the top x%.
        :rtype: float
        """
        return _weighted_top_share(
            np.asarray(self._values, dtype=float),
            np.asarray(self.weights.values, dtype=float),
            float(top_x_pct),
        )

    @scalar_function
    def bottom_x_pct_share(self, bottom_x_pct: float) -> float:
        """Calculates bottom x% share.

        :param bottom_x_pct: Decimal between 0 and 1 of the bottom %, e.g. 0.1,
            0.001.
        :type bottom_x_pct: float
        :returns: The weighted share held by the bottom x%.
        :rtype: float
        """
        return 1 - self.top_x_pct_share(1 - bottom_x_pct)

    @scalar_function
    def bottom_50_pct_share(self) -> float:
        """Calculates bottom 50% share.

        :returns: The weighted share held by the bottom 50%.
        :rtype: float
        """
        return self.bottom_x_pct_share(0.5)

    @scalar_function
    def top_50_pct_share(self) -> float:
        """Calculates top 50% share.

        :returns: The weighted share held by the top 50%.
        :rtype: float
        """
        return self.top_x_pct_share(0.5)

    @scalar_function
    def top_10_pct_share(self) -> float:
        """Calculates top 10% share.

        :returns: The weighted share held by the top 10%.
        :rtype: float
        """
        return self.top_x_pct_share(0.1)

    @scalar_function
    def top_1_pct_share(self) -> float:
        """Calculates top 1% share.

        :returns: The weighted share held by the top 50%.
        :rtype: float
        """
        return self.top_x_pct_share(0.01)

    @scalar_function
    def top_0_1_pct_share(self) -> float:
        """Calculates top 0.1% share.

        :returns: The weighted share held by the top 0.1%.
        :rtype: float
        """
        return self.top_x_pct_share(0.001)

    @scalar_function
    def t10_b50(self) -> float:
        """Calculates ratio between the top 10% and bottom 50% shares.

        :returns: The weighted share held by the top 10% divided by the
            weighted share held by the bottom 50%.
        """
        t10 = self.top_10_pct_share()
        b50 = self.bottom_50_pct_share()
        return t10 / b50

    @vector_function
    def cumsum(self) -> pd.Series:
        logger.warning(
            "cumsum() returns cumulative sums of weighted values as a regular "
            "pandas Series. The original weights have already been applied "
            "and cannot be reused with the cumulative results."
        )
        return pd.Series(self * self.weights).cumsum()

    @vector_function
    def rank(self, pct: Optional[bool] = False) -> pd.Series:
        """Weighted rank of each element.

        Each element's rank is the cumulative weight of all values that are
        less than or equal to it. Tied values therefore share the same rank, so
        downstream bucketing (``decile_rank``, ``quintile_rank``, etc.) lands
        tied rows in the same bucket.

        :param pct: If True, divide ranks by the total weight so they lie in
            ``(0, 1]``.
        :type pct: bool
        :returns: MicroSeries of ranks aligned to ``self``.
        :rtype: MicroSeries
        """
        weights_sum = np.asarray(self.weights.values, dtype=float).sum()
        if weights_sum == 0:
            raise ZeroDivisionError(
                "Cannot calculate rank with zero total weight. "
                "All weights in the MicroSeries are zero, which would "
                "result in division by zero."
            )

        values = np.asarray(self._values)
        weights = np.asarray(self.weights.values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = weights[order]
        cum_w = np.cumsum(sorted_weights)
        # Max rank semantics: every tied group gets the cumulative
        # weight at the *end* of the group, so ties share one rank.
        # searchsorted(side='right') on the sorted values finds the
        # index just past each tied block in sort order.
        group_end = np.searchsorted(sorted_values, sorted_values, side="right") - 1
        sorted_ranks = cum_w[group_end]
        # Invert the sort to put ranks back into the caller's order.
        inverse_order = np.argsort(order, kind="mergesort")
        ranks = sorted_ranks[inverse_order]
        if pct:
            ranks = ranks / weights_sum
            ranks = np.where(ranks > 1.0, 1.0, ranks)
        return MicroSeries(ranks, index=self.index, weights=self.weights)

    @vector_function
    def decile_rank(self, negatives_in_zero: Optional[bool] = False):
        """Calculate decile ranks (1-10) with optional zero decile for
        negatives.

        :param negatives_in_zero: If True, negative values are assigned to
            decile 0. If False (default), all values are ranked 1-10.
        :type negatives_in_zero: bool
        :returns: MicroSeries with decile ranks
        :rtype: MicroSeries
        """
        if negatives_in_zero:
            negative_mask = self < 0
            if negative_mask.any():
                non_negative_values = self[~negative_mask]
                if len(non_negative_values) > 0:
                    non_neg_ranks = non_negative_values.rank(pct=True)
                    deciles = np.minimum(np.ceil(non_neg_ranks * 10), 10)
                else:
                    deciles = np.array([])

                result = np.zeros(len(self))
                result[negative_mask] = 0
                if len(deciles) > 0:
                    result[~negative_mask] = deciles

                return MicroSeries(result, weights=self.weights)

        # Default behavior: rank all values 1-10
        return MicroSeries(
            np.minimum(np.ceil(self.rank(pct=True) * 10), 10),
            weights=self.weights,
        )

    @vector_function
    def quintile_rank(self) -> "MicroSeries":
        return MicroSeries(
            np.minimum(np.ceil(self.rank(pct=True) * 5), 5),
            weights=self.weights,
        )

    @vector_function
    def quartile_rank(self) -> "MicroSeries":
        return MicroSeries(
            np.minimum(np.ceil(self.rank(pct=True) * 4), 4),
            weights=self.weights,
        )

    @vector_function
    def percentile_rank(self) -> "MicroSeries":
        return MicroSeries(
            np.minimum(np.ceil(self.rank(pct=True) * 100), 100),
            weights=self.weights,
        )

    def groupby(self, *args, **kwargs) -> "MicroSeriesGroupBy":
        gb = super().groupby(*args, **kwargs)
        gb.__class__ = MicroSeriesGroupBy
        gb._init()
        gb.weights = pd.Series(self.weights).groupby(*args, **kwargs)
        return gb

    def copy(self, deep: Optional[bool] = True):
        res = super().copy(deep)
        res = MicroSeries(res, weights=self.weights.copy(deep))
        return res

    def clip(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        axis: Optional[int] = None,
        inplace: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> "MicroSeries":
        res = super().clip(
            lower=lower,
            upper=upper,
            axis=axis,
            inplace=inplace,
            *args,
            **kwargs,
        )
        if not inplace:
            return MicroSeries(res, weights=self.weights)
        return self

    def round(self, decimals: Optional[int] = 0, *args, **kwargs) -> "MicroSeries":
        res = super().round(decimals=decimals, *args, **kwargs)
        return MicroSeries(res, weights=self.weights)

    def equals(self, other: "MicroSeries") -> bool:
        equal_values = super().equals(other)
        equal_weights = self.weights.equals(other.weights)
        return equal_values and equal_weights

    def __getitem__(
        self, key: Union[str, int, slice, List, np.ndarray]
    ) -> Union["MicroSeries", pd.Series]:
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            weights = self.weights.__getitem__(key)
            return MicroSeries(result, weights=weights)
        return result

    def __getattr__(self, name: str) -> "MicroSeries":
        return MicroSeries(super().__getattr__(name), weights=self.weights)

    # operators

    def __add__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__add__(other), weights=self.weights)

    def __sub__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__sub__(other), weights=self.weights)

    def __mul__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__mul__(other), weights=self.weights)

    def __floordiv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__floordiv__(other), weights=self.weights)

    def __truediv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__truediv__(other), weights=self.weights)

    def __mod__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__mod__(other), weights=self.weights)

    def __pow__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__pow__(other), weights=self.weights)

    def __xor__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__xor__(other), weights=self.weights)

    def __and__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__and__(other), weights=self.weights)

    def __or__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__or__(other), weights=self.weights)

    def __invert__(self) -> "MicroSeries":
        return MicroSeries(super().__invert__(), weights=self.weights)

    def __radd__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__radd__(other), weights=self.weights)

    def __rsub__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rsub__(other), weights=self.weights)

    def __rmul__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rmul__(other), weights=self.weights)

    def __rfloordiv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rfloordiv__(other), weights=self.weights)

    def __rtruediv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rtruediv__(other), weights=self.weights)

    def __rmod__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rmod__(other), weights=self.weights)

    def __rpow__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rpow__(other), weights=self.weights)

    def __rand__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rand__(other), weights=self.weights)

    def __ror__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__ror__(other), weights=self.weights)

    def __rxor__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__rxor__(other), weights=self.weights)

    def sqrt(self) -> "MicroSeries":
        sqrt_values = np.sqrt(self._values)
        return MicroSeries(sqrt_values, index=self.index, weights=self.weights)

    # comparators

    def __lt__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__lt__(other), weights=self.weights)

    def __le__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__le__(other), weights=self.weights)

    def __eq__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__eq__(other), weights=self.weights)

    def __ne__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__ne__(other), weights=self.weights)

    def __ge__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__ge__(other), weights=self.weights)

    def __gt__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__gt__(other), weights=self.weights)

    # assignment operators

    def __iadd__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__iadd__(other), weights=self.weights)

    def __isub__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__isub__(other), weights=self.weights)

    def __imul__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__imul__(other), weights=self.weights)

    def __ifloordiv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__ifloordiv__(other), weights=self.weights)

    def __idiv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__idiv__(other), weights=self.weights)

    def __itruediv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__itruediv__(other), weights=self.weights)

    def __imod__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__imod__(other), weights=self.weights)

    def __ipow__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__ipow__(other), weights=self.weights)

    # other

    def __neg__(self) -> "MicroSeries":
        return MicroSeries(super().__neg__(), weights=self.weights)

    def __pos__(self) -> "MicroSeries":
        return MicroSeries(super().__pos__(), weights=self.weights)

    def astype(
        self,
        dtype,
        copy: Optional[bool] = True,
        errors: Optional[str] = "raise",
    ) -> "MicroSeries":
        """Convert MicroSeries to specified data type while preserving weights.

        :param dtype: Data type to convert to. Can be numpy dtype or Python
            type.
        :param copy: Whether to make a copy of the data (default True).
        :param errors: How to handle conversion errors (default "raise").
        :return: New MicroSeries with converted data type and preserved
            weights.
        """
        converted_series = super().astype(dtype, copy=copy, errors=errors)
        return MicroSeries(
            converted_series,
            weights=self.weights.copy() if copy else self.weights,
        )

    def __repr__(self) -> str:
        return pd.DataFrame(
            dict(value=self._values, weight=self.weights.values)
        ).__repr__()


MicroSeries.SCALAR_FUNCTIONS = [
    fn
    for fn in dir(MicroSeries)
    if "_rtype" in dir(getattr(MicroSeries, fn))
    and getattr(getattr(MicroSeries, fn), "_rtype") == float
]
MicroSeries.VECTOR_FUNCTIONS = [
    fn
    for fn in dir(MicroSeries)
    if "_rtype" in dir(getattr(MicroSeries, fn))
    and getattr(getattr(MicroSeries, fn), "_rtype") == pd.Series
]
MicroSeries.AGNOSTIC_FUNCTIONS = ["quantile"]
MicroSeries.FUNCTIONS = sum(
    [
        MicroSeries.SCALAR_FUNCTIONS,
        MicroSeries.VECTOR_FUNCTIONS,
        MicroSeries.AGNOSTIC_FUNCTIONS,
    ],
    [],
)


class MicroSeriesGroupBy(pd.core.groupby.generic.SeriesGroupBy):
    def _init(self):
        def _weighted_agg(name) -> Callable:
            def via_micro_series(row, *args, **kwargs):
                return getattr(MicroSeries(row.a, weights=row.w), name)(*args, **kwargs)

            fn = getattr(MicroSeries, name)

            @wraps(fn)
            def _weighted_agg_fn(*args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
                arrays = self.apply(np.array)
                weights = self.weights.apply(np.array)
                df = pd.DataFrame(dict(a=arrays, w=weights))
                is_array = len(args) > 0 and hasattr(args[0], "__len__")
                if (
                    name in MicroSeries.SCALAR_FUNCTIONS
                    or name in MicroSeries.AGNOSTIC_FUNCTIONS
                    and not is_array
                ):
                    result = df.agg(
                        lambda row: via_micro_series(row, *args, **kwargs),
                        axis=1,
                    )
                elif (
                    name in MicroSeries.VECTOR_FUNCTIONS
                    or name in MicroSeries.AGNOSTIC_FUNCTIONS
                    and is_array
                ):
                    if name in MicroSeries.AGNOSTIC_FUNCTIONS and not df.empty:
                        # Concatenate values without keys: concat rejects missing
                        # MultiIndex keys even when groupby(dropna=False) retains
                        # them. Reuse the grouping levels and codes so missing
                        # labels keep the same representation as scalar results.
                        results = [
                            via_micro_series(row, *args, **kwargs)
                            for _, row in df.iterrows()
                        ]
                        result = pd.concat(results)
                        group_index = (
                            df.index
                            if isinstance(df.index, pd.MultiIndex)
                            else pd.MultiIndex.from_arrays([df.index])
                        )
                        quantile_codes, quantile_levels = result.index.factorize(
                            sort=False
                        )
                        result.index = pd.MultiIndex(
                            levels=[*group_index.levels, quantile_levels],
                            codes=[
                                codes.repeat(len(results[0]))
                                for codes in group_index.codes
                            ]
                            + [quantile_codes],
                            names=[*df.index.names, result.index.name],
                            # Existing group codes are valid; checking would
                            # rewrite their retained missing labels to -1.
                            verify_integrity=False,
                        )
                        return result
                    result = df.apply(
                        lambda row: via_micro_series(row, *args, **kwargs),
                        axis=1,
                    )
                    return result.stack()
                return result

            return _weighted_agg_fn

        for fn_name in MicroSeries.FUNCTIONS:
            setattr(self, fn_name, _weighted_agg(fn_name))
