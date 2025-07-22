import logging
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MicroSeries(pd.Series):
    def __init__(self, *args, weights: np.array = None, **kwargs):
        """A Series-inheriting class for weighted microdata. Weights can be
        provided at initialisation, or using set_weights.

        :param weights: Array of weights.
        :type weights: np.array
        """
        super().__init__(*args, **kwargs)
        self.set_weights(weights)

    def weighted_function(fn: Callable) -> Callable:
        @wraps(fn)
        def safe_fn(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ZeroDivisionError:
                return np.NaN

        return safe_fn

    @weighted_function
    def scalar_function(fn: Callable) -> Callable:
        fn._rtype = float
        return fn

    @weighted_function
    def vector_function(fn: Callable) -> Callable:
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
                    np.ones_like(self.values), dtype=float
                )
        else:
            if len(weights) != len(self):
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match "
                    f"length of DataFrame ({len(self)})."
                )

            if preserve_old and self.weights is not None:
                self["old_weights"] = self.weights

            self.weights = pd.Series(weights, dtype=float)

    @vector_function
    def weight(self) -> pd.Series:
        """Calculates the weighted value of the MicroSeries.

        :returns: A Series multiplying the MicroSeries by its weight.
        :rtype: pd.Series
        """
        return self.multiply(self.weights)

    @scalar_function
    def sum(self) -> float:
        """Calculates the weighted sum of the MicroSeries.

        :returns: The weighted sum.
        :rtype: float
        """
        return self.multiply(self.weights).sum()

    @scalar_function
    def count(self) -> float:
        """Calculates the weighted count of the MicroSeries.

        :returns: The weighted count.
        """
        return self.weights.sum()

    @scalar_function
    def mean(self) -> float:
        """Calculates the weighted mean of the MicroSeries.

        :returns: The weighted mean.
        :rtype: float
        """
        return np.average(self.values, weights=self.weights)

    def quantile(self, q: np.array) -> pd.Series:
        """Calculates weighted quantiles of the MicroSeries.

        Doesn't exactly match unweighted quantiles of stacked values.
        See stackoverflow.com/q/21844024#comment102342137_29677616.

        :param q: Array of quantiles to calculate.
        :type q: np.array

        :return: Array of weighted quantiles.
        :rtype: pd.Series
        """
        values = np.array(self.values)
        quantiles = np.array(q)
        sample_weight = np.array(self.weights)
        assert np.all(quantiles >= 0) and np.all(
            quantiles <= 1
        ), "quantiles should be in [0, 1]"
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        result = np.interp(quantiles, weighted_quantiles, values)
        if quantiles.shape == ():
            return result
        return pd.Series(result, index=quantiles)

    @scalar_function
    def median(self) -> float:
        """Calculates the weighted median of the MicroSeries.

        :returns: The weighted median of a DataFrame's column.
        :rtype: float
        """
        return self.quantile(0.5)

    @scalar_function
    def gini(self, negatives: Optional[str] = None) -> float:
        """Calculates Gini index.

        :param negatives: An optional string indicating how to treat negative
            values of x:
            'zero' replaces negative values with zeroes.
            'shift' subtracts the minimum value from all values of x,
            when this minimum is negative. That is, it adds the absolute
            minimum value.
            Defaults to None, which leaves negative values as they are.
        :type q: str
        :returns: Gini index.
        :rtype: float
        """
        x = np.array(self).astype("float")
        if negatives == "zero":
            x[x < 0] = 0
        if negatives == "shift" and np.amin(x) < 0:
            x -= np.amin(x)
        if (self.weights != np.ones(len(self))).any():  # Varying weights.
            sorted_indices = np.argsort(self)
            sorted_x = np.array(self[sorted_indices])
            sorted_w = np.array(self.weights[sorted_indices])
            cumw = np.cumsum(sorted_w)
            cumxw = np.cumsum(sorted_x * sorted_w)
            return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
                cumxw[-1] * cumw[-1]
            )
        else:
            sorted_x = np.sort(self)
            n = len(x)
            cumxw = np.cumsum(sorted_x)
            # The above formula, with all weights equal to 1 simplifies to:
            return (n + 1 - 2 * np.sum(cumxw) / cumxw[-1]) / n

    @scalar_function
    def top_x_pct_share(self, top_x_pct: float) -> float:
        """Calculates top x% share.

        :param top_x_pct: Decimal between 0 and 1 of the top %, e.g. 0.1,
            0.001.
        :type top_x_pct: float
        :returns: The weighted share held by the top x%.
        :rtype: float
        """
        threshold = self.quantile(1 - top_x_pct)
        top_x_pct_sum = self[self >= threshold].sum()
        total_sum = self.sum()
        return top_x_pct_sum / total_sum

    @scalar_function
    def bottom_x_pct_share(self, bottom_x_pct: float) -> float:
        """Calculates bottom x% share.

        :param bottom_x_pct: Decimal between 0 and 1 of the top %, e.g. 0.1,
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
        weights_sum = self.weights.values.sum()
        if weights_sum == 0:
            raise ZeroDivisionError(
                "Cannot calculate rank with zero total weight. "
                "All weights in the MicroSeries are zero, which would result "
                "in division by zero."
            )

        order = np.argsort(self.values)
        inverse_order = np.argsort(order)
        ranks = np.array(self.weights.values)[order].cumsum()[inverse_order]
        if pct:
            ranks /= weights_sum
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

    def round(
        self, decimals: Optional[int] = 0, *args, **kwargs
    ) -> "MicroSeries":
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

    def __floordiv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
        return MicroSeries(super().__floordiv__(other), weights=self.weights)

    def __truediv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
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

    def __rfloordiv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
        return MicroSeries(super().__rfloordiv__(other), weights=self.weights)

    def __rtruediv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
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
        sqrt_values = np.sqrt(self.values)
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

    def __ifloordiv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
        return MicroSeries(super().__ifloordiv__(other), weights=self.weights)

    def __idiv__(self, other: Union[int, float, pd.Series]) -> "MicroSeries":
        return MicroSeries(super().__idiv__(other), weights=self.weights)

    def __itruediv__(
        self, other: Union[int, float, pd.Series]
    ) -> "MicroSeries":
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
            dict(value=self.values, weight=self.weights.values)
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
                return getattr(MicroSeries(row.a, weights=row.w), name)(
                    *args, **kwargs
                )

            fn = getattr(MicroSeries, name)

            @wraps(fn)
            def _weighted_agg_fn(
                *args, **kwargs
            ) -> Union[pd.Series, pd.DataFrame]:
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
                    result = df.apply(
                        lambda row: via_micro_series(row, *args, **kwargs),
                        axis=1,
                    )
                    return result.stack()
                return result

            return _weighted_agg_fn

        for fn_name in MicroSeries.FUNCTIONS:
            setattr(self, fn_name, _weighted_agg(fn_name))
