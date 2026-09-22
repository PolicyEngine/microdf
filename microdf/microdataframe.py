import logging
import warnings
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from microdf.microseries import (
    MicroSeries,
    MicroSeriesGroupBy,
    _pair_options,
    _usable_pair,
    _validate_frequency_weights,
    _weighted_correlation,
    _weighted_covariance,
)
from microdf._weights import (
    WeightPropagationMixin,
    aligned_weights,
    finalize_weights,
    weight_series,
    require_equal_weights,
)

logger = logging.getLogger(__name__)


class MicroDataFrame(WeightPropagationMixin, pd.DataFrame):
    # Declare weight state as pandas metadata. pandas includes
    # _metadata attributes in the pickle state, so weights now survive
    # pickling, to_pickle/read_pickle and copy.deepcopy instead of
    # vanishing and leaving an AttributeError on the next aggregation.
    # Retain the column name for set_weights(..., preserve_old=True).
    _metadata = pd.DataFrame._metadata + ["weights", "weights_col"]

    def __init__(self, *args, weights=None, **kwargs):
        """A DataFrame-inheriting class for weighted microdata.

        Weights can be provided at initialisation, or using set_weights or
        set_weight_col.

        :param weights: Array of weights.
        :type weights: np.ndarray
        """
        super().__init__(*args, **kwargs)
        # pandas normalizes mixed-dimensional concat inputs through this
        # constructor, either as a Series or a one-column mapping. Preserve
        # that Series' row weights before concat loses the original input.
        weight_source = args[0] if args else kwargs.get("data")
        if isinstance(weight_source, dict) and len(weight_source) == 1:
            weight_source = next(iter(weight_source.values()))
        if weights is None and isinstance(weight_source, (MicroSeries, MicroDataFrame)):
            weights = aligned_weights(weight_source, self.index)
        self.weights = weight_series(np.ones(len(self)), self.index)
        self.weights_col = None
        self.set_weights(weights)
        self._link_all_weights()
        self.override_df_functions()

    @property
    def _constructor(self):
        return MicroDataFrame

    # A row or a column-summary Series has no per-observation weights.
    # _ixs wraps column selections using their unambiguous row provenance.
    _constructor_sliced = pd.Series

    def _ixs(self, i, axis=0):
        result = pd.DataFrame(self, copy=False)._ixs(i, axis=axis)
        if axis == 1:
            return MicroSeries(result, weights=self.weights)
        return result

    def _get_item_cache(self, item):
        # Weight arrays are independently mutable; cached column wrappers would
        # retain stale copies after an in-place edit to frame.weights.
        return self._ixs(self.columns.get_loc(item), axis=1)

    def __finalize__(self, other, method=None, **kwargs):
        previous = self.__dict__.get("weights")
        super().__finalize__(other, method=method, **kwargs)
        return finalize_weights(self, other, method, previous)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for value in inputs:
            if value is not self:
                require_equal_weights(self, value)
        if method != "__call__":
            raise NotImplementedError(
                "Use a weighted frame reduction or pd.DataFrame(df) explicitly"
            )
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        def restore(value):
            if isinstance(value, pd.DataFrame):
                return self._weighted_result(value, aligned_weights(self, value.index))
            return value

        return (
            tuple(restore(value) for value in result)
            if isinstance(result, tuple)
            else restore(result)
        )

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        """Apply row functions while retaining row weights on the result."""
        if self._get_axis_number(axis) != 1:
            # Column callbacks must receive the weighted series.
            if raw or result_type is not None or not callable(func):
                raise NotImplementedError("Use pd.DataFrame(df) for this apply form")
            return pd.Series(
                {col: func(self[col], *args, **kwargs) for col in self.columns}
            )
        result = pd.DataFrame(self).apply(
            func, axis=1, raw=raw, result_type=result_type, args=args, **kwargs
        )
        cls = MicroSeries if isinstance(result, pd.Series) else MicroDataFrame
        return cls(result, weights=self.weights)

    def dropna(
        self,
        *,
        axis=0,
        how=None,
        thresh=None,
        subset=None,
        inplace=False,
        ignore_index=False,
    ):
        """Drop missing observations and their weights using row positions."""
        plain = pd.DataFrame(self).copy(deep=False)
        axis = self._get_axis_number(axis)
        original_index = plain.index
        plain.index = pd.RangeIndex(len(plain))
        options = {"axis": axis}
        if subset is not None:
            if axis == 1:
                raise NotImplementedError(
                    "dropna(axis=1, subset=...) requires pd.DataFrame(df)"
                )
            options["subset"] = subset
        if how is not None:
            options["how"] = how
        if thresh is not None:
            options["thresh"] = thresh
        result = plain.dropna(**options)
        positions = np.asarray(result.index, dtype=int)
        result.index = (
            pd.RangeIndex(len(result))
            if ignore_index
            else original_index.take(positions)
        )
        return self._finish_row_operation(result, positions, inplace)

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
        observed=True,
        sort=True,
        **kwargs,
    ):
        """Build a pivot table by applying estimators to weighted groups."""
        plain = pd.DataFrame(self).reset_index(drop=True)

        def wrap(func):
            if isinstance(func, dict):
                return {key: wrap(value) for key, value in func.items()}
            if isinstance(func, (list, tuple)):
                return [wrap(value) for value in func]

            def aggregate(series):
                positions = np.asarray(series.index, dtype=int)
                weighted = MicroSeries(
                    series.to_numpy(),
                    index=self.index.take(positions),
                    name=series.name,
                    weights=self.weights.iloc[positions],
                )
                return (
                    getattr(weighted, func)(**kwargs)
                    if isinstance(func, str)
                    else func(weighted, **kwargs)
                )

            aggregate.__name__ = (
                func
                if isinstance(func, str)
                else getattr(func, "__name__", "aggregate")
            )
            return aggregate

        return plain.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=wrap(aggfunc),
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort,
        )

    def cov(
        self,
        min_periods: Optional[int] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> pd.DataFrame:
        """Pairwise frequency-weighted covariance of the columns.

        Every cell uses the estimator of :meth:`MicroSeries.cov` with this
        frame's weights: ``sum(w * (x - xmean) * (y - ymean)) / (sum(w) -
        ddof)`` over the rows where both columns are present and the weight
        is positive. Integer weights therefore match ``pandas.DataFrame.cov``
        on the replicated sample. Missing values are removed pairwise, so
        each cell can use a different set of rows, as in pandas.

        The result is a plain ``pandas.DataFrame``: it summarises columns,
        so it carries no row weights.

        :param min_periods: Minimum usable row pairs per cell, not the sum
            of frequency weights. Cells with fewer are NaN. Defaults to 1.
        :param ddof: Degrees of freedom subtracted from the weight total.
        :param numeric_only: Use only numeric columns. Otherwise every column
            is converted to float, raising the error pandas raises.
        :returns: Covariance matrix indexed by column in both directions.
        """
        return self._weighted_pairwise(
            "cov", min_periods=min_periods, ddof=ddof, numeric_only=numeric_only
        )

    def corr(
        self,
        method: str = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> pd.DataFrame:
        """Pairwise frequency-weighted Pearson correlation of the columns.

        Every cell uses the estimator of :meth:`MicroSeries.corr` with this
        frame's weights over the rows where both columns are present and the
        weight is positive. Constant columns give NaN. Only ``"pearson"`` is
        supported, as on :meth:`MicroSeries.corr`; for an unweighted rank
        correlation convert to ``pandas.DataFrame`` first.

        The result is a plain ``pandas.DataFrame``: it summarises columns, so
        it carries no row weights.

        :param method: Only "pearson" is supported.
        :param min_periods: Minimum usable row pairs per cell, not the sum of
            frequency weights. Cells with fewer are NaN.
        :param numeric_only: Use only numeric columns. Otherwise every column
            is converted to float, raising the error pandas raises.
        :returns: Correlation matrix indexed by column in both directions.
        """
        if method != "pearson":
            raise ValueError("weighted correlation only supports method='pearson'")
        return self._weighted_pairwise(
            "corr", min_periods=min_periods, ddof=1, numeric_only=numeric_only
        )

    def _weighted_pairwise(
        self,
        statistic: str,
        *,
        min_periods: Optional[int],
        ddof: int,
        numeric_only: bool,
    ) -> pd.DataFrame:
        """Fill a symmetric column matrix one usable pair at a time."""
        min_periods, ddof = _pair_options(min_periods, ddof)
        data = self._get_numeric_data() if numeric_only else self
        frame = pd.DataFrame(data, copy=False)
        # The conversion pandas uses, so non-numeric columns raise its error.
        values = frame.to_numpy(dtype=float, na_value=np.nan)
        weights = np.asarray(self.weights, dtype=float)
        _validate_frequency_weights(weights)
        columns = frame.columns
        matrix = np.full((len(columns), len(columns)), np.nan)
        for i in range(len(columns)):
            for j in range(i, len(columns)):
                pair = _usable_pair(
                    values[:, i], values[:, j], weights, min_periods, ddof, True
                )
                if pair is None:
                    continue
                x, y, pair_weights, denominator = pair
                if statistic == "cov":
                    cell = _weighted_covariance(x, y, pair_weights, denominator)
                else:
                    cell = _weighted_correlation(x, y, pair_weights)
                matrix[i, j] = matrix[j, i] = cell
        result = pd.DataFrame(matrix, index=columns, columns=columns)
        # Column summaries have no observation weights, even if labels match.
        return pd.DataFrame.__finalize__(result, self, method=statistic)

    def __setstate__(self, state) -> None:
        """Restore a pickled MicroDataFrame.

        The weighted aggregations are installed as per-instance closures by
        ``override_df_functions``, which only runs in ``__init__`` — a path
        unpickling skips. Without reinstalling them, ``mdf.sum()`` on an
        unpickled frame silently fell through to the unweighted pandas
        implementation.
        """
        super().__setstate__(state)
        if getattr(self, "weights", None) is None:
            self._link_all_weights()
        self.override_df_functions()

    def override_df_functions(self) -> None:
        """Override DataFrame functions to work with weighted operations."""
        for name in MicroSeries.FUNCTIONS:
            if name == "sum":
                # Sum has its own axis-aware signature and result types.
                continue
            elif name in MicroSeries.SCALAR_FUNCTIONS:
                setattr(self, name, self._create_scalar_function(name))
            elif name in MicroSeries.VECTOR_FUNCTIONS:
                setattr(self, name, self._create_vector_function(name))
            elif name in MicroSeries.AGNOSTIC_FUNCTIONS:
                setattr(self, name, self._create_agnostic_function(name))

    def sum(
        self,
        axis: Optional[Union[int, str]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs,
    ) -> Union[pd.Series, MicroSeries, float]:
        """Sum numeric columns, weighting reductions across observations.

        Column sums (axis=0 or 'index') apply observation weights and return a
        plain Series. Row sums (axis=1 or 'columns') do not multiply row values
        by weights; they return a MicroSeries with an independent copy of the
        original weights for subsequent weighted aggregation.

        Non-numeric columns are excluded, matching other MicroDataFrame
        aggregations. skipna and min_count follow pandas sum semantics.
        Explicit axis=None follows the installed pandas version: column sums in
        pandas 2, and a weighted total over both axes in pandas 3.
        """
        axis_number = None if axis is None else self._get_axis_number(axis)
        values = pd.DataFrame(self)
        numeric_columns = [
            pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes
        ]
        values = values.iloc[:, numeric_columns]
        if axis_number != 1 and self.weights is not None:
            values = values.mul(self.weights, axis=0)
        result = values.sum(
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )
        if axis_number == 1:
            weights = (
                self.weights.copy()
                if self.weights is not None
                else pd.Series(1.0, index=self.index)
            )
            return MicroSeries(result, weights=weights)
        return result

    def _create_scalar_function(self, name: str) -> Callable:
        """Create a scalar function that returns a Series of results.

        :param name: Name of the function to create
        :return: Function that applies the operation to all columns
        """

        def fn(*args, **kwargs) -> pd.Series:
            kwargs.pop("numeric_only", None)
            axis = kwargs.pop("axis", 0)
            if axis not in (0, "index"):
                raise NotImplementedError(
                    f"Weighted {name} only supports axis=0; use pd.DataFrame(df)"
                )
            return pd.Series(
                {
                    col: getattr(self[col], name)(*args, **kwargs)
                    for col in self.columns
                    if pd.api.types.is_numeric_dtype(self[col])
                }
            )

        return fn

    def _create_vector_function(self, name: str) -> Callable:
        """Create a vector function that returns a DataFrame of results.

        :param name: Name of the function to create
        :return: Function that applies the operation to all columns
        """

        def fn(*args, **kwargs) -> pd.DataFrame:
            results = []
            columns = []
            for col in self.columns:
                if pd.api.types.is_numeric_dtype(self[col]):
                    try:
                        result = getattr(self[col], name)(*args, **kwargs)
                        results.append(result)
                        columns.append(col)
                    except TypeError as exc:
                        # Skip columns whose dtype can't take this aggregation.
                        # Deliberately narrow: catching every Exception here also
                        # swallowed real errors (e.g. the ValueError from
                        # gini(negatives=...)) and returned a silently truncated
                        # result instead of raising.
                        logger.debug("skipping column %s in %s: %s", col, name, exc)

            if results:
                df = pd.DataFrame(results)
                df.index = columns
                return df
            else:
                return pd.DataFrame()

        return fn

    def _create_agnostic_function(self, name: str) -> Callable:
        """Create a function that can be either scalar or vector based on
        input.

        :param name: Name of the function to create
        :return: Function that applies the operation to all columns
        """

        def fn(*args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
            # Check if first argument is array-like
            is_array = len(args) > 0 and hasattr(args[0], "__len__")

            if is_array:
                # Use vector function behavior
                results = []
                columns = []
                for col in self.columns:
                    if pd.api.types.is_numeric_dtype(self[col]):
                        try:
                            result = getattr(self[col], name)(*args, **kwargs)
                            results.append(result)
                            columns.append(col)
                        except TypeError as exc:
                            # Skip columns whose dtype can't take this aggregation.
                            # Deliberately narrow: catching every Exception here also
                            # swallowed real errors (e.g. the ValueError from
                            # gini(negatives=...)) and returned a silently truncated
                            # result instead of raising.
                            logger.debug("skipping column %s in %s: %s", col, name, exc)

                if results:
                    df = pd.DataFrame(results)
                    df.index = columns
                    return df
                else:
                    return pd.DataFrame()
            else:
                # Use scalar function behavior
                results = {}
                for col in self.columns:
                    if pd.api.types.is_numeric_dtype(self[col]):
                        try:
                            results[col] = getattr(self[col], name)(*args, **kwargs)
                        except TypeError as exc:
                            # Skip columns whose dtype can't take this aggregation.
                            # Deliberately narrow: catching every Exception here also
                            # swallowed real errors (e.g. the ValueError from
                            # gini(negatives=...)) and returned a silently truncated
                            # result instead of raising.
                            logger.debug("skipping column %s in %s: %s", col, name, exc)
                return pd.Series(results)

        return fn

    def get_args_as_micro_series(*kwarg_names: tuple) -> Callable:
        """Decorator for auto-parsing column names into MicroSeries objects.

        If given, kwarg_names limits arguments checked to keyword arguments
        specified.

        :param arg_names: argument names to restrict to.
        :type arg_names: str
        """

        def arg_series_decorator(fn) -> Callable:
            @wraps(fn)
            def series_function(
                self, *args, **kwargs
            ) -> Union[pd.Series, pd.DataFrame]:
                new_args = []
                new_kwargs = {}
                if len(kwarg_names) == 0:
                    for value in args:
                        if isinstance(value, str):
                            if value not in self.columns:
                                raise Exception("Column not found")
                            new_args += [self[value]]
                        else:
                            new_args += [value]
                    for name, value in kwargs.items():
                        if isinstance(value, str) and (
                            len(kwarg_names) == 0 or name in kwarg_names
                        ):
                            if value not in self.columns:
                                raise Exception("Column not found")
                            new_kwargs[name] = self[value]
                        else:
                            new_kwargs[name] = value
                return fn(self, *new_args, **new_kwargs)

            return series_function

        return arg_series_decorator

    def __setitem__(self, *args, **kwargs) -> None:
        super().__setitem__(*args, **kwargs)
        self._link_all_weights()

    def _link_weights(self, column) -> None:
        # In pandas 3.0+, we can't modify column classes in-place due to CoW.
        # Instead, we rely on __getitem__ to wrap columns as MicroSeries on
        # access. This method is kept for backward compatibility but is now
        # a no-op.
        pass

    def _link_all_weights(self) -> None:
        if self.weights is None or len(self.weights) == 0:
            if len(self) > 0:
                self.set_weights(np.ones((len(self))))
        # In pandas 3.0+, columns are wrapped as MicroSeries on access via
        # __getitem__, not stored as MicroSeries internally.

    def set_weights(
        self,
        weights: Union[np.ndarray, str],
        preserve_old: Optional[bool] = False,
    ) -> None:
        """Sets the weights for the MicroDataFrame.

        If a string is received, it will be assumed to be the column name of
        the weight column.

        :param weights: Array of weights.
        :param preserve_old: If True, keeps the old weights as a column when
            new weights are provided.
        :type weights: np.ndarray
        """
        if preserve_old and self.weights_col is not None:
            self["old_" + self.weights_col] = self.weights

        if isinstance(weights, str):
            self.weights_col = weights
            # Keep stored weights independent from edits to the source column.
            self.weights = pd.Series(
                np.array(self[weights], copy=True),
                index=self.index,
                dtype=float,
            )
            self._link_all_weights()
        elif weights is not None:
            if len(weights) != len(self):
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match "
                    f"length of DataFrame ({len(self)})."
                )
            self.weights_col = None
            # Align weights to self.index. Without this, weighted ops
            # (self[col].multiply(self.weights) in .sum()) align on
            # label, so any non-default index silently produces all-NaN
            # and aggregations collapse to 0. If a Series is passed in,
            # strip its index so we position-align to self.index.
            if isinstance(weights, pd.Series):
                weights = weights.values
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.weights = pd.Series(
                    np.asarray(weights), index=self.index, dtype=float
                )
            self._link_all_weights()

    def set_weight_col(self, column: str, preserve_old: Optional[bool] = False) -> None:
        """Sets the weights for the MicroDataFrame by specifying the name of
        the weight column.

        .. deprecated:: 1.0.2
            Use :meth:`set_weights` with a string argument instead.
            This method will be removed in a future version.

        :param column: Name of the column to use as weights.
        :param preserve_old: If True, keeps the old weights as a column when
            new weights are provided.
        :type column: str
        """
        import warnings

        warnings.warn(
            "set_weight_col is deprecated and will be removed in a "
            "future version. Use set_weights(column_name) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if preserve_old and self.weights_col is not None:
            self["old_" + self.weights_col] = self.weights

        # Delegate to set_weights: it validates length and builds an
        # index-aligned float Series rather than a bare ndarray.
        self.set_weights(column)

    def nullify_weights(self) -> None:
        """Set all weights to 1, effectively making the DataFrame unweighted.

        This is useful for comparing weighted and unweighted statistics or when
        you want to temporarily ignore weights.
        """
        # Route through set_weights so self.weights stays an index-aligned
        # float Series. Assigning a bare ndarray here broke every caller
        # that treats it as a Series (equals(), reindex() in __getitem__).
        self.set_weights(np.ones(len(self)))

    def __getitem__(self, key):
        return super().__getitem__(key)

    def catch_series_relapse(self) -> None:
        # In pandas 3.0+, we don't need to track series class changes since
        # __getitem__ always wraps columns as MicroSeries on access.
        pass

    def __setattr__(self, key, value):
        weights = self.__dict__.get("weights") if key == "index" else None
        super().__setattr__(key, value)
        if weights is not None and len(weights) == len(self.index):
            self.weights = weight_series(weights, self.index)

    def reset_index(
        self,
        level: Optional[int] = None,
        drop: Optional[bool] = False,
        inplace: Optional[bool] = False,
        col_level: Optional[int] = 0,
        col_fill: Optional[str] = "",
        allow_duplicates: Optional[bool] = None,
        names: Optional[List[str]] = None,
    ) -> Union["MicroDataFrame", None]:
        """Reset the index of the MicroDataFrame.

        This method supports all parameters of pandas DataFrame.reset_index(),
        including the 'inplace' parameter.

        :param level: Only remove the given levels from the index. Removes all
            levels by default.
        :param drop: Do not try to insert index into dataframe columns. This
            resets the index to the default integer index.
        :param inplace: Modify the DataFrame in place (do not create a new
            object).
        :param col_level: If the columns have multiple levels, determines which
            level the labels are inserted into.
        :param col_fill: If the columns have multiple levels, determines how
            the other levels are named.
        :param allow_duplicates: Allow duplicate column labels to be created.
        :param names: Using the given string, rename the DataFrame column which
            contains the index data.
        :return: MicroDataFrame with reset index or None if inplace=True.
        """
        if inplace:
            # Snapshot weight *values* positionally — the index is about
            # to change and reset_index preserves row order.
            weight_values = np.array(self.weights, dtype=float, copy=True)
            super().reset_index(
                level=level,
                drop=drop,
                inplace=True,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
            self.weights = weight_series(weight_values, self.index)
            self._link_all_weights()
            return None
        else:
            res = super().reset_index(
                level=level,
                drop=drop,
                inplace=False,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
            # Own a positional copy: reset_index changes labels but
            # preserves row order.
            return MicroDataFrame(res, weights=weight_series(self.weights, res.index))

    def copy(self, deep: Optional[bool] = True) -> "MicroDataFrame":
        return super().copy(deep)

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Drop specified labels from rows or columns.

        This method supports all parameters of pandas DataFrame.drop(),
        including the 'inplace' parameter.

        :param labels: Index or column labels to drop.
        :param axis: Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        :param index: Alternative to specifying axis (labels, axis=0 is
            equivalent to index=labels).
        :param columns: Alternative to specifying axis (labels, axis=1 is
            equivalent to columns=labels).
        :param level: For MultiIndex, level from which the labels will be
            removed.
        :param inplace: If False, return a copy. Otherwise, do operation
            inplace and return None.
        :param errors: If 'ignore', suppress error and only existing labels are
            dropped.
        :return: MicroDataFrame or None if inplace=True.
        """
        return super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    ):
        """Merge DataFrame or named Series objects with a database-style join.

        This method overrides pandas DataFrame.merge() to return a
        MicroDataFrame.

        :param right: Object to merge with.
        :param how: Type of merge to be performed.
        :param on: Column or index level names to join on.
        :param left_on: Column or index level names to join on in the left
            DataFrame.
        :param right_on: Column or index level names to join on in the right
            DataFrame.
        :param left_index: Use the index from the left DataFrame as the join
            key(s).
        :param right_index: Use the index from the right DataFrame as the join
            key(s).
        :param sort: Sort the join keys lexicographically in the result
            DataFrame.
        :param suffixes: A length-2 sequence where each element is optionally a
            string indicating the suffix to add to overlapping column names.
        :param copy: If False, avoid copy if possible.
        :param indicator: If True, adds a column to output DataFrame called
            "_merge".
        :param validate: If specified, checks if merge is of specified type.
        :return: MicroDataFrame with merged data.
        """
        # Attach the left weights as a temporary column so pandas' merge
        # propagates them onto every surviving output row (including
        # many-to-many row duplications, inner-join filtering, and
        # left-with-missing NaNs). We then strip the column back off.
        tmp = "__microdf_weights__"
        # Avoid clobbering if this exact name is already used.
        while tmp in self.columns or tmp in right.columns:
            tmp += "_"
        left_df = pd.DataFrame(self).copy()
        left_df[tmp] = np.asarray(self.weights.values, dtype=float)
        res = left_df.merge(
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )
        # Pull out the propagated weights. Rows with no left match in a
        # right/outer join get NaN weight — fill with 0 so they don't
        # poison later aggregations (a user who needs a different
        # convention can override afterwards).
        merged_weights = res[tmp].fillna(0).to_numpy(dtype=float)
        res = res.drop(columns=[tmp])
        out = MicroDataFrame(res, weights=merged_weights)
        # Ensure the weights Series aligns with res.index regardless of
        # the default-RangeIndex behavior of set_weights.
        out.weights = pd.Series(merged_weights, index=out.index, dtype=float)
        return out

    def __getattr__(self, name):
        """Allow accessing columns as attributes (e.g., df.column_name).

        This enables more intuitive column access while preserving MicroSeries
        functionality when accessing columns.

        :param name: Attribute name to access
        :return: MicroSeries if the attribute is a column, otherwise delegates
            to parent
        """
        if name in self.columns:
            return self[name]
        return super().__getattr__(name)

    def equals(self, other: "MicroDataFrame") -> bool:
        equal_values = super().equals(other)
        equal_weights = self.weights.equals(other.weights)
        return equal_values and equal_weights

    def groupby(self, by: Union[str, List], *args, **kwargs) -> "MicroDataFrameGroupBy":
        """Returns a GroupBy object with MicroSeriesGroupBy objects for each
        column.

        :param by: column to group by
        :type by: Union[str, List]

        return: DataFrameGroupBy object with columns using weights
        rtype: DataFrameGroupBy
        """
        # Build the groupby on a *copy* that carries a ``__tmp_weights``
        # column. We used to set this column on ``self`` directly, which
        # permanently leaked the weight column onto the caller's
        # DataFrame — any later ``df.sum()`` or ``list(df.columns)``
        # would then include it.
        staged = pd.DataFrame(self).copy()
        if "__tmp_weights" in staged.columns:
            raise ValueError("Rename the reserved __tmp_weights column before grouping")
        staged["__tmp_weights"] = np.asarray(self.weights.values, dtype=float)
        gb = staged.groupby(by, *args, **kwargs)
        gb.__class__ = MicroDataFrameGroupBy
        gb._init(by)
        return gb

    @get_args_as_micro_series()
    def poverty_rate(self, income: str, threshold: str) -> float:
        """Return the weighted headcount share strictly below the poverty
        threshold.

        Divide the weight of people in poverty by total population weight. This
        is the Foster-Greer-Thorbecke (FGT) headcount index, FGT(0).

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Poverty rate between zero and one.
        :rtype: float
        """
        pov = income < threshold
        return pov.sum() / pov.count()

    @get_args_as_micro_series()
    def deep_poverty_rate(self, income: str, threshold: str) -> float:
        """Return the weighted headcount share strictly below half the
        threshold.

        Divide the weight of people in deep poverty by total population weight.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Deep poverty rate between zero and one.
        :rtype: float
        """
        pov = income < (threshold / 2)
        return pov.sum() / pov.count()

    @get_args_as_micro_series()
    def poverty_gap(self, income: str, threshold: str) -> float:
        """Return the weighted aggregate poverty gap in income currency units.

        Sum weight times (threshold - income) over people strictly below their
        threshold. This aggregate is not the normalised FGT(1) index.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Weighted aggregate gap in income currency units.
        :rtype: float
        """
        gaps = (threshold - income)[threshold > income]
        return gaps.sum()

    @get_args_as_micro_series()
    def deep_poverty_gap(self, income: str, threshold: str) -> float:
        """Return the weighted aggregate deep poverty gap in income currency
        units.

        Sum weight times (threshold / 2 - income) over people strictly below
        half their threshold.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Weighted aggregate deep gap in income currency units.
        :rtype: float
        """
        deep_threshold = threshold / 2
        gaps = (deep_threshold - income)[deep_threshold > income]
        return gaps.sum()

    @get_args_as_micro_series()
    def squared_poverty_gap(self, income: str, threshold: str) -> float:
        """Return the weighted aggregate squared gap in squared currency units.

        Sum weight times (threshold - income) squared over people strictly
        below their threshold. This aggregate is not the normalised FGT(2)
        poverty severity index.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Weighted aggregate squared gap in squared income currency units.
        :rtype: float
        """
        gaps = (threshold - income)[threshold > income]
        squared_gaps = gaps**2
        return squared_gaps.sum()

    @get_args_as_micro_series()
    def poverty_count(
        self,
        income: Union[MicroSeries, str],
        threshold: Union[MicroSeries, str],
    ) -> int:
        """Calculates the number of entities with income below a poverty
        threshold.

        :param income: income array or column name
        :type income: Union[MicroSeries, str]

        :param threshold: threshold array or column name
        :type threshold: Union[MicroSeries, str]

        return: number of entities in poverty
        rtype: int
        """
        in_poverty = income < threshold
        return in_poverty.sum()

    def astype(
        self,
        dtype,
        copy: Optional[bool] = True,
        errors: Optional[str] = "raise",
    ) -> "MicroDataFrame":
        """Convert MicroDataFrame to specified data type while preserving
        weights.

        :param dtype: Data type to convert to. Can be numpy dtype, Python type,
            or dict.
        :param copy: Whether to make a copy of the data (default True).
        :param errors: How to handle conversion errors (default "raise").
        :return: New MicroDataFrame with converted data types and preserved
            weights.
        """
        converted_df = super().astype(dtype, copy=copy, errors=errors)
        return MicroDataFrame(
            converted_df, weights=self.weights.copy() if copy else self.weights
        )

    def __repr__(self) -> str:
        df = pd.DataFrame(self)
        df["weight"] = self.weights
        return df[[df.columns[-1]] + list(df.columns[:-1])].__repr__()


class MicroDataFrameGroupBy(pd.core.groupby.generic.DataFrameGroupBy):
    def _init(self, by, columns=None, weights=None):
        self._by = by
        self.columns = (
            columns
            if columns is not None
            else [
                col
                for col in self.obj.columns
                if col != "__tmp_weights" and col not in self.exclusions
            ]
        )
        self.numeric_columns = [
            col for col in self.columns if pd.api.types.is_numeric_dtype(self.obj[col])
        ]
        self._weights_groupby = (
            weights if weights is not None else self._gotitem("__tmp_weights", ndim=1)
        )
        for name in MicroSeries.FUNCTIONS + [
            "sem",
            "skew",
            "kurt",
            "kurtosis",
            "prod",
            "product",
            "idxmax",
            "idxmin",
            "rolling",
            "expanding",
            "ewm",
            "mode",
            "value_counts",
        ]:

            def reduction(*args, _name=name, **kwargs):
                kwargs.pop("numeric_only", None)
                result = pd.DataFrame(
                    {
                        col: getattr(self[col], _name)(*args, **kwargs)
                        for col in self.numeric_columns
                    }
                )
                return result if self.as_index else result.reset_index()

            setattr(self, name, reduction)

    def __getitem__(self, key):
        if pd.api.types.is_hashable(key):
            if key not in self.columns:
                raise KeyError(key)
            result = self._gotitem(key, ndim=1)
        else:
            result = super().__getitem__(key)
        if isinstance(result, pd.core.groupby.generic.SeriesGroupBy):
            result.__class__ = MicroSeriesGroupBy
            result._init()
            result.weights = self._weights_groupby
        else:
            result.__class__ = MicroDataFrameGroupBy
            result._init(self._by, list(key), self._weights_groupby)
        return result

    def aggregate(self, func=None, *args, **kwargs):
        """Apply named, dictionary, list and callable weighted aggregations."""
        if func is not None and (
            kwargs.get("engine") is not None or "engine_kwargs" in kwargs
        ):
            raise NotImplementedError(
                "Weighted aggregation does not support engine overrides"
            )
        numeric_only = kwargs.pop("numeric_only", False) if func is not None else False
        if func is None:
            if not kwargs or not all(
                isinstance(value, tuple) and len(value) == 2
                for value in kwargs.values()
            ):
                raise TypeError("Named aggregation requires output=(column, function)")
            results = {
                label: self[column].agg(reducer, *args)
                for label, (column, reducer) in kwargs.items()
            }
        elif isinstance(func, dict):
            results = {
                column: self[column].agg(reducer, *args, **kwargs)
                for column, reducer in func.items()
            }
        else:
            columns = (
                self.numeric_columns
                if numeric_only or isinstance(func, str)
                else self.columns
            )
            results = {
                column: self[column].agg(func, *args, **kwargs) for column in columns
            }
        if any(isinstance(value, pd.DataFrame) for value in results.values()):
            # pandas uses a second level for all columns when any reducer is a list.
            tables = {
                key: value
                if isinstance(value, pd.DataFrame)
                else value.to_frame(func[key] if isinstance(func, dict) else func)
                for key, value in results.items()
            }
            result = pd.concat(tables, axis=1)
        else:
            result = pd.DataFrame(results)
        return result if self.as_index else result.reset_index()

    agg = aggregate
