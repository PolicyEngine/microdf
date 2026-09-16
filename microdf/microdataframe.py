import copy
import logging
import warnings
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from microdf.microseries import MicroSeries, MicroSeriesGroupBy
from microdf._weights import (
    WeightPropagationMixin,
    aligned_weights,
    finalize_weights,
    weight_series,
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
        :type weights: np.array
        """
        super().__init__(*args, **kwargs)
        # pandas normalizes mixed-dimensional concat inputs through this
        # constructor, either as a Series or a one-column mapping. Preserve
        # that Series' row weights before concat loses the original input.
        weight_source = args[0] if args else kwargs.get("data")
        if isinstance(weight_source, dict) and len(weight_source) == 1:
            weight_source = next(iter(weight_source.values()))
        if weights is None and isinstance(weight_source, MicroSeries):
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

    @wraps(pd.DataFrame.cov)
    def cov(self, *args, **kwargs) -> pd.DataFrame:
        # Column summaries have no observation weights, even if labels match.
        result = pd.DataFrame(self, copy=False).cov(*args, **kwargs)
        return result.__finalize__(self, method="cov")

    @wraps(pd.DataFrame.corr)
    def corr(self, *args, **kwargs) -> pd.DataFrame:
        result = pd.DataFrame(self, copy=False).corr(*args, **kwargs)
        return result.__finalize__(self, method="corr")

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
        :type weights: np.array
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
            weight_values = np.asarray(self.weights.values, dtype=float)
            super().reset_index(
                level=level,
                drop=drop,
                inplace=True,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
            self.weights = pd.Series(weight_values, index=self.index, dtype=float)
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
            out = MicroDataFrame(res, weights=self.weights.values)
            # Ensure weights align to res.index (reset_index changes the
            # index but preserves row order, so pass values positionally).
            out.weights = pd.Series(
                np.asarray(self.weights.values, dtype=float),
                index=out.index,
                dtype=float,
            )
            return out

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

    @get_args_as_micro_series()
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
        staged["__tmp_weights"] = np.asarray(self.weights.values, dtype=float)
        gb = staged.groupby(by, *args, **kwargs)
        weights = copy.deepcopy(gb["__tmp_weights"])
        for col in staged.columns:  # df.groupby(...)[col]s use weights
            res = gb[col]
            res.__class__ = MicroSeriesGroupBy
            res._init()
            res.weights = weights
            setattr(gb, col, res)
        gb.__class__ = MicroDataFrameGroupBy
        gb._init(by)
        return gb

    @get_args_as_micro_series()
    def poverty_rate(self, income: str, threshold: str) -> float:
        """Calculate poverty rate, i.e., the population share with income below
        their poverty threshold.

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
        """Calculate deep poverty rate, i.e., the population share with income
        below half their poverty threshold.

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
        """Calculate poverty gap, i.e., the total gap between income and
        poverty thresholds for all people in poverty.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Poverty gap.
        :rtype: float
        """
        gaps = (threshold - income)[threshold > income]
        return gaps.sum()

    @get_args_as_micro_series()
    def deep_poverty_gap(self, income: str, threshold: str) -> float:
        """Calculate deep poverty gap, i.e., the total gap between income and
        half of poverty thresholds for all people in deep poverty.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Deep poverty gap.
        :rtype: float
        """
        deep_threshold = threshold / 2
        gaps = (deep_threshold - income)[deep_threshold > income]
        return gaps.sum()

    @get_args_as_micro_series()
    def squared_poverty_gap(self, income: str, threshold: str) -> float:
        """Calculate squared poverty gap, i.e., the total squared gap between
        income and poverty thresholds for all people in poverty. Also known as
        the poverty severity index.

        :param income: Column indicating income.
        :type income: str
        :param threshold: Column indicating threshold.
        :type threshold: str
        :return: Squared poverty gap.
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
    def _init(self, by: Union[str, List]):
        self._by = by
        self.columns = list(self.obj.columns)
        if isinstance(by, list):
            for column in by:
                self.columns.remove(column)
        elif isinstance(by, str):
            self.columns.remove(by)
        self.columns.remove("__tmp_weights")
        # Filter to only numeric columns
        self.numeric_columns = [
            col for col in self.columns if pd.api.types.is_numeric_dtype(self.obj[col])
        ]
        # Store reference to weights groupby for column selection
        self._weights_groupby = copy.deepcopy(super().__getitem__("__tmp_weights"))
        for fn_name in MicroSeries.SCALAR_FUNCTIONS:

            def get_fn(name):
                def fn(*args, **kwargs):
                    results = {}
                    for col in self.numeric_columns:
                        try:
                            results[col] = getattr(getattr(self, col), name)(
                                *args, **kwargs
                            )
                        except TypeError as exc:
                            # Skip columns whose dtype can't take this aggregation.
                            # Deliberately narrow: catching every Exception here also
                            # swallowed real errors (e.g. the ValueError from
                            # gini(negatives=...)) and returned a silently truncated
                            # result instead of raising.
                            logger.debug("skipping column %s in %s: %s", col, name, exc)
                    # Return plain DataFrame - aggregated results don't have
                    # per-row weights (weights were already applied)
                    return pd.DataFrame(results) if results else pd.DataFrame()

                return fn

            setattr(self, fn_name, get_fn(fn_name))
        for fn_name in MicroSeries.VECTOR_FUNCTIONS:

            def get_fn(name) -> Callable:
                def fn(*args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
                    results = {}
                    for col in self.numeric_columns:
                        try:
                            results[col] = getattr(getattr(self, col), name)(
                                *args, **kwargs
                            )
                        except TypeError as exc:
                            # Skip columns whose dtype can't take this aggregation.
                            # Deliberately narrow: catching every Exception here also
                            # swallowed real errors (e.g. the ValueError from
                            # gini(negatives=...)) and returned a silently truncated
                            # result instead of raising.
                            logger.debug("skipping column %s in %s: %s", col, name, exc)
                    # Return plain DataFrame - aggregated results don't have
                    # per-row weights (weights were already applied)
                    return pd.DataFrame(results) if results else pd.DataFrame()

                return fn

            setattr(self, fn_name, get_fn(fn_name))

    def __getitem__(
        self, key: Union[str, List]
    ) -> Union["MicroSeriesGroupBy", "MicroDataFrameGroupBy"]:
        """Select columns from the groupby object while preserving weights.

        This ensures that operations like groupby(col)["y"].sum() or
        groupby(col)[["y"]].sum() use weighted aggregation.

        :param key: Column name or list of column names
        :return: MicroSeriesGroupBy for single column, MicroDataFrameGroupBy
            for multiple columns
        """
        if isinstance(key, str):
            # Single column - return MicroSeriesGroupBy
            result = super().__getitem__(key)
            result.__class__ = MicroSeriesGroupBy
            result._init()
            result.weights = self._weights_groupby
            return result
        else:
            # Multiple columns - return a new MicroDataFrameGroupBy
            # with only the selected columns
            result = super().__getitem__(key)
            result.__class__ = MicroDataFrameGroupBy
            # Re-initialize with the subset of columns
            result._by = self._by
            result.columns = list(key) if hasattr(key, "__iter__") else [key]
            result.numeric_columns = [
                col
                for col in result.columns
                if pd.api.types.is_numeric_dtype(result.obj[col])
            ]
            result._weights_groupby = self._weights_groupby
            # Set up the column attributes as MicroSeriesGroupBy
            for col in result.columns:
                col_gb = super().__getitem__(col)
                col_gb.__class__ = MicroSeriesGroupBy
                col_gb._init()
                col_gb.weights = self._weights_groupby
                setattr(result, col, col_gb)
            # Set up the scalar and vector functions
            for fn_name in MicroSeries.SCALAR_FUNCTIONS:

                def get_scalar_fn(name, res):
                    def fn(*args, **kwargs):
                        results = {}
                        for col in res.numeric_columns:
                            try:
                                results[col] = getattr(getattr(res, col), name)(
                                    *args, **kwargs
                                )
                            except TypeError as exc:
                                # Skip columns whose dtype can't take this aggregation.
                                # Deliberately narrow: catching every Exception here also
                                # swallowed real errors (e.g. the ValueError from
                                # gini(negatives=...)) and returned a silently truncated
                                # result instead of raising.
                                logger.debug(
                                    "skipping column %s in %s: %s", col, name, exc
                                )
                        # Return plain DataFrame - aggregated results don't
                        # have per-row weights (weights were already applied)
                        return pd.DataFrame(results) if results else pd.DataFrame()

                    return fn

                setattr(result, fn_name, get_scalar_fn(fn_name, result))
            for fn_name in MicroSeries.VECTOR_FUNCTIONS:

                def get_vector_fn(name, res):
                    def fn(*args, **kwargs):
                        results = {}
                        for col in res.numeric_columns:
                            try:
                                results[col] = getattr(getattr(res, col), name)(
                                    *args, **kwargs
                                )
                            except TypeError as exc:
                                # Skip columns whose dtype can't take this aggregation.
                                # Deliberately narrow: catching every Exception here also
                                # swallowed real errors (e.g. the ValueError from
                                # gini(negatives=...)) and returned a silently truncated
                                # result instead of raising.
                                logger.debug(
                                    "skipping column %s in %s: %s", col, name, exc
                                )
                        # Return plain DataFrame - aggregated results don't
                        # have per-row weights (weights were already applied)
                        return pd.DataFrame(results) if results else pd.DataFrame()

                    return fn

                setattr(result, fn_name, get_vector_fn(fn_name, result))
            return result
