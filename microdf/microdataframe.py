import copy
import logging
import warnings
from functools import wraps
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from microdf.microseries import MicroSeries, MicroSeriesGroupBy

logger = logging.getLogger(__name__)


class MicroDataFrame(pd.DataFrame):
    def __init__(self, *args, weights=None, **kwargs):
        """A DataFrame-inheriting class for weighted microdata. Weights can be
        provided at initialisation, or using set_weights or set_weight_col.

        :param weights: Array of weights.
        :type weights: np.array
        """
        super().__init__(*args, **kwargs)
        self.weights = None
        self.set_weights(weights)
        self._link_all_weights()
        self.override_df_functions()

    def override_df_functions(self) -> None:
        """Override DataFrame functions to work with weighted operations."""
        for name in MicroSeries.FUNCTIONS:
            if name in MicroSeries.SCALAR_FUNCTIONS:
                setattr(self, name, self._create_scalar_function(name))
            elif name in MicroSeries.VECTOR_FUNCTIONS:
                setattr(self, name, self._create_vector_function(name))
            elif name in MicroSeries.AGNOSTIC_FUNCTIONS:
                setattr(self, name, self._create_agnostic_function(name))

    def _create_scalar_function(self, name: str) -> Callable:
        """Create a scalar function that returns a Series of results.

        :param name: Name of the function to create
        :return: Function that applies the operation to all columns
        """

        def fn(*args, **kwargs) -> pd.Series:
            results = pd.Series(
                [
                    getattr(self[col], name)(*args, **kwargs)
                    for col in self.columns
                ]
            )
            results.index = self.columns
            return results

        return fn

    def _create_vector_function(self, name: str) -> Callable:
        """Create a vector function that returns a DataFrame of results.

        :param name: Name of the function to create
        :return: Function that applies the operation to all columns
        """

        def fn(*args, **kwargs) -> pd.DataFrame:
            results = pd.DataFrame(
                [
                    getattr(self[col], name)(*args, **kwargs)
                    for col in self.columns
                ]
            )
            results.index = self.columns
            return results

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
                results = pd.DataFrame(
                    [
                        getattr(self[col], name)(*args, **kwargs)
                        for col in self.columns
                    ]
                )
                results.index = self.columns
                return results
            else:
                # Use scalar function behavior
                results = pd.Series(
                    [
                        getattr(self[col], name)(*args, **kwargs)
                        for col in self.columns
                    ]
                )
                results.index = self.columns
                return results

        return fn

    def get_args_as_micro_series(*kwarg_names: tuple) -> Callable:
        """Decorator for auto-parsing column names into MicroSeries objects. If
        given, kwarg_names limits arguments checked to keyword arguments
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
        # self[column] = ... triggers __setitem__, which forces pd.Series
        # this workaround avoids that
        self[column].__class__ = MicroSeries
        self[column].set_weights(self.weights)

    def _link_all_weights(self) -> None:
        if self.weights is None:
            if len(self) > 0:
                self.set_weights(np.ones((len(self))))
        for column in self.columns:
            if column != self.weights_col:
                self._link_weights(column)

    def set_weights(
        self,
        weights: Union[np.ndarray, str],
        preserve_old: Optional[bool] = False,
    ) -> None:
        """Sets the weights for the MicroDataFrame. If a string is received, it
        will be assumed to be the column name of the weight column.

        :param weights: Array of weights.
        :param preserve_old: If True, keeps the old weights as a column when
            new weights are provided.
        :type weights: np.array
        """
        if preserve_old and self.weights_col is not None:
            self["old_" + self.weights_col] = self.weights

        if isinstance(weights, str):
            self.weights_col = weights
            self.weights = pd.Series(self[weights], dtype=float)
        elif weights is not None:
            if len(weights) != len(self):
                raise ValueError(
                    f"Length of weights ({len(weights)}) does not match "
                    f"length of DataFrame ({len(self)})."
                )
            self.weights_col = None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.weights = pd.Series(weights, dtype=float)
            self._link_all_weights()

    def set_weight_col(
        self, column: str, preserve_old: Optional[bool] = False
    ) -> None:
        """Sets the weights for the MicroDataFrame by specifying the name of
        the weight column.

        :param weights: Array of weights.
        :param preserve_old: If True, keeps the old weights as a column when
            new weights are provided.
        :type weights: np.array
        """
        if preserve_old and self.weights_col is not None:
            self["old_" + self.weights_col] = self.weights

        self.weights = np.array(self[column])
        self.weights_col = column
        self._link_all_weights()

    def __getitem__(
        self, key: Union[str, List]
    ) -> Union[pd.Series, pd.DataFrame]:
        result = super().__getitem__(key)
        if isinstance(result, pd.DataFrame):
            try:
                weights = self.weights[key]
            except Exception:
                weights = self.weights
            return MicroDataFrame(result, weights=weights)
        return result

    def catch_series_relapse(self) -> None:
        for col in self.columns:
            if self[col].__class__ == pd.Series:
                self._link_weights(col)

    def __setattr__(self, key, value) -> None:
        super().__setattr__(key, value)
        self.catch_series_relapse()

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
            weights_backup = self.weights.copy()
            # Perform in-place reset on the parent DataFrame
            super().reset_index(
                level=level,
                drop=drop,
                inplace=True,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
            self.weights = weights_backup
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
            return MicroDataFrame(res, weights=self.weights)

    def copy(self, deep: Optional[bool] = True) -> "MicroDataFrame":
        res = super().copy(deep)
        # This changes the original columns to Series. Undo it:
        for col in self.columns:
            self[col] = MicroSeries(self[col])
        res = MicroDataFrame(res, weights=self.weights.copy(deep))
        return res

    def equals(self, other: "MicroDataFrame") -> bool:
        equal_values = super().equals(other)
        equal_weights = self.weights.equals(other.weights)
        return equal_values and equal_weights

    @get_args_as_micro_series()
    def groupby(
        self, by: Union[str, List], *args, **kwargs
    ) -> "MicroDataFrameGroupBy":
        """Returns a GroupBy object with MicroSeriesGroupBy objects for each
        column.

        :param by: column to group by
        :type by: Union[str, List]

        return: DataFrameGroupBy object with columns using weights
        rtype: DataFrameGroupBy
        """
        self["__tmp_weights"] = self.weights
        gb = super().groupby(by, *args, **kwargs)
        weights = copy.deepcopy(gb["__tmp_weights"])
        for col in self.columns:  # df.groupby(...)[col]s use weights
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
        self.columns = list(self.obj.columns)
        if isinstance(by, list):
            for column in by:
                self.columns.remove(column)
        elif isinstance(by, str):
            self.columns.remove(by)
        self.columns.remove("__tmp_weights")
        for fn_name in MicroSeries.SCALAR_FUNCTIONS:

            def get_fn(name):
                def fn(*args, **kwargs):
                    return MicroDataFrame(
                        {
                            col: getattr(getattr(self, col), name)(
                                *args, **kwargs
                            )
                            for col in self.columns
                        }
                    )

                return fn

            setattr(self, fn_name, get_fn(fn_name))
        for fn_name in MicroSeries.VECTOR_FUNCTIONS:

            def get_fn(name) -> Callable:
                def fn(*args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
                    return MicroDataFrame(
                        {
                            col: getattr(getattr(self, col), name)(
                                *args, **kwargs
                            )
                            for col in self.columns
                        }
                    )

                return fn

            setattr(self, fn_name, get_fn(fn_name))
