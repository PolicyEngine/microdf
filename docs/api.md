# API reference

`microdf` exposes two classes. `MicroSeries` is a `pandas.Series` carrying a weight
vector; `MicroDataFrame` is a `pandas.DataFrame` carrying a weight column. Both
behave like their pandas counterparts, and the methods below either add a
weighted estimator or preserve weights through an operation that would otherwise
drop them.

```python
import microdf as mdf

df = mdf.MicroDataFrame({"income": [10_000, 30_000, 120_000]}, weights=[800, 1_200, 50])
df.income.gini()
```

## MicroSeries

### Weighted aggregation

These have the same names as their pandas equivalents and return weighted results.

| Method | Signature | Description |
|---|---|---|
| `sum` | `(axis: Union[int, str, NoneType] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs) -> float` | Calculates the weighted sum of the MicroSeries. |
| `count` | `(skipna: bool = True) -> float` | Calculates the weighted count of the MicroSeries. |
| `mean` | `(skipna: bool = True) -> float` | Calculates the weighted mean of the MicroSeries. |
| `median` | `(skipna: bool = True) -> float` | Calculates the weighted median of the MicroSeries. |
| `quantile` | `(q: ndarray, skipna: bool = True) -> Series` | Calculates weighted quantiles of the MicroSeries. |
| `var` | `(ddof: int = 1, skipna: bool = True) -> float` | Calculates the weighted variance of the MicroSeries. |
| `std` | `(ddof: int = 1, skipna: bool = True) -> float` | Calculates the weighted standard deviation of the MicroSeries. |
| `cov` | `(other: Series, min_periods: Optional[int] = None, ddof: int = 1, *, skipna: bool = True) -> float` | Calculate frequency-weighted covariance with another Series. |
| `corr` | `(other: Series, method: str = 'pearson', min_periods: Optional[int] = None, *, ddof: int = 1, skipna: bool = True) -> float` | Calculate frequency-weighted Pearson correlation. |
| `rank` | `(pct: Optional[bool] = False) -> Series` | Weighted rank of each element. |

### Weight-preserving operations

Operations that change the shape or type of the data, overridden so weights stay aligned with their rows.

| Method | Signature | Description |
|---|---|---|
| `groupby` | `(*args, **kwargs) -> MicroSeriesGroupBy` | Group into `MicroSeriesGroupBy`, carrying weights into each group. |
| `cumsum` | `() -> Series` | Cumulative sum of value times weight. Returns a plain `pandas.Series`: the weights have been applied and are not carried forward, so this is the one method here that does not preserve them. |
| `astype` | `(dtype, copy: Optional[bool] = True, errors: Optional[str] = 'raise') -> MicroSeries` | Convert MicroSeries to specified data type while preserving weights. |
| `clip` | `(lower: Optional[float] = None, upper: Optional[float] = None, axis: Optional[int] = None, inplace: Optional[bool] = False, *args, **kwargs) -> MicroSeries` | Trim values at the given thresholds, preserving weights. |
| `round` | `(decimals: Optional[int] = 0, *args, **kwargs) -> MicroSeries` | Round each value, preserving weights. |
| `repeat` | `(repeats, axis=None)` | Repeat elements, repeating their weights alongside. |
| `sqrt` | `() -> MicroSeries` | Element-wise square root, preserving weights. |
| `copy` | `(deep: Optional[bool] = True)` | Copy the series and its weights. |
| `equals` | `(other: MicroSeries) -> bool` | True when both the values and the weights are equal. |
| `values` | *attribute* | Access underlying numpy array. |
| `to_numpy` | `(*args, **kwargs)` | Convert to numpy array. |

### Inequality and distribution

| Method | Signature | Description |
|---|---|---|
| `gini` | `(negatives: Optional[str] = None) -> float` | Calculates Gini index. |
| `top_1_pct_share` | `() -> float` | Calculates top 1% share. |
| `top_10_pct_share` | `() -> float` | Calculates top 10% share. |
| `top_50_pct_share` | `() -> float` | Calculates top 50% share. |
| `bottom_50_pct_share` | `() -> float` | Calculates bottom 50% share. |
| `top_0_1_pct_share` | `() -> float` | Calculates top 0.1% share. |
| `top_x_pct_share` | `(top_x_pct: float) -> float` | Calculates top x% share. |
| `bottom_x_pct_share` | `(bottom_x_pct: float) -> float` | Calculates bottom x% share. |
| `t10_b50` | `() -> float` | Calculates ratio between the top 10% and bottom 50% shares. |

### Ranking

| Method | Signature | Description |
|---|---|---|
| `decile_rank` | `(negatives_in_zero: Optional[bool] = False)` | Calculate decile ranks (1-10) with optional zero decile for negatives. |
| `quintile_rank` | `() -> MicroSeries` | Calculate weighted quintile ranks (1-5). |
| `quartile_rank` | `() -> MicroSeries` | Calculate weighted quartile ranks (1-4). |
| `percentile_rank` | `() -> MicroSeries` | Calculate weighted percentile ranks (1-100). |

### Variance from replicate weights

| Method | Signature | Description |
|---|---|---|
| `replicate_standard_error` | `(statistic: Callable, replicate_weights, method: str = 'jackknife', fay_k: Optional[float] = None, *, centering: str = 'full-sample') -> float` | Standard error of ``statistic`` from a set of replicate weights. |

`replicate_standard_error` accepts `method` of `jackknife`, `brr`, `bootstrap`,
`successive-difference`, or `fay` (which also requires `fay_k`). Because it
resamples rather than applying an analytic formula, it works for any statistic
the series can compute, including the Gini coefficient and quantiles.

### Weights

| Method | Signature | Description |
|---|---|---|
| `set_weights` | `(weights: ndarray, preserve_old: Optional[bool] = False) -> None` | Sets the weight values. |
| `nullify_weights` | `() -> None` | Set all weights to 1, effectively making the Series unweighted. |
| `weight` | `() -> Series` | Calculates the weighted value of the MicroSeries. |

## MicroDataFrame

### Weighted aggregation

| Method | Signature | Description |
|---|---|---|
| `sum` | `(axis: Union[int, str, NoneType] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs) -> Union[Series, MicroSeries, float]` | Sum numeric columns, weighting reductions across observations. |

```{warning}
`MicroDataFrame.cov()` and `MicroDataFrame.corr()` return pandas' **unweighted**
results; the weights are ignored. Use `MicroSeries.cov()` and `MicroSeries.corr()`
on a pair of columns for the frequency-weighted values. See
[#327](https://github.com/PolicyEngine/microdf/issues/327).
```

| Method | Signature | Description |
|---|---|---|
| `cov` | `(min_periods: Optional[int] = None, ddof: Optional[int] = 1, numeric_only: bool = False) -> DataFrame` | Pairwise covariance of the columns, **unweighted**. |
| `corr` | `(method: CorrelationMethod = 'pearson', min_periods: int = 1, numeric_only: bool = False) -> DataFrame` | Pairwise Pearson correlation of the columns, **unweighted**. |

### Weight-preserving operations

| Method | Signature | Description |
|---|---|---|
| `groupby` | `(by: Union[str, list], *args, **kwargs) -> MicroDataFrameGroupBy` | Returns a GroupBy object with MicroSeriesGroupBy objects for each column. |
| `merge` | `(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)` | Database-style join that carries the weight column through. |
| `reset_index` | `(level: Optional[int] = None, drop: Optional[bool] = False, inplace: Optional[bool] = False, col_level: Optional[int] = 0, col_fill: Optional[str] = '', allow_duplicates: Optional[bool] = None, names: Optional[list[str]] = None) -> Optional[MicroDataFrame]` | Reset the index, keeping weights aligned to their rows. |
| `drop` | `(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')` | Drop rows or columns, keeping weights aligned to the remaining rows. |
| `astype` | `(dtype, copy: Optional[bool] = True, errors: Optional[str] = 'raise') -> MicroDataFrame` | Convert MicroDataFrame to specified data type while preserving weights. |
| `copy` | `(deep: Optional[bool] = True) -> MicroDataFrame` | Copy the frame and its weights. |
| `equals` | `(other: MicroDataFrame) -> bool` | True when both the values and the weights are equal. |

### Poverty

| Method | Signature | Description |
|---|---|---|
| `poverty_rate` | `(income: str, threshold: str) -> float` | Calculate poverty rate, i.e., the population share with income below their poverty threshold. |
| `poverty_gap` | `(income: str, threshold: str) -> float` | Calculate poverty gap, i.e., the total gap between income and poverty thresholds for all people in poverty. |
| `poverty_count` | `(income: Union[MicroSeries, str], threshold: Union[MicroSeries, str]) -> int` | Calculates the number of entities with income below a poverty threshold. |
| `deep_poverty_rate` | `(income: str, threshold: str) -> float` | Calculate deep poverty rate, i.e., the population share with income below half their poverty threshold. |
| `deep_poverty_gap` | `(income: str, threshold: str) -> float` | Calculate deep poverty gap, i.e., the total gap between income and half of poverty thresholds for all people in deep poverty. |
| `squared_poverty_gap` | `(income: str, threshold: str) -> float` | Calculate squared poverty gap, i.e., the total squared gap between income and poverty thresholds for all people in poverty. Also known as the poverty severity index. |

### Weights

| Method | Signature | Description |
|---|---|---|
| `set_weights` | `(weights: Union[ndarray, str], preserve_old: Optional[bool] = False) -> None` | Sets the weights for the MicroDataFrame. |
| `set_weight_col` | `(column: str, preserve_old: Optional[bool] = False) -> None` | Sets the weights for the MicroDataFrame by specifying the name of the weight column. |
| `nullify_weights` | `() -> None` | Set all weights to 1, effectively making the DataFrame unweighted. |

## Module-level functions

| Function | Description |
|---|---|
| `microdf.replicate_variance` | Variance of a statistic from replicate weights. |
| `microdf.replicate_standard_error` | Square root of the above. |

## A note on estimator conventions

Quantiles follow the inverse cumulative distribution function, so results can be
checked against `survey::svyquantile` in R. Weighted variance treats weights as
frequency weights, so integer weights agree with `numpy` computed on the
replicated sample. Top-share cutoffs split a record that straddles the boundary
in proportion, rather than assigning it wholly to one side.
