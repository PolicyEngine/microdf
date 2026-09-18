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
| `quintile_rank` | `() -> 'MicroSeries'` |  |
| `quartile_rank` | `() -> 'MicroSeries'` |  |
| `percentile_rank` | `() -> 'MicroSeries'` |  |

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
| `set_weights` | `(weights: <built-in function array>, preserve_old: Optional[bool] = False) -> None` | Sets the weight values. |
| `nullify_weights` | `() -> None` | Set all weights to 1, effectively making the Series unweighted. |
| `weight` | `() -> pandas.core.series.Series` | Calculates the weighted value of the MicroSeries. |

## MicroDataFrame

### Poverty

| Method | Signature | Description |
|---|---|---|
| `poverty_rate` | `(income: str, threshold: str) -> float` | Calculate poverty rate, i.e., the population share with income below their poverty threshold. |
| `poverty_gap` | `(income: str, threshold: str) -> float` | Calculate poverty gap, i.e., the total gap between income and poverty thresholds for all people in poverty. |
| `poverty_count` | `(income: Union[microdf.microseries.MicroSeries, str], threshold: Union[microdf.microseries.MicroSeries, str]) -> int` | Calculates the number of entities with income below a poverty threshold. |
| `deep_poverty_rate` | `(income: str, threshold: str) -> float` | Calculate deep poverty rate, i.e., the population share with income below half their poverty threshold. |
| `deep_poverty_gap` | `(income: str, threshold: str) -> float` | Calculate deep poverty gap, i.e., the total gap between income and half of poverty thresholds for all people in deep poverty. |
| `squared_poverty_gap` | `(income: str, threshold: str) -> float` | Calculate squared poverty gap, i.e., the total squared gap between income and poverty thresholds for all people in poverty. Also known as the poverty severity index. |

### Weights

| Method | Signature | Description |
|---|---|---|
| `set_weights` | `(weights: Union[numpy.ndarray, str], preserve_old: Optional[bool] = False) -> None` | Sets the weights for the MicroDataFrame. |
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
