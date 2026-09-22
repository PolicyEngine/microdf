# Supported operations

Generated from `microdf/tests/test_fail_closed.py` by `uv run python docs/build_support.py`.

The regression suite checks these contracts on pandas 2 and 3. Aggregated results carry no row weights; transformations retain independent copies of the input weights.

| Operation | Behaviour |
|---|---|
| groupby column mean | Weighted result / preserved weights |
| groupby dict agg | Weighted result / preserved weights |
| groupby named agg | Weighted result / preserved weights |
| groupby callable agg | Weighted result / preserved weights |
| SeriesGroupBy callable agg | Weighted result / preserved weights |
| callable pivot_table | Weighted result / preserved weights |
| row apply | Weighted result / preserved weights |
| frame numeric mean | Weighted result / preserved weights |
| frame numeric median | Weighted result / preserved weights |
| groupby numeric mean | Weighted result / preserved weights |
| groupby numeric median | Weighted result / preserved weights |
| frame constructor | Weighted result / preserved weights |
| series constructor | Weighted result / preserved weights |
| pd.cut preserves weights | Weighted result / preserved weights |
| pd.qcut preserves weights | Weighted result / preserved weights |
| pd.to_numeric preserves weights | Weighted result / preserved weights |
| series explode | Weighted result / preserved weights |
| np.average | Weighted result / preserved weights |
| np.mean | Weighted result / preserved weights |
| np.median | Weighted result / preserved weights |
| weighted value_counts | Weighted result / preserved weights |
| weighted mode | Weighted result / preserved weights |
| numeric frame dropna | Weighted result / preserved weights |
| `rolling` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `expanding` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `ewm` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `sem` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `skew` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `kurt` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `kurtosis` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `prod` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `product` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `idxmax` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| `idxmin` | Raises; use `pd.Series(s)` for unweighted pandas behaviour |
| Arithmetic with conflicting weights | Raises `ValueError` in either operand order |

`pd.cut` and `pd.qcut` retain row weights, but choose bin edges using pandas' unweighted rules. Supply explicit bin edges for weighted quantile bins.

`pivot_table` grouping keys must name columns; external Series, callable
groupers and index-level groupers raise. Weighted `Series.value_counts` and
`Series.mode` return plain summary Series; frame and grouped variants raise.

`microdf.concat` rejects mixed weighted/plain inputs in either order. Direct
`pd.concat` still bypasses microdf when its first input is plain pandas; the
regression suite records this upstream dispatch limitation as an expected failure.
Use Micro objects for every input to `pd.concat`, or use `microdf.concat`.
