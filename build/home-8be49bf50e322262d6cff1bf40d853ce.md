`microdf` documentation
=======================

`microdf` provides weighted data structures for survey microdata analysis in
Python. `MicroSeries` and `MicroDataFrame` carry sampling weights inside the
object, so the weights stay aligned with their rows through merges, filters,
grouping and reindexing, and the weighted estimators — quantiles, variance,
Gini, top shares, poverty rates — use documented conventions rather than ad hoc
ones.

```python
import microdf as mdf

df = mdf.MicroDataFrame({"income": [10_000, 30_000, 120_000]}, weights=[800, 1_200, 50])
df.income.median()  # 30000, weighted
df.income.gini()
```

Install with `pip install microdf-python`.

- [Examples](examples.md) — worked analyses, and how weights survive a pipeline
- [API reference](api.md) — every weighted estimator and weight-handling method
