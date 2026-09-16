Examples
========

See these rendered Jupyter notebooks for examples of `microdf` usage.
## Keeping weights through pandas operations

`MicroSeries` and `MicroDataFrame` retain independent copies of their row
weights through `fillna`, `replace`, `astype`, `reset_index`, sorting,
sampling, and row or column selections. Positional operations move the weights with each row,
including duplicate index labels and sampling with replacement.

```python
import pandas as pd
from microdf import MicroDataFrame

survey = MicroDataFrame(
    {"income": [30.0, None, 10.0]},
    index=[7, 7, 3],
    weights=[2, 5, 11],
)
filled = survey.fillna(4)
ordered = filled.sort_values("income")
assert ordered.weights.tolist() == [5, 11, 2]
assert ordered["income"].sum() == 190

sample = filled.sample(n=5, replace=True, random_state=17)
combined = pd.concat([filled.iloc[:2], filled.iloc[2:]], ignore_index=True)
assert combined.weights.tolist() == [2, 5, 11]
```

Row concatenation combines the inputs' weights, including a mix of
`MicroDataFrame` and `MicroSeries` inputs in either order. Constructing a
`MicroDataFrame` from a `MicroSeries` or a one-column mapping containing one
also retains its weights unless explicit weights are supplied. Column concatenation aligns
weights to the output rows and requires matching weights wherever inputs
share a row. Conflicting weights raise `ValueError`. New rows introduced by
`reindex`, an ambiguous row alignment, and DataFrame transposition also raise
`ValueError`: these operations need an explicit choice of result weights.
Convert to a plain pandas object, perform the operation, then construct a
new Micro object with appropriate weights when that choice is intentional.
A single DataFrame row is a plain pandas `Series`, because its entries are
columns rather than weighted observations.

`MicroDataFrame.cov()` and `.corr()` retain pandas' unweighted calculations
and return plain pandas `DataFrame` matrices. Their rows describe columns,
so observation weights do not apply to the result or subsequent operations
such as `.sum()`. These methods accept the installed pandas version's
arguments and defaults, including missing-value handling and correlation
methods.

Use Micro objects for **every input** to `pd.concat`. A mixed concat raises
`ValueError` when pandas calls the Micro object's hooks. If a plain pandas
object comes first, pandas can bypass those hooks and return an unweighted
object; microdf cannot intercept that dispatch. Convert each input to
`MicroSeries` or `MicroDataFrame` with meaningful weights before concatenating.
These guarantees cover the operations above; they do not establish weighted
semantics for every pandas operation.
