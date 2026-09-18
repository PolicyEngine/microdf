[![Build](https://github.com/PolicyEngine/microdf/workflows/Pull%20request/badge.svg)](https://github.com/PolicyEngine/microdf/actions)
[![Codecov](https://codecov.io/gh/PolicyEngine/microdf/branch/main/graph/badge.svg)](https://codecov.io/gh/PolicyEngine/microdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22829459.svg)](https://doi.org/10.5281/zenodo.22829459)

# microdf
Weighted pandas DataFrames and Series for survey microdata analysis.

## Why this exists

Survey microdata comes with weights, and analysing it in pandas means getting two
things right that pandas will not do for you.

**The estimators are not the obvious ones.** A weighted median is not the median of
weighted values. Weighted variance requires choosing between treating weights as
frequencies or as precision. A top-1% share requires deciding what happens to a
record that straddles the cutoff. Each of these is a decision, and hand-rolling it
per analysis means making it differently each time. `microdf` makes each choice
once, documents it, and tests it — quantiles follow the inverse CDF so they can be
checked against R's `survey::svyquantile`, and variance treats weights as
frequencies so integer weights agree with `numpy` on the replicated sample.

**Weights have to survive the pipeline.** Before any estimator runs, weights must
stay aligned with their rows through merges, filters, grouping and reindexing. When
they do not, nothing raises. The pipeline completes and returns a plausible wrong
number. This is the harder of the two problems, and it is why `microdf` carries
weights inside the object rather than beside it.

If you are computing a poverty rate or a Gini on weighted survey data, those are
the two ways to get a believable-looking wrong answer.

## Key Features
- **MicroDataFrame**: A pandas DataFrame with an integrated weight column
- **MicroSeries**: A pandas Series with integrated weights
- **Weighted operations**: All aggregations (sum, mean, median, etc.) automatically use weights
- **Inequality metrics**: Built-in Gini coefficient calculation
- **Poverty analysis**: Integrated poverty rate and gap calculations

## Installation
Install with:

    pip install microdf-python

Or for development:

    pip install git+https://github.com/PolicyEngine/microdf.git

## Usage
```python
import microdf as mdf
import pandas as pd

# Create sample data with weights
df = pd.DataFrame(
    {"income": [10_000, 20_000, 30_000, 40_000, 50_000], "weights": [1, 2, 3, 2, 1]}
)

# Create a MicroDataFrame
mdf_df = mdf.MicroDataFrame(df, weights="weights")

# All operations are weight-aware
print(mdf_df.income.mean())  # Weighted mean
print(mdf_df.income.gini())  # Gini coefficient
```

## Questions
Contact the maintainer, Max Ghenis (max@policyengine.org).

## Citation
You may cite the source of your analysis as "microdf release #.#.#, author's calculations."
