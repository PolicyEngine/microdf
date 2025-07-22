[![Build](https://github.com/PolicyEngine/microdf/workflows/Pull%20request/badge.svg)](https://github.com/PolicyEngine/microdf/actions)
[![Codecov](https://codecov.io/gh/PolicyEngine/microdf/branch/master/graph/badge.svg)](https://codecov.io/gh/PolicyEngine/microdf)

# microdf
Weighted pandas DataFrames and Series for survey microdata analysis.

## Overview
microdf provides `MicroDataFrame` and `MicroSeries` classes that extend pandas functionality with integrated weighting support, essential for accurate survey data analysis.

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
df = pd.DataFrame({
    'income': [10_000, 20_000, 30_000, 40_000, 50_000],
    'weights': [1, 2, 3, 2, 1]
})

# Create a MicroDataFrame
mdf_df = mdf.MicroDataFrame(df, weights='weights')

# All operations are weight-aware
print(mdf_df.income.mean())  # Weighted mean
print(mdf_df.income.gini())  # Gini coefficient
```

## Questions
Contact the maintainer, Max Ghenis (max@policyengine.org).

## Citation
You may cite the source of your analysis as "microdf release #.#.#, author's calculations."
