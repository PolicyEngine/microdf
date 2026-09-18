---
title: "microdf: Weighted DataFrames and Series for Survey Microdata Analysis"
tags:
  - Python
  - survey statistics
  - microdata
  - inequality
  - poverty
  - pandas
authors:
  - name: Vahid Ahmadi
    orcid: 0009-0004-1093-6272
    affiliation: '1'
    corresponding: true
  - name: Max Ghenis
    orcid: 0000-0002-1335-8277
    affiliation: '1'
  - name: María Juaristi
    orcid: 0009-0007-4946-2248
    affiliation: '1'
  - name: Nikhil Woodruff
    orcid: 0009-0009-5004-4910
    affiliation: '1'
affiliations:
  - name: PolicyEngine, Washington, DC, United States
    index: '1'
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`microdf` provides weighted data structures for survey microdata analysis in Python. Survey records carry sampling weights: each row stands for many households, and the weights vary by orders of magnitude within a single file. Statistics that ignore the weights describe the sample rather than the population the sample was drawn to represent.

The package's central design choice is to store the weight on the data structure itself. `MicroSeries` and `MicroDataFrame` subclass the pandas [@mckinney2010pandas; @pandas2020] structures and carry a weight vector through the operations an analysis pipeline performs. Selection, merging, grouping, reindexing, dropping and type conversion are overridden so that the weight follows the rows it describes, and the aggregations pandas defines are overridden to use it.

On that foundation the package implements the estimators that distributional analysis reports: quantiles by the inverse cumulative distribution function, frequency-weighted variance, the Gini coefficient from the Lorenz curve, top and bottom shares with proportional handling of records tied at the cutoff, and the Foster-Greer-Thorbecke poverty measures [@foster1984fgt], reported as a headcount rate and as aggregate poverty and squared-poverty gaps.

# Statement of need

Analysts working with survey microdata in Python face two problems, and the second causes silent errors. The first is in the estimators. A weighted median is not the median of the weighted values. A weighted variance requires deciding whether weights are frequencies or precision weights, and the two give different answers. A top-1% share requires deciding what happens to the records straddling the cutoff: assigning them wholly to one side introduces a bias that grows as weights grow coarser. Each is a decision that hand-written code makes implicitly and rarely records, so implementations diverge on exactly the edge cases that matter.

The second problem is that weights must stay aligned with the data through every transformation before the estimator runs. Building an analysis dataset means merging administrative variables onto survey records, filtering to a subpopulation, grouping by geography, reindexing after a sort. Each of these can leave the weight vector misaligned with the rows it describes, and nothing raises when it does: the pipeline completes and returns a plausible wrong number. In our experience maintaining microsimulation datasets, this is a more frequent source of error than the estimator formulas, and a harder one to detect.

`microdf` addresses both. It makes the estimator decisions once, documents them and tests them: the quantile estimator follows the inverse CDF definition and matches the default behaviour of R's `survey::svyquantile` [@lumley2004survey; @lumley2010complex], so results can be checked against an established implementation; the variance treats weights as frequencies, so with integer weights it agrees with `numpy` on the replicated sample; the top-share estimator splits the record at the cutoff proportionally, so a constant distribution returns the share it should. It also carries the weight with the data through every transformation between loading a file and computing a statistic, so those guarantees hold at the point the statistic is taken.

# State of the field

Several tools compute weighted statistics. `microdf` combines pandas-native structures, a distributional estimator set, and replicate-weight variance without requiring a complex-survey design object.

|  | `microdf` | `samplics` [@samplics] | `statsmodels` [@seabold2010statsmodels] | R `survey` [@lumley2004survey] | pandas, weighted by hand |
|---|---|---|---|---|---|
| Weighted quantiles | Yes | Yes | `DescrStatsW` only | Yes | Hand-written |
| Inequality and poverty measures | Gini, top and bottom shares, FGT poverty | No | No | Limited | Hand-written |
| pandas-native | Yes | Partly | Partly | No (R) | Yes |
| Design-based variance | Replicate weights | Yes | No | Yes | No |

R's `survey` package is the reference implementation for design-based survey inference and remains the right tool when standard errors under a complex design are required. `samplics` brings much of that machinery to Python, also centred on sampling design. Neither implements the inequality and poverty estimators that distributional policy analysis reports, and neither returns objects that behave like a `DataFrame` in an existing pandas pipeline.

`microdf` estimates variance from replicate weights. Given the replicate weight matrix that products such as the CPS and ACS publish, it recomputes a statistic once per replicate and scales the spread by the factor for the replication scheme. This needs no analytic formula, so it works for the Gini coefficient and quantiles as readily as for a mean. It does not derive variance from stratum and cluster identifiers, so analysts who need standard errors under a design specification, or who hold no replicate weights, should use `survey` or `samplics`. The replicate estimators also assume weights as published: weights calibrated to external targets no longer correspond to the original replication scheme.

# Software design

`MicroSeries` extends `pandas.Series` with a weight vector of equal length; `MicroDataFrame` extends `pandas.DataFrame`, holds a weight column, and exposes each column as a `MicroSeries`.

Pandas methods fall into three groups. *Scalar* methods return a single weighted statistic and are overridden to use the weights. *Vector* methods return a series aligned to the input and carry the weights through to the result. *Agnostic* methods return either a scalar or a vector depending on their arguments, and are dispatched accordingly. `cov` and `corr` are frequency-weighted on both classes; each cell of the frame matrix is the estimator applied to that pair of columns. `values` and `to_numpy` hand back plain data by design and warn that the result is unweighted. Shape-changing operations are overridden so the weight vector follows the rows it describes: selection, `merge`, `groupby`, `reset_index`, `drop` and `astype`.

The classification is a deliberate step: each method is supported only after it has been considered.

```python
import microdf as mdf

df = mdf.MicroDataFrame(
    {
        "household_id": [1, 2, 3],
        "income": [10_000, 30_000, 120_000],
        "threshold": [15_000, 15_000, 15_000],
    },
    weights=[800, 1_200, 50],
)

df.income.median()  # 30000, weighted
df.income.gini()  # Lorenz-curve Gini over the weighted distribution
df.income.top_10_pct_share()  # proportional split at the cutoff
df.poverty_rate("income", "threshold")

# Weights follow a join and a grouping.
regional = df.merge(geography, on="household_id")
regional.groupby("region").income.median()
```

Standard errors come from replicate weights, so they are available for every estimator, including those with no tractable variance formula. The scale applied to the spread across replicates depends on how the replicates were constructed [@wolter2007variance], including the Fay-type replication with a 4/R scale used for the ACS and the CPS ASEC [@fay1995successive], which publish 80 and 160 replicates respectively:

```python
series.replicate_standard_error(
    lambda s: s.gini(), replicate_weights, method="successive-difference"
)
```

Statistics that require a decision take it as an explicit argument: `gini` accepts a `negatives` policy, `var` takes `ddof`, and behaviour on zero-weight records is documented and tested. The estimators are small and independently checkable: weighted quantiles sort by value, accumulate weight, and return the smallest value whose cumulative weight share reaches `q`, after dropping zero-weight records so they cannot be selected; the Gini is computed from the Lorenz curve over weighted cumulative population and income; poverty measures follow Foster, Greer and Thorbecke in form and are reported as a headcount rate and as aggregate gaps in currency units, with deep variants at half the threshold.

# Research impact statement

`microdf` is part of the foundation PolicyEngine's microsimulation stack is built on. `policyengine` [@policyengine_py] depends on it, so the poverty rates, decile impacts and Gini changes published through PolicyEngine's analyses and at [policyengine.org](https://policyengine.org) are computed through its estimators. It is also used directly in public policy reform analysis in the United Kingdom and the United States. Public since June 2018, it has over 800 commits from eight contributors and averages around 1,700 downloads a day.

# Acknowledgements

Arnold Ventures [@arnold_ventures], NEO Philanthropy [@neo_philanthropy], the Gerald Huff Fund for Humanity, and the National Science Foundation (NSF POSE Phase I, Award 2518372) [@nsf_pose] funded this work in the US. The Nuffield Foundation has funded the UK work since September 2024 [@nuffield2024grant]. These funders had no involvement in the design, development, or content of this software or paper. All authors are employed by PolicyEngine and may benefit reputationally from the software's adoption; this relationship is disclosed here as a potential conflict of interest.

Max Ghenis created `microdf` in 2018 and wrote most of the estimators and the weight-preserving class machinery; María Juaristi contributed to the estimators, the test suite and the release infrastructure, including the weight preservation described above; Nikhil Woodruff to the pandas integration and the packaging; and Vahid Ahmadi contributed the replicate-weight variance estimation and prepared this paper. We thank Anthony Volk and Jason DeBacker for their contributions to the package, and all other contributors to `microdf`. We also thank Thomas Lumley, whose `survey` package provides the reference against which several of the estimators here are checked.

# AI usage disclosure

The authors used generative AI tools, specifically Claude by Anthropic [@claude2026], to assist with code refactoring, test authoring, and drafting of this paper. Human authors reviewed, edited, and validated all AI-assisted outputs, and made all decisions regarding estimator definitions and software design. The authors remain fully responsible for the accuracy, originality, and correctness of all submitted materials.

# References
