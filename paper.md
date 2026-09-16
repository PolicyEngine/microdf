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
    affiliation: '1'
affiliations:
  - name: PolicyEngine, Washington, DC, United States
    index: '1'
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`microdf` provides weighted data structures for survey microdata analysis in Python. Survey records carry sampling weights: each row stands for many households, and the weights vary by orders of magnitude within a single file. Statistics computed without them describe the sample rather than the population, and the two can differ substantially.

The package's central design choice is that the weight is a property of the data structure rather than an argument to a function. `MicroSeries` and `MicroDataFrame` subclass the pandas [@mckinney2010pandas; @pandas2020] structures and carry a weight vector through the operations an analysis pipeline performs. Selection, merging, grouping, reindexing, dropping and type conversion are overridden so that the weight follows the rows it describes, and aggregations that pandas defines but that would silently ignore weights are overridden rather than inherited, so a method either returns a weighted result or warns that it cannot.

On top of that foundation the package implements the estimators distributional analysis reports: quantiles by the inverse cumulative distribution function, frequency-weighted variance, the Gini coefficient from the Lorenz curve, top and bottom shares with proportional handling of records tied at the cutoff, and the Foster-Greer-Thorbecke family of poverty measures [@foster1984fgt].

# Statement of Need

Analysts working with survey microdata in Python face two distinct problems, and the second is the one that causes silent errors.

The first is that the estimators themselves require care. A weighted median is not the median of the weighted values. A weighted variance requires deciding whether weights are frequencies or precision weights, and the two give different answers. A top-1% share requires deciding what happens to the records straddling the cutoff: assigning them wholly to one side introduces a bias that grows as weights grow coarser. These are solvable problems, but each is a decision that hand-written code makes implicitly, usually without recording it, so implementations diverge between analysts on exactly the edge cases that matter.

The second problem is that weights must stay aligned with the data through every transformation before the estimator runs. Building an analysis dataset means merging administrative variables onto survey records, filtering to a subpopulation, grouping by geography, reindexing after a sort. Each of these is an opportunity for the weight vector to fall out of alignment with the rows it describes, and nothing raises when it does: the pipeline completes and returns a number that is wrong by an amount nobody can see. In our experience maintaining microsimulation datasets, this is a more frequent source of error than the estimator formulas, and a harder one to detect, because the result remains plausible.

`microdf` addresses both. It makes the estimator decisions once, documents them, and tests them: the quantile estimator follows the inverse CDF definition, matching the default behaviour of R's `survey::svyquantile` [@lumley2004survey; @lumley2010complex] so that results can be checked against an established implementation; the variance treats weights as frequencies, so that with integer weights it agrees with `numpy` on the replicated sample; the top-share estimator splits the record at the cutoff proportionally, so a constant distribution returns the share it should. And it carries the weight with the data across the transformations between loading a file and computing a statistic, which is what makes those estimator guarantees worth anything in a real pipeline.

# State of the Field

Several tools compute weighted statistics, but the combination `microdf` occupies — pandas-native structures, a distributional estimator set, and no complex-survey design object — is not otherwise filled.

| Tool | Weighted quantiles | Inequality and poverty measures | pandas-native | Design-based variance |
|---|---|---|---|---|
| `microdf` | Yes | Gini, top and bottom shares, FGT poverty | Yes | No |
| `samplics` [@samplics] | Yes | No | Partly | Yes |
| `statsmodels` [@seabold2010statsmodels] | Limited | No | Partly | Partly |
| R `survey` [@lumley2004survey] | Yes | Limited | No (R) | Yes |
| pandas + manual weighting | Hand-written | Hand-written | Yes | No |

R's `survey` package is the reference implementation for design-based survey inference and remains the right tool when standard errors under a complex design are required. `samplics` brings much of that machinery to Python, again centred on sampling design. Neither is built around the inequality and poverty estimators that distributional policy analysis reports, and neither returns objects that behave like a `DataFrame` in an existing pandas pipeline.

`microdf` deliberately does not implement design-based variance estimation. Its weights are population weights: it estimates the statistic, not the sampling error around it. Analysts needing standard errors under a stratified or clustered design should use `survey` or `samplics`. The trade is a much smaller interface, and statistics that compose with the pandas code analysts already have.

# Software Design

`MicroSeries` extends `pandas.Series` with a weight vector of equal length; `MicroDataFrame` extends `pandas.DataFrame`, holds a weight column, and exposes each column as a `MicroSeries`.

Pandas methods are classified into three groups. *Scalar* methods return a single weighted statistic and are overridden to use the weights. *Vector* methods return a series aligned to the input and carry the weights through to the result. *Agnostic* methods do not depend on weighting and are inherited unchanged. Methods that would need weighting but do not yet implement it, currently `cov` and `corr`, fall through to pandas and emit a warning rather than returning an unweighted number silently. Shape-changing operations — selection, `merge`, `groupby`, `reset_index`, `drop`, `astype` — are overridden so the weight vector follows the rows it describes.

The classification is explicit rather than inherited, which is a deliberate trade: a method must be considered before it is supported, and one that has not been is not silently assumed safe. Extending the set of preserved operations is the package's main axis of ongoing work.

```python
import microdf as mdf

df = mdf.MicroDataFrame(
    {"income": [10_000, 30_000, 120_000], "threshold": [15_000, 15_000, 15_000]},
    weights=[800, 1_200, 50],
)

df.income.median()            # 30000.0, weighted
df.income.gini()              # Lorenz-curve Gini over the weighted distribution
df.income.top_10_pct_share()  # proportional split at the cutoff
df.poverty_rate("income", "threshold")

regional = df.merge(geography, on="household_id")  # weights follow the join
regional.groupby("region").income.median()         # and the grouping
```

Statistics requiring a decision take it as an explicit argument rather than choosing silently: `gini` accepts a `negatives` policy, `var` takes `ddof`, and behaviour on zero-weight records is documented and tested. The estimators are small and independently checkable: weighted quantiles sort by value, accumulate weight, and return the smallest value whose cumulative weight share reaches `q`, after dropping zero-weight records so they cannot be selected; the Gini is computed from the Lorenz curve over weighted cumulative population and income; poverty measures follow the FGT family, with rate, gap, deep gap, and squared gap.

# Research Impact Statement

`microdf` has been public since June 2018, with 749 commits across 16 contributors and 11 releases. It is a dependency of both `policyengine-us` and `policyengine-uk`, and therefore sits in the computational path of PolicyEngine's published distributional estimates — the poverty rates, decile impacts, and Gini changes reported in its analyses and through its web application [@policyengine_py]. It is also used directly in standalone policy studies, including analyses of free school meals, extended childcare entitlements, and national insurance reforms.

The package's role is that of infrastructure: it is not the visible output of an analysis, but the layer that determines whether a reported poverty rate is a population estimate or a sample artefact. Its adoption is best measured by the analyses that depend on it rather than by direct use.

# Acknowledgements

Arnold Ventures [@arnold_ventures], NEO Philanthropy [@neo_philanthropy], the Gerald Huff Fund for Humanity, and the National Science Foundation (NSF POSE Phase I, Award 2518372) [@nsf_pose] funded this work in the US. The Nuffield Foundation has funded the UK work since September 2024 [@nuffield2024grant]. These funders had no involvement in the design, development, or content of this software or paper. All authors are employed by PolicyEngine and may benefit reputationally from the software's adoption; this relationship is disclosed here as a potential conflict of interest.

We thank all contributors to `microdf`, and Thomas Lumley, whose `survey` package provides the reference against which several of the estimators here are checked.

# AI Usage Disclosure

The authors used generative AI tools, specifically Claude Opus by Anthropic [@claude2026], to assist with code refactoring, test authoring, and drafting of this paper. Human authors reviewed, edited, and validated all AI-assisted outputs, and made all decisions regarding estimator definitions and software design. The authors remain fully responsible for the accuracy, originality, and correctness of all submitted materials.

# References
