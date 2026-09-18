# `microdf` roadmap

`microdf` provides weighted data structures for survey microdata: weighted
quantiles and moments, inequality measures including the Gini coefficient and
top and bottom shares, Foster-Greer-Thorbecke poverty measures, and variance
estimation from replicate weights.

Planned work:

* Variance from stratum and cluster identifiers, for designs where replicate
  weights are not published
* Presets for common datasets, such as suggesting the appropriate weight
  variable for the SCF and the CPS
* Wider coverage of pandas methods that change shape, so fewer operations need
  an explicit override

See the [issues page](https://github.com/PolicyEngine/microdf/issues) to view
and suggest other items.
