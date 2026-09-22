# `microdf` roadmap

`microdf` provides weighted data structures for survey microdata: weighted
quantiles and moments, inequality measures including the Gini coefficient and
top and bottom shares, poverty measures, and variance estimation from
replicate weights.

`microdf` is in maintenance. policyengine.py is taking the weighted layer into
an internal module (see its
[weighted-module design note](https://github.com/PolicyEngine/policyengine.py/pull/529));
`microdf` stays maintained for policyengine-core, the country packages and the
analysis repositories that depend on it, and new estimators land in
policyengine.py.

Planned work:

* Fail closed: any operation that cannot carry weights raises instead of
  returning an unweighted result or a Micro object whose weights were reset
  ([#333](https://github.com/PolicyEngine/microdf/issues/333),
  [#264](https://github.com/PolicyEngine/microdf/issues/264))
* Exact definitions and tests for the poverty measures
  ([#334](https://github.com/PolicyEngine/microdf/issues/334))
* Documentation that matches the code
  ([#335](https://github.com/PolicyEngine/microdf/issues/335))

Not planned: wider coverage of pandas methods that change shape, which the
design note above explains; variance from stratum and cluster identifiers, for
which `svy` and R's `survey` exist; and dataset presets.

See the [issues page](https://github.com/PolicyEngine/microdf/issues) to view
and suggest other items.
