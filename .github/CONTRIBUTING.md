# Contributing to microdf

See the [shared PolicyEngine contribution guide](https://github.com/PolicyEngine/.github/blob/main/CONTRIBUTING.md) for cross-repo conventions (towncrier changelog fragments, `uv run`, PR description format, anti-patterns). This file covers microdf specifics.

## Commands

```bash
make install         # install deps
make format          # format (required)
make test            # test suite
uv run pytest microdf/tests/path/to/test.py::test_name -v
```

Default branch: `main`.

## What lives here

microdf provides weighted-microdata primitives used throughout PolicyEngine:

- `MicroSeries` — weighted pandas Series with weighted quantiles, means, medians, ginis
- `MicroDataFrame` — weighted DataFrame of `MicroSeries`
- `generic` — weight-aware reductions, groupby helpers

Consumers: `policyengine-core`, every country package, `policyengine.py`, `policyengine-api`. API stability matters — think twice before changing public signatures.

## Repo-specific anti-patterns

- **Don't** change weighted-statistic semantics without a migration plan. Downstream analyses depend on exact numerical output.
- **Don't** introduce heavyweight dependencies — microdf is imported by every PolicyEngine Python package; bloat compounds.
- **Don't** hand-roll weighted statistics; extend the existing `MicroSeries` / `MicroDataFrame` APIs so new behaviour is discoverable.
