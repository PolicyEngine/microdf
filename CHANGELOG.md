## [1.3.0] - 2026-04-17

### Added

- Added weighted `MicroSeries.std()` and `MicroSeries.var()` implementations that treat the weights as frequency counts (matching `numpy.std`/`numpy.var` on the replicated sample). `MicroSeries.cov()` and `MicroSeries.corr()` still fall through to the unweighted pandas implementations but now emit a `UserWarning` so callers aren't silently given an unweighted result. `MicroDataFrame.std()` / `.var()` pick up the weighted implementations via the scalar-function override.

### Changed

- Point CONTRIBUTING.md at the shared PolicyEngine contribution guide (https://github.com/PolicyEngine/.github) and trim the per-repo file to commands, repo-specific conventions, and anti-patterns. Removes the stale `changelog_entry.yaml` / `make changelog` instructions.
- Changed `MicroSeries.rank()` to use max-rank semantics for tied values — every row in a tie group shares the cumulative weight at the end of the group. This makes `decile_rank`, `quintile_rank`, `quartile_rank`, and `percentile_rank` place tied rows in the same bucket (e.g. a constant series now lands entirely in decile 10 rather than being split across deciles). The method now has a docstring explaining the semantic. Unique-valued series produce the same cumulative-weight values as before.

### Fixed

- Fixed `MicroSeries.count()` silently including NaN-row weights, contrary to pandas semantics. `count()` now skips NaN by default (matching `pandas.Series.count`) and accepts `skipna=False` to recover the old behaviour.
- Fixed `MicroDataFrame.drop` leaving stale weights when rows were dropped. Previously `drop(index=..., inplace=True)` copied the weights Series before dropping and then reassigned the full-length copy, leaving `self.weights` out of sync with `self` and causing subsequent weighted ops to raise a length-mismatch `ValueError`. `reset_index(inplace=True)` had the same pattern and is also fixed.
- Made `MicroSeries.gini(negatives=...)` actually apply its option (previously the branches sorted `self` instead of the mutated local, so `negatives='zero'` and `negatives='shift'` were silently ignored). `gini()` now warns on negative inputs with `negatives=None`, short-circuits all-zero and empty series to 0 (instead of dividing by zero and returning NaN with a `RuntimeWarning`), and raises on an unknown `negatives` option.
- Fixed `MicroDataFrame.groupby` leaking a `__tmp_weights` column onto the caller. Previously, calling `df.groupby(...)` permanently added the weight column to `df.columns`, so any subsequent `df.sum()` or iteration included it. The implementation now stages the weights on a copy before calling `super().groupby`, leaving `self` untouched.
- Fixed `MicroDataFrame.merge` raising `ValueError` on any row-changing join. The implementation now attaches the left-side weights as a temporary column before the merge so pandas propagates them onto each surviving output row (handling inner filtering, left-with-missing, many-to-many duplication, and outer joins). Right-only outer rows default to a 0 weight.
- Fixed `MicroSeries.quantile` returning values with weight 0. Previously, when the first (or an internal) sorted element had zero weight, the inverse-CDF search still picked it (e.g. `MicroSeries([10, 20, 30], weights=[0, 1, 1]).quantile(0)` returned 10 instead of 20). Zero-weight rows are now dropped before computing the CDF; an all-zero-weight series returns NaN.
- Fixed `MicroSeries.top_x_pct_share` overstating the top share when rows tied at the threshold (e.g. a constant series always returned 1.0 regardless of `top_x_pct`), and `top_x_pct_share(0)` returning the max bucket's share instead of 0. The implementation now sorts by value, cumulates weight, and splits the tied-at-cutoff row proportionally — matching the standard wealth-share algorithm. Downstream `bottom_x_pct_share`, `top_50/10/1/0.1_pct_share`, `bottom_50_pct_share`, and `t10_b50` inherit the fix.
- Aligned weights Series to `self.index` in `MicroSeries.set_weights` and `MicroDataFrame.set_weights` so weighted operations (`.sum()`, `.weight()`, `.top_x_pct_share()`, `.gini()`, etc.) return correct values when the data uses a non-default index. Previously they silently returned `0.0`.

### Removed

- Removed the `MicroSeries.weighted_function` decorator that wrapped `scalar_function` / `vector_function` at class-body execution (not the decorated methods), so the `ZeroDivisionError -> np.NaN` fallback was never reached at runtime. It also referenced `np.NaN`, which was removed in numpy 2.0, so any legitimate trigger would have raised `AttributeError`.


## [1.2.4] - 2026-03-10

No significant changes.


## [1.2.3] - 2026-03-06

### Changed

- Replaced black, isort, flake8, and linecheck with ruff for code formatting.


## [1.2.2] - 2026-02-24

### Changed

- Migrated from changelog_entry.yaml to towncrier fragments to eliminate merge conflicts.


Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.2.1] - 2026-01-25 13:40:28

### Fixed

- Fixed pandas 3.0 compatibility issues with MicroSeries method access and Copy-on-Write behavior

## [1.2.0] - 2026-01-24 15:42:22

### Added

- Added pandas 3.0 compatibility test suite

### Fixed

- MicroDataFrame.loc[] and .iloc[] now preserve MicroDataFrame type and weights when filtering rows (fixes issue
- MicroDataFrame.groupby(col)["y"].sum() and groupby(col)[["y"]].sum() now use weighted aggregation (fixes issue
- Documentation build updated to use Jupyter Book 2.0 / MyST

## [1.1.2] - 2026-01-07 12:05:33

### Fixed

- Fix mean() method to properly handle skipna parameter

## [1.1.1] - 2025-12-01 14:13:25

### Fixed

- Fix weighted quantile/median to use inverse CDF method instead of interpolation

## [1.1.0] - 2025-11-26 01:45:27

### Fixed

- MicroDataFrame.loc[] and .iloc[] now preserve MicroDataFrame type and weights when filtering rows (fixes issue
- MicroDataFrame.groupby(col)["y"].sum() and groupby(col)[["y"]].sum() now use weighted aggregation (fixes issue
- Documentation build updated to use Jupyter Book 2.0 / MyST

## [1.0.2] - 2025-07-24 12:20:41

### Added

- __getattr__ method to MicroDataFrame for intuitive column access via dot notation (Fixes
- Full pandas argument support to drop() and merge() methods (Addresses
- nullify_weights() method to both MicroDataFrame and MicroSeries to set all weights to 1 (Fixes
- Test coverage for set_weights() with string column name argument

### Fixed

- MicroDataFrame.merge() now works correctly by implementing inplace support for the drop() method
- merge() now returns a MicroDataFrame instead of a regular DataFrame (Fixes
- MicroDataFrame aggregation functions now skip non-numeric columns instead of raising errors (Fixes

## [1.0.1] - 2025-07-24 02:03:31

### Fixed

- Allowed a MicroDataFrame to handle an empty index subset

## [1.0.0] - 2025-07-22 19:04:55

### Changed

- Update package description to reflect focused scope on weighted DataFrames and Series.
- Update README with clearer documentation and usage examples.
- Version bumped to 1.0.0 to reflect major breaking changes.
- Remove pip from dev dependencies as it's not needed.
- Update to Python 3.13 as main version, supporting Python 3.9+.
- Test against Python 3.9, 3.10, 3.11, 3.12, and 3.13 in CI.

## [0.6.0] - 2025-07-22 16:16:51

## [0.5.0] - 2025-07-22 15:08:32

### Added

- Add astype and sqrt methods to MicroSeries and MicroDataFrame.
- Support in-place reset_index.
- Split generic.py into microdataframe.py and microseries.py.
- Add optional preserve_old parameter when setting weights.

## [0.4.7] - 2025-07-18 12:53:43

### Changed

- Deleted visualization functionality.

## [0.4.6] - 2025-07-18 10:50:26

### Added

- Create pyproject.toml and uv.lock files to move away from setup.py.
- Create unified workflow files.
- Fix documentation errors.
- Ensure MicroSeries functions return MicroSeries objects.
- Fix unary operations to not require `other`.

## [0.4.5] - 2025-07-17 12:00:00

### Added

- Initialized changelog.



[1.2.1]: https://github.com/PolicyEngine/microcalibrate/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/PolicyEngine/microcalibrate/compare/1.1.2...1.2.0
[1.1.2]: https://github.com/PolicyEngine/microcalibrate/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/PolicyEngine/microcalibrate/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/PolicyEngine/microcalibrate/compare/1.0.2...1.1.0
[1.0.2]: https://github.com/PolicyEngine/microcalibrate/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/PolicyEngine/microcalibrate/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.6.0...1.0.0
[0.6.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.7...0.5.0
[0.4.7]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.6...0.4.7
[0.4.6]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.5...0.4.6

