## [1.5.8] - 2026-09-18

### Changed

- Releases are now created automatically from the tag, so Zenodo archives every version rather than only those released by hand.


## [1.5.7] - 2026-09-18

### Fixed

- `MicroDataFrame.cov()` and `.corr()` now return frequency-weighted pairwise matrices using the same estimator as `MicroSeries.cov` and `.corr`, instead of unweighted pandas results. As on `MicroSeries`, `corr` accepts only `method="pearson"`.


## [1.5.6] - 2026-09-18

### Changed

- Adds a Zenodo DOI badge to the README.


## [1.5.5] - 2026-09-18

### Changed

- Adds .zenodo.json so the Zenodo archive records the paper's author list and title.


## [1.5.4] - 2026-09-18

### Changed

- The README now opens with the problem the package solves rather than a description of its classes.


## [1.5.3] - 2026-09-18

### Fixed

- Release tagging now works: the tag script reads the version from pyproject.toml rather than a helper that never existed, and fails loudly instead of silently.


## [1.5.2] - 2026-09-17

### Fixed

- Preserve aligned, independent MicroSeries weights in binary NumPy operations such as maximum and in both results of divmod with a pandas Series on the left, including interoperability with higher-priority Series subclasses that inherit pandas' NumPy handling. Reject unknown or ambiguous row weights before writing explicit output buffers.


## [1.5.1] - 2026-09-17

### Fixed

- Preserve label-aligned calling Series weights in binary operators and named arithmetic and comparison methods across pandas versions. Keep DataFrame reset-index weights independently mutable.


## [1.5.0] - 2026-09-17

### Added

- Added variance and standard error estimation from replicate weights for scalar statistics, preserving input dtypes and supporting common-factor jackknife, BRR, Fay's BRR, bootstrap and successive-difference schemes with explicit full-sample or replicate-mean centering. Reference-relative centering and scaled accumulation preserve representable variances at extreme magnitudes. Full-sample callbacks receive independent copies so in-place transformations preserve caller data and subsequent replicate inputs. Statistical validity depends on the statistic and survey design; nonsmooth quantiles can require an appropriate replication method or smoothing.


## [1.4.1] - 2026-09-16

### Fixed

- Preserve independent, aligned weights through pandas sorting, sampling, row and column selection, index resetting, fillna, and concatenation of Micro objects, including duplicate indexes and mixed MicroDataFrame/MicroSeries inputs. Keep selection weights independently mutable on pandas 2 and 3. Reject conflicting or ambiguous weight propagation, and document the limitation of mixed pandas/Micro concatenation.
  Preserve unweighted pandas DataFrame covariance and correlation matrices without attaching observation weights to column summaries.


## [1.4.0] - 2026-09-16

### Added

- MicroSeries.cov and corr now calculate frequency-weighted covariance and Pearson correlation using aligned observations and the left Series weights. Both support pairwise missing-value handling and minimum observation counts; covariance and correlation accept a degrees-of-freedom adjustment. Centered, scaled calculations preserve small differences around large offsets, retain population moments for subnormal positive weights, and avoid overflowing raw frequency-weighted moments when the result is representable. Unsupported correlation methods raise instead of silently using unweighted results.


## [1.3.10] - 2026-09-16

### Fixed

- DataFrame and Series sums accept positional and named axes, skipna and min_count without silently dropping columns, including numeric columns with duplicate labels. DataFrame row sums retain independent observation weights without multiplying them into row values, and empty numeric row sums follow pandas min_count behavior. Explicit axis=None follows the installed pandas version.


## [1.3.9] - 2026-09-16

### Fixed

- Restore tests and lint on pushes to main and build the deployed documentation with MyST instead of the retired Jupyter Book command. Include and validate the `.nojekyll` marker so GitHub Pages can publish the prebuilt site.


## [1.3.8] - 2026-09-15

### Fixed

- Weights and the weight-column name now survive pickling and
  to_pickle/read_pickle. Weighted aggregations also remain available on
  unpickled MicroDataFrames, and weights are preserved by copy.deepcopy.
  Renaming an object preserves its copied weights instead of sharing mutable
  weight state with the original object.


## [1.3.7] - 2026-09-15

### Fixed

- __version__ is now read from package metadata instead of a hardcoded
  0.1.0 that had drifted from pyproject.toml.

  Codecov badge, ROADMAP issue link and CLAUDE.md branch name now point at
  main and the PolicyEngine org.


## [1.3.6] - 2026-09-15

### Fixed

- DataFrame-level aggregations now propagate non-TypeError failures instead of silently omitting columns. The documented gini(negatives="shift") option also accepts nonnegative and empty data.


## [1.3.5] - 2026-09-15

### Fixed

- Weighted quantile() and median() now skip NaN values by default and accept a skipna argument. Quantiles with skipna=False preserve missing groups, and quantile bounds are validated even when values are missing. Grouped vector quantiles retain missing grouping keys with dropna=False, including for multiple keys and empty quantile requests.


## [1.3.4] - 2026-09-15

### Fixed

- MicroDataFrame.nullify_weights and set_weight_col now store weights as an index-aligned Series instead of a bare ndarray, so equals() no longer raises. Selected weight-column values remain independent from the source column.


## [1.3.3] - 2026-09-15

### Fixed

- MicroSeries.nullify_weights now aligns weights to the Series index, so aggregations no longer return 0 on a non-default index.


## [1.3.2] - 2026-09-15

### Fixed

- README code block formatting so make lint passes with ruff 0.16.


## [1.3.1] - 2026-04-28

No significant changes.


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

