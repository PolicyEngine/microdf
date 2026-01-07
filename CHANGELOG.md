Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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

