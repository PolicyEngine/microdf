Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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



[1.0.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.6.0...1.0.0
[0.6.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.7...0.5.0
[0.4.7]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.6...0.4.7
[0.4.6]: https://github.com/PolicyEngine/microcalibrate/compare/0.4.5...0.4.6

