# Claude Development Guidelines for microdf

## Code Style
- All files must end with a newline character
- **ALWAYS run `make format` before committing** - this will auto-fix most style issues
- Run `make lint` after formatting to check if there are any remaining issues

## Changelog Requirements
- Every PR must include a `changelog_entry.yaml` file at the root
- Format:
  ```yaml
  - bump: patch|minor|major
    changes:
      added|changed|removed|fixed:
      - Description of change
  ```
- The file must not be empty and must end with a newline

## Testing
- Install development dependencies first: `make install` or `pip install -e ".[dev]"`
- Run all tests: `make test` or `pytest -q --cov=microdf --cov-report=xml`
- Run specific test: `python3 -m pytest microdf/tests/test_microseries_dataframe.py::test_df_init -v`
- Ensure all tests pass before creating a PR
- If tests fail, check for:
  - Missing `_link_all_weights()` calls after setting weights
  - Proper handling of both array and string arguments in `set_weights()`
  - Deprecation warnings properly configured with `stacklevel=2`

## Pull Request Process
1. Create a feature branch from master
2. Make changes and ensure they follow code style guidelines
3. Add a changelog entry
4. Create PR with descriptive title and body
5. PRs should close related issues using "Closes #XXX" in the body

## Dependencies
- Dependencies are managed in `pyproject.toml`
- Optional dependencies go in `[project.optional-dependencies]`
- When removing dependencies, also remove from `microdf/_optional.py` VERSIONS dict

## Documentation
- Documentation notebooks are in `docs/` directory
- When removing functionality, consider impact on documentation examples

## Common CI Failures and Solutions

### Test Failures
1. **set_weights() with string not working**: Ensure `_link_all_weights()` is called for both string and array cases
2. **Deprecation warnings in tests**: Import warnings and suppress with `warnings.simplefilter("ignore", DeprecationWarning)` in test setup
3. **MicroSeries not properly linked**: Check that all DataFrame operations call `_link_all_weights()` after modifying structure

### Linting Failures
1. **Line length**: Run `make format` to auto-fix most issues
2. **Import order**: `isort` is configured to work with black, run `make format`
3. **Docstring formatting**: `docformatter` enforces 79-char wrapping, run `make format`
4. **File endings**: Ensure all files end with a newline

### Before Pushing
**CRITICAL: Always run these commands locally before pushing:**
```bash
make format  # Auto-fix style issues (ALWAYS RUN THIS FIRST!)
make lint    # Check for remaining issues (should pass after format)
make test    # Run all tests
```

If CI fails with linting errors, it's almost always because `make format` wasn't run.