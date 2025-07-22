# Claude Development Guidelines for microdf

## Code Style
- All files must end with a newline character
- Run `make lint` before committing to catch style issues

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
- Run tests with: `python3 -m pytest microdf/tests/ -v`
- Ensure all tests pass before creating a PR

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