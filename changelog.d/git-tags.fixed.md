Release tagging now works: the tag script reads the version from pyproject.toml rather than a helper that never existed, and fails loudly instead of silently.
