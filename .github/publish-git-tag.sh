#!/usr/bin/env bash
# Tag the commit that published this version.
#
# Run from the publish job after the version bump has landed on main. Exits
# non-zero if tagging fails, so a broken release is visible rather than silent.
set -euo pipefail

PYTHON=$(command -v python || command -v python3)
VERSION=$("$PYTHON" -c "import re, pathlib; print(re.search(r'^version\s*=\s*\"(\d+\.\d+\.\d+)\"', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))")
TAG="v${VERSION}"

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Tag ${TAG} already exists locally; nothing to do."
  exit 0
fi

if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "Tag ${TAG} already exists on the remote; nothing to do."
  exit 0
fi

echo "Tagging ${TAG}"
git tag "${TAG}"
git push origin "${TAG}"

# Zenodo archives on the GitHub release, not the tag, so a tag alone leaves the
# DOI pointing at whatever was last released by hand.
if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "Release ${TAG} already exists; nothing to do."
else
  echo "Creating release ${TAG}"
  gh release create "${TAG}" --title "${TAG}" --generate-notes
fi
