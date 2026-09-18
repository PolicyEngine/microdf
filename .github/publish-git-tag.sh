#!/usr/bin/env bash
# Tag the commit that published this version, and create its GitHub release.
#
# Run from the publish job after the version bump has landed on main. Exits
# non-zero if either step fails, so a broken release is visible rather than
# silent. Both steps are independently idempotent: an existing tag does not
# stop the release from being created, so a rerun after a partial failure
# finishes the job rather than passing with nothing done.
set -euo pipefail

PYTHON=$(command -v python || command -v python3)
VERSION=$("$PYTHON" -c "import re, pathlib; print(re.search(r'^version\s*=\s*\"(\d+\.\d+\.\d+)\"', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))")
TAG="v${VERSION}"

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Tag ${TAG} already exists locally."
elif git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
  echo "Tag ${TAG} already exists on the remote."
else
  echo "Tagging ${TAG}"
  git tag "${TAG}"
  git push origin "${TAG}"
fi

# Zenodo archives on the GitHub release, not on the tag, so a tag alone leaves
# the DOI pointing at whatever was last released by hand.
if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "Release ${TAG} already exists."
  exit 0
fi

# --verify-tag so gh aborts if the tag is missing from the remote, rather than
# creating one itself at the default branch head.
echo "Creating release ${TAG}"
NOTES=$("$PYTHON" .github/release_notes.py "${VERSION}" 2>/dev/null || true)
if [ -n "${NOTES}" ]; then
  printf '%s\n' "${NOTES}" | gh release create "${TAG}" --title "${TAG}" --verify-tag --notes-file -
else
  gh release create "${TAG}" --title "${TAG}" --verify-tag --generate-notes
fi
