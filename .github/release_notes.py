"""Print the CHANGELOG section for one version, for use as release notes.

towncrier writes the curated entry; `gh release create --generate-notes` would
write a raw commit list instead. Prints nothing if the section is absent, and
the caller falls back to generated notes.
"""

import re
import sys
from pathlib import Path


def section_for(changelog: str, version: str) -> str:
    pattern = rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)"
    match = re.search(pattern, changelog, re.M | re.S)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    path = Path("CHANGELOG.md")
    if not path.exists():
        sys.exit(0)
    print(section_for(path.read_text(), sys.argv[1]))
