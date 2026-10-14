"""Generate the support matrix from the executable regression cases."""

from pathlib import Path
from microdf.tests.test_fail_closed import support_matrix

if __name__ == "__main__":
    Path(__file__).with_name("support.md").write_text(support_matrix())
