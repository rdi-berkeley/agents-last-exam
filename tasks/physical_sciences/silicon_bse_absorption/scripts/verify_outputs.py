"""Local verifier for silicon_bse_absorption fixtures."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.physical_sciences._shared.materials_science._common import SILICON_BSE_ABSORPTION_SPEC, cli_verify


if __name__ == "__main__":
    raise SystemExit(cli_verify(SILICON_BSE_ABSORPTION_SPEC))
