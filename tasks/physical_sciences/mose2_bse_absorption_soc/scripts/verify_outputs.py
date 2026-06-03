"""Local verifier for mose2_bse_absorption_soc fixtures."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.physical_sciences._shared.materials_science._common import MOSE2_BSE_ABSORPTION_SOC_SPEC, cli_verify


if __name__ == "__main__":
    raise SystemExit(cli_verify(MOSE2_BSE_ABSORPTION_SOC_SPEC))
