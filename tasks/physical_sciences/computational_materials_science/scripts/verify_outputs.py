"""Local verifier for computational_materials_science silicon GW fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.physical_sciences._shared.materials_science._common import (  # noqa: E402
    SILICON_GW_BANDGAP_SPEC,
    verify_local_directories,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    agent_root = Path(args.agent_dir)
    reference_root = Path(args.reference_dir)
    result = verify_local_directories(
        SILICON_GW_BANDGAP_SPEC,
        agent_dir=agent_root / "silicon",
        reference_dir=reference_root / "silicon",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
