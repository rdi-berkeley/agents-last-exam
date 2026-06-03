"""Local verifier for computational_materials_science fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.physical_sciences._shared.materials_science._common import (
    MOSE2_BSE_ABSORPTION_SOC_SPEC,
    SILICON_BSE_ABSORPTION_SPEC,
    SILICON_GW_BANDGAP_SPEC,
    verify_local_directories,
)

SUBCASE_SPECS = {
    "silicon": SILICON_GW_BANDGAP_SPEC,
    "silicon-BSE": SILICON_BSE_ABSORPTION_SPEC,
    "MoSe2-BSE": MOSE2_BSE_ABSORPTION_SOC_SPEC,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    agent_root = Path(args.agent_dir)
    reference_root = Path(args.reference_dir)
    failures: list[str] = []

    for subcase, spec in SUBCASE_SPECS.items():
        result = verify_local_directories(
            spec,
            agent_dir=agent_root / subcase,
            reference_dir=reference_root / subcase,
        )
        if result["failures"]:
            failures.extend([f"{subcase}: {failure}" for failure in result["failures"]])

    payload = {"score": 1.0 if not failures else 0.0, "passed": not failures, "failures": failures}
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
