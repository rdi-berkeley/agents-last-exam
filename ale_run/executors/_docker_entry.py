"""Supervise Docker deployers and background tools until completion or timeout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ._sandbox_entry import supervise


SPEC_PATH = Path("/work/_spec.json")
RESULT_PATH = Path("/work/_result.json")


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    source = spec["ale_src_root"]
    if source not in sys.path:
        sys.path.insert(0, source)
    out = supervise(SPEC_PATH, spec)
    RESULT_PATH.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
