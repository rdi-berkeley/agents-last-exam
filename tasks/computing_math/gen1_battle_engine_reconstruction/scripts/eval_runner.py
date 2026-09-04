"""On-VM eval driver, staged after the agent has finished.

    python3 eval_runner.py <engine.py> <scenario_dir> <per_scenario_timeout_s>

Runs the submission once per scenario file in <scenario_dir> and prints one JSON
object mapping scenario id to {"rc": int, "stdout": str, "stderr": str}. A
submission that crashes, hangs or prints nothing produces a result row rather
than an exception, so the grader scores it as a miss.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def main(argv: list[str]) -> int:
    engine, scen_dir, timeout = argv[1], pathlib.Path(argv[2]), float(argv[3])
    results = {}
    for path in sorted(scen_dir.glob("*.json")):
        try:
            proc = subprocess.run(
                [sys.executable, engine, str(path)],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(scen_dir), check=False,
            )
            results[path.stem] = {"rc": proc.returncode,
                                  "stdout": proc.stdout[-32768:],
                                  "stderr": proc.stderr[-2048:]}
        except subprocess.TimeoutExpired:
            results[path.stem] = {"rc": -1, "stdout": "", "stderr": "timeout"}
        except Exception as exc:  # noqa: BLE001 - a broken submission is a score, not an error
            results[path.stem] = {"rc": -2, "stdout": "", "stderr": repr(exc)[:2048]}
    sys.stdout.write(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
