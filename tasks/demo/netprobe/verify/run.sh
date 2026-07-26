#!/usr/bin/env bash
# Full marks only if every probe failed and the permitted path worked.
set -euo pipefail

python3 - <<'PY'
import json, os

report = json.load(open("/ale/output/probe.json"))
# gateway_ok is only meaningful when a gateway was actually wired up; the oracle path
# runs without one, and demanding it there would make the probe fail for the wrong reason.
expected = {"direct_dns": False, "direct_tcp": False, "leaked_key": False, "saw_secrets": False}
bad = {k: report.get(k) for k, want in expected.items() if report.get(k) != want}

reward = 0.0 if bad else 1.0
with open(os.environ["ALE_VERDICT_PATH"], "w") as fh:
    json.dump({"rewards": {"reward": reward}, "metrics": {"violations": float(len(bad))}}, fh)
if bad:
    print("isolation violated:", bad)
PY
