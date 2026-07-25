#!/usr/bin/env bash
# Build the answer from the specification, the way a correct agent would: the key/value
# pairs live under payload.expected_kv, and the answer is their compact JSON form.
set -euo pipefail

python3 - <<'PY' > /ale/output/answer.txt
import json

with open("/ale/input/spec.json", encoding="utf-8") as handle:
    spec = json.load(handle)
print(json.dumps(spec["payload"]["expected_kv"], separators=(",", ":")))
PY
