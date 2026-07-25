#!/usr/bin/env bash
# Runs after the agent, in a workspace the agent can no longer touch.
#
# Write the rewards to $ALE_VERDICT_PATH. A non-zero exit, a missing file or malformed
# JSON is reported as task_error — a defect in the task, which is deliberately distinct
# from the agent legitimately scoring zero.
set -euo pipefail

expected="$(python3 -c 'import json,os; print(json.load(open(os.environ["ALE_PARAMS_JSON"]))["greeting"])') world"
actual="$(cat /ale/output/result.txt 2>/dev/null || true)"

if [ "$actual" = "$expected" ]; then
    printf '{"rewards": {"reward": 1.0}}' > "$ALE_VERDICT_PATH"
else
    printf '{"rewards": {"reward": 0.0}}' > "$ALE_VERDICT_PATH"
fi
