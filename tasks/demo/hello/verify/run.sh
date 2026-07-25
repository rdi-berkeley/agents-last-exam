#!/usr/bin/env bash
# Compare against the gold answer, which only exists during this stage.
set -euo pipefail

expected="$(tr -d '[:space:]' < /ale/reference/expected.txt)"
actual="$(tr -d '[:space:]' < /ale/output/answer.txt 2>/dev/null || true)"

if [ -n "$actual" ] && [ "$actual" = "$expected" ]; then
    printf '{"rewards": {"reward": 1.0}}' > "$ALE_VERDICT_PATH"
else
    printf '{"rewards": {"reward": 0.0}}' > "$ALE_VERDICT_PATH"
fi
