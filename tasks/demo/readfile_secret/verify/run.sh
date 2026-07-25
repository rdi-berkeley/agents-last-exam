#!/usr/bin/env bash
# Compare the agent's answer with the token that was actually staged.
#
# Ground truth is read from disk rather than regenerated: a token minted a second time
# would differ from the one the agent saw, and the task would fail agents that were
# right.
set -euo pipefail

truth="$(tr -d '[:space:]' < /ale/input/secret.txt)"
answer="$(tr -d '[:space:]' < /ale/output/answer.txt 2>/dev/null || true)"

if [ -n "$answer" ] && [ "$answer" = "$truth" ]; then
    printf '{"rewards": {"reward": 1.0}}' > "$ALE_VERDICT_PATH"
else
    printf '{"rewards": {"reward": 0.0}}' > "$ALE_VERDICT_PATH"
fi
