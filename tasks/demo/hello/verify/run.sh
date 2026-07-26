#!/usr/bin/env bash
# Compare the answer with the gold file, whatever this variant calls it.
#
# The two variants ship differently named reference files, so the file is discovered
# rather than assumed — one verifier serves both.
set -euo pipefail

gold="$(find /ale/reference -maxdepth 1 -type f | head -1)"
expected="$(tr -d '[:space:]' < "$gold")"
actual="$(cat /ale/output/* 2>/dev/null | tr -d '[:space:]' || true)"

if [ -n "$actual" ] && [ "$actual" = "$expected" ]; then
    printf '{"rewards": {"reward": 1.0}}' > "$ALE_VERDICT_PATH"
else
    printf '{"rewards": {"reward": 0.0}}' > "$ALE_VERDICT_PATH"
fi
