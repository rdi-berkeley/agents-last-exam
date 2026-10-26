#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { printf 'uv is required. See env/README.md.\n' >&2; exit 1; }
cd -- "$REPO_ROOT"
uv lock --check --offline
# `uv pip check` on an aarch64 CPU host flags the locked x86-only NVIDIA CUDA
# wheels as "built for a different platform"; nothing here loads them (no GPU on
# the orchestrator), so only other incompatibilities fail the check.
if ! check_out="$(uv pip check 2>&1)"; then
  if printf '%s\n' "$check_out" | grep -v -E '^(Resolved|Checked|Found [0-9]+ incompat)|^The package `nvidia-[a-z0-9-]+` was built for a different platform' | grep -q .; then
    printf '%s\n' "$check_out" >&2; exit 1
  fi
fi
exec uv run --locked --no-sync python -m ale_run.orchestration.host_preflight "$@"
