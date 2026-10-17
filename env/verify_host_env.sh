#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null || { printf 'uv is required. See env/README.md.\n' >&2; exit 1; }
cd -- "$REPO_ROOT"
uv lock --check --offline
uv pip check
exec uv run --locked --no-sync python -m ale_run.orchestration.host_preflight "$@"
