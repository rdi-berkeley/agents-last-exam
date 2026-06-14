#!/usr/bin/env bash
# Verify a task's system environment (run AFTER install_task_deps.sh).
# Usage: env/verify_task_env.sh <task_card.json> [<task_base_dir>]
#   - runs each declared package's meta.json "verify" command, AND
#   - if <task_base_dir>/input/runtime_env exists, builds it with uv (proves the
#     task's pinned Python runtime resolves against the installed system libs).
# This is generic: per-task verification logic lives in package metas + the
# task's own runtime_env, so no per-task verify scripts are needed.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD="${1:?usage: verify_task_env.sh <task_card.json> [task_base_dir]}"
BASE="${2:-}"
command -v jq >/dev/null 2>&1 || { apt-get update && apt-get install -y jq >/dev/null 2>&1; }
mapfile -t PKGS < <(jq -r '.requiredSystemPackages[]? // empty' "$CARD")
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }

# Start dockerd if a docker-dependent package is present and the daemon is down.
if printf '%s\n' "${PKGS[@]}" | grep -qE 'docker-ce|orfs-image'; then
  if ! docker info >/dev/null 2>&1; then
    (command -v sudo >/dev/null 2>&1 && sudo sh -c 'nohup dockerd >/var/log/dockerd.log 2>&1 &') || nohup dockerd >/var/log/dockerd.log 2>&1 &
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
fi

echo "[verify] packages: ${PKGS[*]:-<none>}"
for p in "${PKGS[@]}"; do
  v="$(jq -r '.verify // empty' "$SCRIPT_DIR/packages/$p/meta.json" 2>/dev/null)"
  [ -n "$v" ] || { echo "[verify]  - $p: (no verify cmd, install self-checked)"; continue; }
  if bash -c "$v" >/dev/null 2>&1; then echo "[verify]  - $p: OK"; else fail "package '$p' verify failed: $v"; fi
done

if [ -n "$BASE" ]; then
  RTE=""
  [ -f "$BASE/input/runtime_env/pyproject.toml" ] && RTE="$BASE/input/runtime_env"
  [ -z "$RTE" ] && [ -f "$BASE/input/pyproject.toml" ] && RTE="$BASE/input"
  if [ -n "$RTE" ]; then
    export UV_PROJECT_ENVIRONMENT="$BASE/output/.verify_venv" UV_CACHE_DIR="$BASE/output/.uv_cache"
    # --no-install-project: install only the declared dependencies, not the task's
    # own meta-project. Many runtime_env manifests set [tool.uv] package = false and
    # ship no source dir; building the root would falsely fail (e.g. simglucose).
    S="uv sync --frozen --no-install-project --project $RTE"; [ -f "$RTE/uv.lock" ] || S="uv sync --no-install-project --project $RTE"
    echo "[verify] building task runtime_env: $S"
    $S >/dev/null 2>&1 || $S || fail "task runtime_env failed to build (missing system lib?)"
    echo "[verify] runtime_env built OK"
  fi
fi
echo "[verify] PASS"
