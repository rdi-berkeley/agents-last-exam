#!/usr/bin/env bash
# Verify — ranking_node: the vendored pytest env imports + runs via python_task_env.sh
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/ranking_node_feature_parity_recovery_instance_1/base}"
ENVSH="$CANON/software/python_task_env.sh"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$ENVSH" || fail "software/python_task_env.sh not executable"
"$ENVSH" -c "import pytest, pluggy, iniconfig, packaging; print('[verify] vendored pytest', pytest.__version__, 'importable')" || fail "vendored pytest stack not importable via python_task_env.sh"
"$ENVSH" -c "import sys; print('[verify] python', sys.version.split()[0])" || fail "python_task_env.sh python failed"
echo "[verify] PASS"
