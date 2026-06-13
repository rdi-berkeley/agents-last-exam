#!/usr/bin/env bash
# Verify — computing_math/os_log_permission_guard_v1 (bash/coreutils/tar + python stdlib)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/computing_math/os_log_permission_guard_v1/base}"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
command -v tar >/dev/null || fail "tar missing"
/usr/bin/python - <<'PY' || fail "python stdlib failed"
import json,csv,os,sys,stat
print("[verify] python", sys.version.split()[0], "stdlib (json/csv/os/stat) OK")
PY
SNAP="$CANON/input/fs_snapshot.tar.gz"; test -f "$SNAP" && tar tzf "$SNAP" >/dev/null 2>&1 && echo "[verify] fs_snapshot.tar.gz lists OK" || fail "cannot read fs_snapshot.tar.gz"
for f in ownership.csv permissions.csv active_writers.json; do test -f "$CANON/input/$f" && echo "[verify] read input/$f" || fail "missing input/$f"; done
echo "[verify] PASS"
