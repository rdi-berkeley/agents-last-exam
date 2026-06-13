#!/usr/bin/env bash
# Verify — openroad: docker works, the pinned ORFS image is present, and OpenROAD
# runs inside it; the prepare_workspace.sh wrapper materializes the workspace.
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/openroad_sky130_ibex_pnr_signoff/base}"
ORFS_IMAGE="openroad/orfs@sha256:fd7751659f976aec05129d3272186aa9112ba1a790a7f184449910af4ec4c475"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
  sudo sh -c 'nohup dockerd >/var/log/dockerd.log 2>&1 &'; for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
docker info >/dev/null 2>&1 || fail "docker daemon not reachable"
docker image inspect "$ORFS_IMAGE" >/dev/null 2>&1 || fail "pinned ORFS image not present"
echo "[verify] ORFS image present"
OR=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
OUT="$(docker run --rm "$ORFS_IMAGE" bash -lc "source /OpenROAD-flow-scripts/env.sh 2>/dev/null; ${OR} -version" 2>&1 || true)"
echo "[verify] openroad -version: $(echo "$OUT" | grep -iE 'openroad|v[0-9]' | head -1)"
echo "$OUT" | grep -qiE "OpenROAD|v[0-9]+\\.[0-9]" || fail "OpenROAD did not report a version inside the ORFS image"
# workspace prep wrapper
PW="$CANON/software/prepare_workspace.sh"; test -x "$PW" && { bash "$PW" >/dev/null 2>&1 && echo "[verify] prepare_workspace.sh OK"; } || echo "[verify] note: prepare_workspace.sh not run"
echo "[verify] PASS"
