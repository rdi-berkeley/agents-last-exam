#!/usr/bin/env bash
# Verify — fds: software/fds runs FDS 6.10.1 (with bundled Intel MPI runtime)
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/transport_safety/fds_single_compartment_detector_reconstruction/base}"
SW="$CANON/software"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
test -x "$SW/fds" || fail "software/fds not executable"
OUT="$("$SW/fds" 2>&1 | grep -iE 'fire dynamics|version|6\.10\.1' | head -2 || true)"
echo "[verify] fds: $(echo "$OUT" | head -1)"
echo "$OUT" | grep -q "6.10.1" || fail "FDS did not report version 6.10.1"
test -x "$SW/smokeview" && echo "[verify] smokeview wrapper present" || fail "software/smokeview missing"
echo "[verify] PASS"
