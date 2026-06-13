#!/usr/bin/env bash
# Verify — humanoid_wbc: system env for the staged mjlab/MuJoCo runtime.
#  * MuJoCo's GL/EGL/OSMesa loaders are present (renderer runs headless)
#  * uv + python present (the agent builds mjlab's env from the staged uv.lock)
#  * the staged mjlab runtime archive + asset zips unpack
set -uo pipefail
CANON="${CANON:-/media/user/data/agenthle/engineering/humanoid_wbc_policy_evaluation/base}"
RTE="$CANON/input/runtime_env"
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
for lib in libGL.so.1 libEGL.so.1 libOSMesa.so.8 libglfw.so.3; do
  ldconfig -p | grep -q "$lib" || fail "MuJoCo render lib $lib missing"
done
echo "[verify] MuJoCo GL/EGL/OSMesa/GLFW loaders present"
command -v uv >/dev/null && /usr/bin/python --version >/dev/null || fail "python/uv missing"
W="$(mktemp -d)"
unzip -qq "$RTE/mjlab.zip" -d "$W" 2>/dev/null || fail "mjlab.zip will not unpack"
MJDIR="$(find "$W" -maxdepth 1 -mindepth 1 -type d | head -1)"
test -f "$MJDIR/pyproject.toml" && test -f "$MJDIR/uv.lock" && echo "[verify] mjlab runtime unpacked (pyproject + uv.lock present)" || fail "mjlab archive missing pyproject/uv.lock"
for z in motions-1.zip policies.zip; do test -f "$RTE/$z" && unzip -l "$RTE/$z" >/dev/null 2>&1 && echo "[verify] $z lists OK" || fail "$z missing/unreadable"; done
echo "[verify] PASS"
