#!/usr/bin/env bash
# AmberTools 23 conda-forge env at /opt/ambertools-23 (AMBERHOME). The task
# wrapper execs $AMBERHOME/bin/{tleap,MMPBSA.py,antechamber,cpptraj,parmchk2}.
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg ambertools-23] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/ambertools-23
if [ ! -x "$P/bin/tleap" ]; then
  export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
  "$MM" create -y -p "$P" -c conda-forge ambertools=23
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
test -x "$P/bin/tleap" || { echo "[pkg ambertools-23] FATAL: tleap missing" >&2; exit 1; }
echo "[pkg ambertools-23] OK ($("$P/bin/tleap" --version 2>&1 | head -1 || echo tleap-present))"
