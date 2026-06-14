#!/usr/bin/env bash
# rtplan-matrad micromamba env (Octave 6.4 + python) + matRad checkout at
# /opt/matrad-c014dc82. software/run_matrad.sh execs `micromamba run -n
# rtplan-matrad octave --path /opt/matrad-c014dc82`.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg matrad-rtplan] FATAL: micromamba required first" >&2; exit 1; }
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if ! "$MM" env list 2>/dev/null | grep -q "rtplan-matrad"; then
  "$MM" create -y -n rtplan-matrad -c conda-forge "octave=6.4" numpy scipy
  chown -R 1000:0 "$MAMBA_ROOT_PREFIX" 2>/dev/null || true
  mkdir -p /home/kasm-user/.cache/mamba; chown -R 1000:0 /home/kasm-user/.cache 2>/dev/null || true
fi
if [ ! -d /opt/matrad-c014dc82/.git ]; then
  command -v git >/dev/null 2>&1 || { apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*; }
  git clone --quiet https://github.com/e0404/matRad.git /opt/matrad-c014dc82
  git -C /opt/matrad-c014dc82 checkout --quiet c014dc82 || true
  chown -R 1000:0 /opt/matrad-c014dc82 2>/dev/null || true
fi
"$MM" run -n rtplan-matrad octave --no-gui --eval 'disp(version)' >/dev/null 2>&1 || { echo "[pkg matrad-rtplan] FATAL: octave env not runnable" >&2; exit 1; }
test -d /opt/matrad-c014dc82 || { echo "[pkg matrad-rtplan] FATAL: matRad checkout missing" >&2; exit 1; }
echo "[pkg matrad-rtplan] OK (octave 6.4 env + matRad)"
