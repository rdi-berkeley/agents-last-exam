#!/usr/bin/env bash
# LibreOffice Calc + a /usr/bin/libreoffice shim. The base image's global
# LD_LIBRARY_PATH (CUA/nvidia) breaks LibreOffice's UNO bootstrap
# (DeploymentException); the shim clears it so soffice uses its bundled libs.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if [ ! -x /usr/lib/libreoffice/program/soffice ]; then
  echo "[pkg libreoffice-calc] installing libreoffice-calc ..."
  apt-get update && apt-get install -y libreoffice-calc && rm -rf /var/lib/apt/lists/*
fi
if ! head -1 /usr/bin/libreoffice 2>/dev/null | grep -q 'ALE-DEPS-SHIM'; then
  rm -f /usr/bin/libreoffice
  cat > /usr/bin/libreoffice <<'SHIM'
#!/bin/bash
# ALE-DEPS-SHIM: drop the base image's global LD_LIBRARY_PATH so LibreOffice's
# UNO bootstrap uses its own bundled libraries ($ORIGIN RUNPATH).
unset LD_LIBRARY_PATH
exec /usr/lib/libreoffice/program/soffice "$@"
SHIM
  chmod +x /usr/bin/libreoffice
fi
VER="$(/usr/bin/libreoffice --version 2>/dev/null | head -1)"
test -n "$VER" || { echo "[pkg libreoffice-calc] FATAL: libreoffice will not run" >&2; exit 1; }
echo "[pkg libreoffice-calc] OK ($VER)"
