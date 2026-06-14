#!/usr/bin/env bash
# micromamba at ~/.local/bin/micromamba for the kasm-user (conda-forge envs).
set -euo pipefail
U=kasm-user; H=/home/$U
if [ ! -x "$H/.local/bin/micromamba" ]; then
  mkdir -p "$H/.local/bin"
  curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xj -C /tmp bin/micromamba
  install -m0755 /tmp/bin/micromamba "$H/.local/bin/micromamba"
  chown -R 1000:0 "$H/.local"
fi
test -x "$H/.local/bin/micromamba" || { echo "[pkg micromamba] FATAL: micromamba missing" >&2; exit 1; }
echo "[pkg micromamba] OK ($("$H/.local/bin/micromamba" --version 2>&1))"
