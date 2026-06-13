#!/usr/bin/env bash
# EnergyPlus 22.1.0 -> /opt/energyplus-22.1.0 (NREL GitHub release tarball).
set -euo pipefail
P=/opt/energyplus-22.1.0
if [ ! -x "$P/energyplus" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/ep.tgz" \
    "https://github.com/NREL/EnergyPlus/releases/download/v22.1.0/EnergyPlus-22.1.0-ed759b17ee-Linux-Ubuntu20.04-x86_64.tar.gz"
  mkdir -p "$P"; tar -xzf "$T/ep.tgz" -C "$P" --strip-components=1; rm -rf "$T"
fi
VER="$("$P/energyplus" --version 2>/dev/null || true)"
echo "$VER" | grep -q "22.1.0" || { echo "[pkg energyplus-22.1.0] FATAL: not 22.1.0 ($VER)" >&2; exit 1; }
echo "[pkg energyplus-22.1.0] OK ($VER)"
