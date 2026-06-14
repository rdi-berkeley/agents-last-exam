#!/usr/bin/env bash
# NIST FDS + Smokeview 6.10.1 -> /opt/fds-smv-6.10.1 (installer `extract` mode).
set -euo pipefail
P=/opt/fds-smv-6.10.1
if [ ! -x "$P/bin/fds" ]; then
  T="$(mktemp -d)"; cd "$T"
  curl --fail --location --silent --show-error -o fds.sh \
    "https://github.com/firemodels/fds/releases/download/FDS-6.10.1/FDS-6.10.1_SMV-6.10.1_lnx.sh"
  printf 'extract\n' | bash fds.sh >/dev/null 2>&1 || true
  TGZ="$(ls FDS-6.10.1_SMV-6.10.1_lnx.tar.gz 2>/dev/null | head -1)"
  test -f "$TGZ" || { echo "[pkg fds-smv-6.10.1] FATAL: installer produced no payload" >&2; exit 1; }
  mkdir -p "$P"; tar xzf "$TGZ" -C "$P"; cd /; rm -rf "$T"
fi
VER="$(LD_LIBRARY_PATH=$P/bin/INTEL/lib "$P/bin/fds" 2>&1 | grep -iE 'version|6\.10\.1' | head -1 || true)"
echo "$VER" | grep -q "6.10.1" || { echo "[pkg fds-smv-6.10.1] FATAL: fds not 6.10.1 ($VER)" >&2; exit 1; }
echo "[pkg fds-smv-6.10.1] OK ($VER)"
