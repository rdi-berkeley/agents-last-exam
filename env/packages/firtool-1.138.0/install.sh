#!/usr/bin/env bash
# firtool 1.138.0 (CIRCT) -> /opt/firtool-1.138.0.
set -euo pipefail
P=/opt/firtool-1.138.0
if [ ! -x "$P/bin/firtool" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/ft.tgz" \
    "https://github.com/llvm/circt/releases/download/firtool-1.138.0/firrtl-bin-linux-x64.tar.gz"
  tar -xzf "$T/ft.tgz" -C /opt
  test -x "$P/bin/firtool" || { mkdir -p "$P"; tar -xzf "$T/ft.tgz" -C "$P" --strip-components=1; }
  rm -rf "$T"
fi
"$P/bin/firtool" --version 2>&1 | grep -q "1.138.0" || { echo "[pkg firtool-1.138.0] FATAL: not 1.138.0" >&2; exit 1; }
echo "[pkg firtool-1.138.0] OK"
