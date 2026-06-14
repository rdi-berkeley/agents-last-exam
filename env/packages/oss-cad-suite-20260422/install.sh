#!/usr/bin/env bash
# OSS CAD Suite 20260422 (yosys + friends) -> /opt/oss-cad-suite-20260422.
set -euo pipefail
P=/opt/oss-cad-suite-20260422
if [ ! -x "$P/bin/yosys" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/oss.tgz" \
    "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-04-22/oss-cad-suite-linux-x64-20260422.tgz"
  mkdir -p "$P"; tar -xzf "$T/oss.tgz" -C "$P" --strip-components=1; rm -rf "$T"
fi
out="$("$P/bin/yosys" --version 2>&1 || true)"; echo "$out" | grep -q "Yosys" || { echo "[pkg oss-cad-suite] FATAL: yosys not runnable" >&2; exit 1; }
echo "[pkg oss-cad-suite-20260422] OK ($out)"
