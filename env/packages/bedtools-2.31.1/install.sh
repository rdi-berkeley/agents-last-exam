#!/usr/bin/env bash
# BEDTools 2.31.1 -> /opt/bedtools-2.31.1 (built from source; no static release
# for 2.31.1, and jammy apt ships only 2.30.0).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
P=/opt/bedtools-2.31.1
if [ ! -x "$P/bin/bedtools" ]; then
  for d in build-essential zlib1g-dev libbz2-dev liblzma-dev; do dpkg -s "$d" >/dev/null 2>&1 || { apt-get update; apt-get install -y build-essential zlib1g-dev libbz2-dev liblzma-dev; rm -rf /var/lib/apt/lists/*; break; }; done
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/b.tgz" \
    "https://github.com/arq5x/bedtools2/releases/download/v2.31.1/bedtools-2.31.1.tar.gz"
  tar -xzf "$T/b.tgz" -C "$T"
  make -C "$T/bedtools2" -j"$(nproc)" >/dev/null
  mkdir -p "$P/bin"; install -m0755 "$T/bedtools2/bin/"* "$P/bin/"
  rm -rf "$T"
fi
"$P/bin/bedtools" --version 2>&1 | grep -q "v2.31.1" || { echo "[pkg bedtools-2.31.1] FATAL: not v2.31.1" >&2; exit 1; }
echo "[pkg bedtools-2.31.1] OK ($("$P/bin/bedtools" --version 2>&1))"
