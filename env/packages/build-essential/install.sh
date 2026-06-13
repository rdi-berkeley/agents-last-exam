#!/usr/bin/env bash
# C/C++ toolchain (g++, make) for C-extension/native builds
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in build-essential; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg build-essential] installing: build-essential"
  apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg build-essential] OK"
