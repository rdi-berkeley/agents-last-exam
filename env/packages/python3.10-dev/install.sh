#!/usr/bin/env bash
# CPython 3.10 dev headers (Python.h) for compiling C exts
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in python3.10-dev; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg python3.10-dev] installing: python3.10-dev"
  apt-get update && apt-get install -y python3.10-dev && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg python3.10-dev] OK"
