#!/usr/bin/env bash
# GAP computational discrete algebra system (4.11.x on jammy)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in gap; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg gap] installing: gap"
  apt-get update && apt-get install -y gap && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg gap] OK"
