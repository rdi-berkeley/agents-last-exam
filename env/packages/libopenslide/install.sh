#!/usr/bin/env bash
# OpenSlide C runtime library (for openslide-python)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in libopenslide0; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg libopenslide] installing: libopenslide0"
  apt-get update && apt-get install -y libopenslide0 && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg libopenslide] OK"
