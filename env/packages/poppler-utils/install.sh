#!/usr/bin/env bash
# pdftotext + poppler CLI (usually already in base)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in poppler-utils; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg poppler-utils] installing: poppler-utils"
  apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg poppler-utils] OK"
