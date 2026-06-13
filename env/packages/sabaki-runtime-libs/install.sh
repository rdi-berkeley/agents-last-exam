#!/usr/bin/env bash
# FUSE + Electron/Chromium runtime libs (for AppImage GUIs like Sabaki)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in libfuse2 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgbm1 libgtk-3-0 libasound2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 libx11-xcb1 libxss1 libxtst6 libsecret-1-0 libnotify4; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg sabaki-runtime-libs] installing: libfuse2 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgbm1 libgtk-3-0 libasound2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 libx11-xcb1 libxss1 libxtst6 libsecret-1-0 libnotify4"
  apt-get update && apt-get install -y libfuse2 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgbm1 libgtk-3-0 libasound2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 libx11-xcb1 libxss1 libxtst6 libsecret-1-0 libnotify4 && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg sabaki-runtime-libs] OK"
