#!/usr/bin/env bash
# OpenCV + PyQt5(xcb) runtime system libraries
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in libgl1 libglib2.0-0 libegl1 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-cursor0 libxcb-util1; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg qt-opencv-libs] installing: libgl1 libglib2.0-0 libegl1 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-cursor0 libxcb-util1"
  apt-get update && apt-get install -y libgl1 libglib2.0-0 libegl1 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-cursor0 libxcb-util1 && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg qt-opencv-libs] OK"
