#!/usr/bin/env bash
# MuJoCo GL/EGL/OSMesa/GLFW render libraries
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in libgl1 libglib2.0-0 libegl1 libosmesa6 libglfw3 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg mujoco-gl-libs] installing: libgl1 libglib2.0-0 libegl1 libosmesa6 libglfw3 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6"
  apt-get update && apt-get install -y libgl1 libglib2.0-0 libegl1 libosmesa6 libglfw3 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg mujoco-gl-libs] OK"
