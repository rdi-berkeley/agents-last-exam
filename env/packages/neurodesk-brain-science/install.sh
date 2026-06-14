#!/usr/bin/env bash
# Apptainer + the brain_science computer-use GUI benchmark bundle at
# /home/user/brain_science/computer_use_benchmark_bundle (Neurodesk .simg images
# for Slicer/FSL/etc., multi-GB, GUI). The bundle + Neurodesk images are large
# and GUI-bound; provide them via $BRAIN_SCIENCE_BUNDLE (a prepared tarball) and
# $NEURODESK_IMAGES, otherwise this installs apptainer and reports what is missing.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v apptainer >/dev/null 2>&1 || {
  add-apt-repository -y ppa:apptainer/ppa 2>/dev/null || true
  apt-get update && apt-get install -y apptainer || apt-get install -y singularity-ce || true
  rm -rf /var/lib/apt/lists/* || true; }
B=/home/user/brain_science/computer_use_benchmark_bundle
if [ ! -x "$B/run_scene.sh" ]; then
  if [ -n "${BRAIN_SCIENCE_BUNDLE:-}" ] && [ -f "${BRAIN_SCIENCE_BUNDLE}" ]; then
    mkdir -p /home/user/brain_science; tar -xzf "$BRAIN_SCIENCE_BUNDLE" -C /home/user/brain_science
  else
    echo "[pkg neurodesk-brain-science] BLOCKED: needs the brain_science bundle + Neurodesk GUI .simg" >&2
    echo "[pkg neurodesk-brain-science] images (multi-GB). Set BRAIN_SCIENCE_BUNDLE=/path/bundle.tgz to finish." >&2
    command -v apptainer >/dev/null 2>&1 && echo "[pkg neurodesk-brain-science] apptainer installed." >&2
    exit 3
  fi
fi
echo "[pkg neurodesk-brain-science] OK"
