#!/usr/bin/env bash
# Apptainer 1.3.0 (matches dev VM dev-ubuntu22) — runs the Neurodesk .simg GUI
# containers for the brain-science scene tasks (scene2_resample, scene3_skullstrip_qc).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v apptainer >/dev/null 2>&1; then
  T="$(mktemp -d)"
  curl -fsSL -o "$T/apptainer.deb" \
    "https://github.com/apptainer/apptainer/releases/download/v1.3.0/apptainer_1.3.0_amd64.deb"
  apt-get update -qq
  apt-get install -y -qq "$T/apptainer.deb"   # apt resolves the .deb's dependencies
  rm -rf "$T"
fi
out="$(apptainer --version 2>/dev/null || true)"
echo "$out" | grep -q "1.3.0" || { echo "[pkg apptainer-1.3.0] FATAL: apptainer not 1.3.0 (got: $out)" >&2; exit 1; }
echo "[pkg apptainer-1.3.0] OK ($out)"
