#!/usr/bin/env bash
# Nextflow (self-installing launcher) -> /usr/local/bin/nextflow. Needs Java.
set -euo pipefail
if ! command -v nextflow >/dev/null 2>&1; then
  command -v java >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: java (jdk-default) required first" >&2; exit 1; }
  curl -fsSL https://get.nextflow.io -o /usr/local/bin/nextflow
  chmod +x /usr/local/bin/nextflow
  NXF_OFFLINE=false nextflow -version >/dev/null 2>&1 || nextflow -version >/dev/null 2>&1 || true
fi
command -v nextflow >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: nextflow missing" >&2; exit 1; }
echo "[pkg nextflow] OK"
