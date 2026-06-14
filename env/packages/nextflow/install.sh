#!/usr/bin/env bash
# Nextflow (self-installing launcher) -> /usr/local/bin/nextflow. Needs Java.
set -euo pipefail
if ! command -v nextflow >/dev/null 2>&1; then
  command -v java >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: java (jdk-default) required first" >&2; exit 1; }
  curl -fsSL https://get.nextflow.io -o /usr/local/bin/nextflow
  chmod +x /usr/local/bin/nextflow
  # bootstrap the framework jars as the kasm-user so first `nextflow -version`
  # (run by the agent / verify at uid 1000) doesn't need to download/write to a
  # root-owned ~/.nextflow.
  export NXF_HOME=/home/kasm-user/.nextflow
  mkdir -p "$NXF_HOME"
  HOME=/home/kasm-user NXF_HOME="$NXF_HOME" nextflow -version >/dev/null 2>&1 || true
  chown -R 1000:0 "$NXF_HOME" 2>/dev/null || true
fi
command -v nextflow >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: nextflow missing" >&2; exit 1; }
echo "[pkg nextflow] OK"
