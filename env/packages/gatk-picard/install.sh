#!/usr/bin/env bash
# GATK4 + Picard -> /opt (gatk on PATH). Requires Java.
set -euo pipefail
command -v java >/dev/null 2>&1 || { echo "[pkg gatk-picard] FATAL: java required first" >&2; exit 1; }
if [ ! -x /opt/gatk-4.5.0.0/gatk ]; then
  T="$(mktemp -d)"; curl -fsSL -o "$T/g.zip" "https://github.com/broadinstitute/gatk/releases/download/4.5.0.0/gatk-4.5.0.0.zip"
  (cd /opt && unzip -q "$T/g.zip"); rm -rf "$T"; ln -sf /opt/gatk-4.5.0.0/gatk /usr/local/bin/gatk
fi
if [ ! -f /opt/picard.jar ]; then
  curl -fsSL -o /opt/picard.jar "https://github.com/broadinstitute/picard/releases/download/3.1.1/picard.jar"
fi
gatk --version >/dev/null 2>&1 || { echo "[pkg gatk-picard] FATAL: gatk not runnable" >&2; exit 1; }
echo "[pkg gatk-picard] OK"
