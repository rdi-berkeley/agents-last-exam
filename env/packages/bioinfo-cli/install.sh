#!/usr/bin/env bash
# Core read-mapping/variant CLIs: bwa, samtools, bcftools, fastqc (apt) + multiqc.
# (jammy apt versions are functional; tasks call them on PATH.)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
PKGS="bwa samtools bcftools fastqc"
need=0; for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
[ "$need" = "1" ] && { apt-get update && apt-get install -y $PKGS && rm -rf /var/lib/apt/lists/*; }
if ! command -v multiqc >/dev/null 2>&1; then
  command -v uv >/dev/null 2>&1 && uv tool install multiqc >/dev/null 2>&1 || pip install --break-system-packages -q multiqc || true
  [ -x "$HOME/.local/bin/multiqc" ] && ln -sf "$HOME/.local/bin/multiqc" /usr/local/bin/multiqc 2>/dev/null || true
fi
for b in bwa samtools bcftools fastqc; do command -v $b >/dev/null 2>&1 || { echo "[pkg bioinfo-cli] FATAL: $b missing" >&2; exit 1; }; done
echo "[pkg bioinfo-cli] OK (bwa/samtools/bcftools/fastqc$(command -v multiqc >/dev/null && echo /multiqc))"
