#!/usr/bin/env bash
# Ensembl VEP via conda-forge/bioconda at /opt/ensembl-vep (vep on PATH). The
# species cache is large task DATA, not installed here.
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg ensembl-vep] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/ensembl-vep
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if [ ! -x "$P/bin/vep" ]; then
  "$MM" create -y -p "$P" -c bioconda -c conda-forge ensembl-vep
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
ln -sf "$P/bin/vep" /usr/local/bin/vep
test -x "$P/bin/vep" || { echo "[pkg ensembl-vep] FATAL: vep missing" >&2; exit 1; }
echo "[pkg ensembl-vep] OK"
