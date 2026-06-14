#!/usr/bin/env bash
# R + Bioconductor (SingleCellExperiment, tradeSeq) + mgcv via bioconda BINARIES
# at /opt/r-bioc (avoids the source-compile ABI failure of tradeSeq on rig R).
# Symlinks /usr/bin/Rscript so the task's direct Rscript resolves it.
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg bioc-tradeseq-conda] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/r-bioc
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if [ ! -x "$P/bin/Rscript" ]; then
  "$MM" create -y -p "$P" -c bioconda -c conda-forge \
    bioconductor-singlecellexperiment bioconductor-tradeseq r-mgcv
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
ln -sf "$P/bin/Rscript" /usr/bin/Rscript; ln -sf "$P/bin/R" /usr/bin/R
ln -sf "$P/bin/Rscript" /usr/local/bin/Rscript; ln -sf "$P/bin/R" /usr/local/bin/R
Rscript -e 'stopifnot(requireNamespace("tradeSeq",quietly=TRUE), requireNamespace("SingleCellExperiment",quietly=TRUE))' || { echo "[pkg bioc-tradeseq-conda] FATAL: Bioc pkgs missing" >&2; exit 1; }
echo "[pkg bioc-tradeseq-conda] OK"
