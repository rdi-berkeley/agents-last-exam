#!/usr/bin/env bash
# CellProfiler 4.2.8 conda-forge env at /opt/cellprofiler-4.2.8 (wrapper execs
# /opt/cellprofiler-4.2.8/bin/cellprofiler).
set -euo pipefail
MM=/home/kasm-user/.local/bin/micromamba
[ -x "$MM" ] || { echo "[pkg cellprofiler-4.2.8] FATAL: micromamba required first" >&2; exit 1; }
P=/opt/cellprofiler-4.2.8
export MAMBA_ROOT_PREFIX=/home/kasm-user/.local/share/micromamba
if [ ! -x "$P/bin/cellprofiler" ]; then
  "$MM" create -y -p "$P" -c bioconda -c conda-forge cellprofiler=4.2.8.1
  chown -R 1000:0 "$P" 2>/dev/null || true
fi
test -x "$P/bin/cellprofiler" || { echo "[pkg cellprofiler-4.2.8] FATAL: cellprofiler missing" >&2; exit 1; }
echo "[pkg cellprofiler-4.2.8] OK"
