#!/usr/bin/env bash
# R 4.3.2 via rig (r-lib R Installation Manager), symlinked to /usr/bin/{R,Rscript}
# so task software/Rscript wrappers resolve it. Used where a task pins R 4.5.x and
# CRAN/Bioconductor binaries (PPM) exist for that R minor (e.g. tradeSeq).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v rig >/dev/null 2>&1; then
  curl -fsSL -o /tmp/rig.tgz "https://github.com/r-lib/rig/releases/download/v0.8.0/rig-linux-0.8.0.tar.gz"
  tar xzf /tmp/rig.tgz -C /usr/local; rm -f /tmp/rig.tgz
fi
if [ ! -x /opt/R/4.3.2/bin/Rscript ]; then rig add 4.3.2; fi
rig default 4.3.2 >/dev/null 2>&1 || true
ln -sf /opt/R/4.3.2/bin/R  /usr/bin/R;        ln -sf /opt/R/4.3.2/bin/Rscript /usr/bin/Rscript
ln -sf /opt/R/4.3.2/bin/R  /usr/local/bin/R;  ln -sf /opt/R/4.3.2/bin/Rscript /usr/local/bin/Rscript
Rscript --version 2>&1 | grep -q "4.3.2" || { echo "[pkg r-base-4.3.2] FATAL: R 4.3.2 not active" >&2; exit 1; }
echo "[pkg r-base-4.3.2] OK ($(Rscript -e 'cat(R.version.string)' 2>/dev/null))"
