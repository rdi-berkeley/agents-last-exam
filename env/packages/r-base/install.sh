#!/usr/bin/env bash
# R (latest 4.x) + Rscript from the CRAN apt repo (jammy ships only 4.1).
# Task-specific R libraries (dplyr, BiocManager/TCGAbiolinks, ...) are installed
# by the task itself (analogous to RTENV python); this provides the R runtime.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v Rscript >/dev/null 2>&1 || ! Rscript --version 2>&1 | grep -qE "4\.[2-9]"; then
  apt-get update; apt-get install -y --no-install-recommends ca-certificates gnupg curl
  curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor -o /usr/share/keyrings/r-project.gpg
  echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/r-project.list
  apt-get update; apt-get install -y --no-install-recommends r-base r-base-dev; rm -rf /var/lib/apt/lists/*
fi
command -v Rscript >/dev/null 2>&1 || { echo "[pkg r-base] FATAL: Rscript missing" >&2; exit 1; }
echo "[pkg r-base] OK ($(Rscript -e 'cat(R.version.string)' 2>/dev/null))"
