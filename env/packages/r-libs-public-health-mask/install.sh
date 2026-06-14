#!/usr/bin/env bash
# R library closure for a task (installed into R's site-library). Uses Posit
# Package Manager jammy binaries for speed; falls back to source compile.
set -euo pipefail
command -v Rscript >/dev/null 2>&1 || { echo "[pkg r-libs-public-health-mask] FATAL: r-base required first" >&2; exit 1; }
export PPM="https://packagemanager.posit.co/cran/__linux__/jammy/latest"
Rscript --no-init-file -e '
options(repos=c(CRAN=Sys.getenv("PPM")),
        HTTPUserAgent=sprintf("R/%s R (%s)", getRversion(),
          paste(getRversion(), R.version$platform, R.version$arch, R.version$os)),
        Ncpus=parallel::detectCores())
cran <- strsplit("dplyr jsonlite nlme MASS"," ")[[1]]; cran <- cran[nzchar(cran)]
need <- cran[!cran %in% rownames(installed.packages())]
if (length(need)) install.packages(need)
bioc <- strsplit(""," ")[[1]]; bioc <- bioc[nzchar(bioc)]
if (length(bioc)) {
  if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
  need2 <- bioc[!bioc %in% rownames(installed.packages())]
  if (length(need2)) BiocManager::install(need2, ask=FALSE, update=FALSE)
}
all <- c(cran, bioc)
for (p in all) if (!requireNamespace(p, quietly=TRUE)) stop(paste("missing R package:", p))
cat("[pkg r-libs-public-health-mask] R libs OK:", paste(all, collapse=", "), "\n")
'
