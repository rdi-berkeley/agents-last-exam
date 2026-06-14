#!/usr/bin/env bash
set -euo pipefail
command -v Rscript >/dev/null 2>&1 || { echo "[pkg r-libs-ltmle] FATAL: an R package required first" >&2; exit 1; }
export PPM="https://packagemanager.posit.co/cran/__linux__/jammy/latest"
Rscript --no-init-file -e '
options(repos=c(CRAN=Sys.getenv("PPM")), HTTPUserAgent=sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)), Ncpus=parallel::detectCores())
pk <- strsplit("abind assertthat backports bitops caTools checkmate cli cvAUC data.table digest dplyr foreach future future.apply gam generics globals glue gplots gtools isotone iterators lifecycle listenv magrittr nnls origami parallelly pillar pkgconfig progressr R6 rlang ROCR SuperLearner tibble tidyselect utf8 vctrs withr"," ")[[1]]; pk <- pk[nzchar(pk)]
need <- pk[!pk %in% rownames(installed.packages())]; if (length(need)) install.packages(need)
for (p in pk) if (!requireNamespace(p, quietly=TRUE)) stop(paste("missing R package:", p))
cat("[pkg r-libs-ltmle] R libs OK\n")'
