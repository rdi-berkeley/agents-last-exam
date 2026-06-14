#!/usr/bin/env bash
# data.table for the legacy R 3.5.1 runtime (healthcare_bias_audit). On R 3.5 PPM
# has no jammy binaries -> source compile (data.table is self-contained C).
# Use a temp .R file (R 3.5.1's Rscript -e is finicky with multi-line scripts).
set -euo pipefail
command -v Rscript >/dev/null 2>&1 || { echo "[pkg r-libs-bias-audit] FATAL: r-base-3.5.1 required first" >&2; exit 1; }
R="$(mktemp --suffix=.R)"
cat > "$R" <<'RS'
options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=parallel::detectCores())
if (!requireNamespace("data.table", quietly=TRUE)) install.packages("data.table")
stopifnot(requireNamespace("data.table", quietly=TRUE))
cat("[pkg r-libs-bias-audit] data.table OK\n")
RS
Rscript "$R"; rm -f "$R"
