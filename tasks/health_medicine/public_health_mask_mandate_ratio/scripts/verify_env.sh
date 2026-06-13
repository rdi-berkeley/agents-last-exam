#!/usr/bin/env bash
# Verify — public_health_mask: Rscript runs + the required R packages load + a tiny model fits
set -uo pipefail
fail(){ echo "[verify] FATAL: $*" >&2; exit 1; }
command -v Rscript >/dev/null 2>&1 || fail "Rscript missing"
echo "[verify] $(Rscript -e 'cat(R.version.string)' 2>/dev/null)"
Rscript -e '
for (p in c("nlme","MASS","splines","dplyr","jsonlite")) stopifnot(requireNamespace(p, quietly=TRUE))
suppressMessages(library(nlme)); suppressMessages(library(dplyr))
d <- data.frame(x=1:20, y=rnorm(20)); m <- lm(y~x, d); j <- jsonlite::toJSON(list(ok=TRUE))
cat("[verify] R packages load + lm()/jsonlite OK; coef=", round(coef(m)[2],3), "\n")
' || fail "R package/model probe failed"
echo "[verify] PASS"
