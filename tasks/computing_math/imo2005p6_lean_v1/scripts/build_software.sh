#!/usr/bin/env bash
# Build the task's software/ artifact (Lean 4.27.0 toolchain + Mathlib v4.27 oleans).
# Usage: build_software.sh <ELAN_TOOLCHAIN_DIR> <MATHLIB_LAKE_PACKAGES_DIR> <OUT_DIR>
#   ELAN_TOOLCHAIN_DIR   e.g. ~/.elan/toolchains/leanprover--lean4---v4.27.0
#   MATHLIB_LAKE_PACKAGES_DIR  e.g. <checkout>/.lake/packages  (has mathlib/, aesop/, ...)
#   OUT_DIR              destination software/ tree
set -euo pipefail
TC="$1"; PKGS="$2"; OUT="$3"
mkdir -p "$OUT/elan/toolchains" "$OUT/mathlib/packages"
cp -a "$TC" "$OUT/elan/toolchains/"
for p in aesop batteries Cli importGraph LeanSearchClient mathlib plausible proofwidgets Qq; do
  mkdir -p "$OUT/mathlib/packages/$p/.lake/build/lib"
  cp -a "$PKGS/$p/.lake/build/lib/lean" "$OUT/mathlib/packages/$p/.lake/build/lib/lean"
done
echo "software/ built at $OUT ($(du -sh "$OUT" | cut -f1))"
