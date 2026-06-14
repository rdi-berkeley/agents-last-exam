#!/usr/bin/env bash
# NCBI BLAST+ 2.15.0 -> /opt/ncbi-blast-2.15.0 (+ bin on /usr/local/bin for PATH).
set -euo pipefail
P=/opt/ncbi-blast-2.15.0
if [ ! -x "$P/bin/blastp" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/b.tgz" \
    "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.15.0/ncbi-blast-2.15.0+-x64-linux.tar.gz"
  mkdir -p "$P"; tar -xzf "$T/b.tgz" -C "$P" --strip-components=1; rm -rf "$T"
fi
for b in "$P"/bin/*; do ln -sf "$b" "/usr/local/bin/$(basename "$b")"; done
blastp -version 2>&1 | grep -q "2.15.0" || { echo "[pkg ncbi-blast-2.15.0] FATAL: not 2.15.0" >&2; exit 1; }
echo "[pkg ncbi-blast-2.15.0] OK ($(blastp -version 2>&1 | head -1))"
