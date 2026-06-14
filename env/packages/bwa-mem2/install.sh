#!/usr/bin/env bash
# bwa-mem2 (precompiled) -> /opt + PATH.
set -euo pipefail
P=/opt/bwa-mem2-2.2.1
if [ ! -x "$P/bwa-mem2" ]; then
  command -v bzip2 >/dev/null 2>&1 || { apt-get update && apt-get install -y bzip2 && rm -rf /var/lib/apt/lists/*; }
  T="$(mktemp -d)"; curl --fail -sSL -o "$T/b.tbz2" "https://github.com/bwa-mem2/bwa-mem2/releases/download/v2.2.1/bwa-mem2-2.2.1_x64-linux.tar.bz2"
  mkdir -p "$P"; tar -xjf "$T/b.tbz2" -C "$P" --strip-components=1; rm -rf "$T"
fi
ln -sf "$P/bwa-mem2" /usr/local/bin/bwa-mem2
command -v bwa-mem2 >/dev/null 2>&1 || { echo "[pkg bwa-mem2] FATAL: missing" >&2; exit 1; }
echo "[pkg bwa-mem2] OK"
