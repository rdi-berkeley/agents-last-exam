#!/usr/bin/env bash
# bwa-mem2 (precompiled) -> /opt + PATH.
set -euo pipefail
P=/opt/bwa-mem2-2.2.1
if [ ! -x "$P/bwa-mem2" ]; then
  command -v bzip2 >/dev/null 2>&1 || { apt-get update && apt-get install -y bzip2 && rm -rf /var/lib/apt/lists/*; }
  T="$(mktemp -d)"; curl --fail -sSL -o "$T/b.tbz2" "https://github.com/bwa-mem2/bwa-mem2/releases/download/v2.2.1/bwa-mem2-2.2.1_x64-linux.tar.bz2"
  mkdir -p "$P"; tar --no-same-owner -xjf "$T/b.tbz2" -C "$P" --strip-components=1; rm -rf "$T"
fi
# bwa-mem2's `bwa-mem2` is a dispatcher that execs an arch-specific sibling
# (bwa-mem2.avx2/.avx512bw/.sse41/...). Two pitfalls: (1) the siblings must be on
# PATH next to the dispatcher, and (2) the dispatcher can mis-pick avx512bw on a
# CPU that flags AVX512 but can't actually run that build. So: symlink all siblings,
# then point the canonical `bwa-mem2` at the FASTEST variant that actually runs here
# (best->worst), bypassing the flaky auto-dispatch. Dev/eval GCP VMs are avx2-class.
for b in "$P"/bwa-mem2.*; do ln -sf "$b" "/usr/local/bin/$(basename "$b")"; done
BEST=""
for v in avx512bw avx2 sse42 sse41; do
  if [ -x "$P/bwa-mem2.$v" ] && "$P/bwa-mem2.$v" version 2>/dev/null | grep -q "2.2.1"; then BEST="$v"; break; fi
done
[ -n "$BEST" ] || { echo "[pkg bwa-mem2] FATAL: no runnable bwa-mem2 variant on this CPU" >&2; exit 1; }
ln -sf "$P/bwa-mem2.$BEST" /usr/local/bin/bwa-mem2
out="$(bwa-mem2 version 2>/dev/null || true)"
echo "$out" | grep -q "2.2.1" || { echo "[pkg bwa-mem2] FATAL: 'bwa-mem2 version' did not run (got: $out)" >&2; exit 1; }
echo "[pkg bwa-mem2] OK (bwa-mem2 $out, variant=$BEST)"
