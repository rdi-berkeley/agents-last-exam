#!/usr/bin/env bash
# RTG Tools (RealTimeGenomics) -> /opt/rtg-tools-3.12.1; provides `rtg` (incl. vcfeval)
# for WGS_Variant_Calling's benchmarking checkpoint. Self-contained Java app + bundled JRE.
set -euo pipefail
P=/opt/rtg-tools-3.12.1
if [ ! -x "$P/rtg" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/rtg.zip" \
    "https://github.com/RealTimeGenomics/rtg-tools/releases/download/3.12.1/rtg-tools-3.12.1-linux-x64.zip"
  mkdir -p /opt; unzip -q "$T/rtg.zip" -d /opt; rm -rf "$T"
  # release unzips to rtg-tools-3.12.1/ ; accept RTG usage non-interactively
  printf 'RTG_TALKBACK=false\nRTG_USAGE=false\n' >> "$P/rtg.cfg" 2>/dev/null || true
fi
ln -sf "$P/rtg" /usr/local/bin/rtg
"$P/rtg" version 2>&1 | grep -qi "3.12.1" || { echo "[pkg rtg-tools-3.12.1] FATAL: rtg not 3.12.1" >&2; exit 1; }
echo "[pkg rtg-tools-3.12.1] OK ($("$P/rtg" version 2>&1 | head -1))"
