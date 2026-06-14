#!/usr/bin/env bash
# System -dev libraries needed to compile common CRAN/Bioconductor R packages
# (curl/xml/ssl for httr/xml2; font/freetype/png/tiff/jpeg for ggplot2/ragg;
# fontconfig/harfbuzz/fribidi for systemfonts/textshaping).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
PKGS="libcurl4-openssl-dev libxml2-dev libssl-dev libfontconfig1-dev libharfbuzz-dev \
libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev zlib1g-dev libbz2-dev liblzma-dev"
need=0; for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
[ "$need" = "1" ] && { apt-get update && apt-get install -y $PKGS && rm -rf /var/lib/apt/lists/*; }
echo "[pkg r-build-deps] OK"
