#!/usr/bin/env bash
# Headless JRE (e.g. for sbt / JVM tools)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
need=0; for p in openjdk-17-jre-headless; do dpkg -s "$p" >/dev/null 2>&1 || need=1; done
if [ "$need" = "1" ]; then
  echo "[pkg jdk-default] installing: openjdk-17-jre-headless"
  apt-get update && apt-get install -y openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
fi
echo "[pkg jdk-default] OK"
