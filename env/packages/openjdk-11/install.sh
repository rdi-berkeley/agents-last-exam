#!/usr/bin/env bash
# OpenJDK 11 (headless). Some tasks pin Java 11 (e.g. InterProScan's launcher
# requires java 11). Coexists with openjdk-17; the wrapper selects java-11.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! ls /usr/lib/jvm/java-11-openjdk-amd64/bin/java >/dev/null 2>&1; then
  apt-get update && apt-get install -y openjdk-11-jre-headless && rm -rf /var/lib/apt/lists/*
fi
test -x /usr/lib/jvm/java-11-openjdk-amd64/bin/java || { echo "[pkg openjdk-11] FATAL: java 11 missing" >&2; exit 1; }
echo "[pkg openjdk-11] OK ($(/usr/lib/jvm/java-11-openjdk-amd64/bin/java -version 2>&1 | head -1))"
