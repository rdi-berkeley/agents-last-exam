#!/usr/bin/env bash
# sbt 1.9.9 -> /opt/sbt-1.9.9 (needs a JDK; declare jdk-default alongside).
set -euo pipefail
P=/opt/sbt-1.9.9
if [ ! -x "$P/bin/sbt" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/sbt.tgz" \
    "https://github.com/sbt/sbt/releases/download/v1.9.9/sbt-1.9.9.tgz"
  mkdir -p "$P"; tar -xzf "$T/sbt.tgz" -C "$P" --strip-components=1; rm -rf "$T"
fi
test -x "$P/bin/sbt" || { echo "[pkg sbt-1.9.9] FATAL: sbt missing" >&2; exit 1; }
echo "[pkg sbt-1.9.9] OK"
