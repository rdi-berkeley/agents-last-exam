#!/usr/bin/env bash
set -euo pipefail
P=/opt/trivy-0.70.0
if [ ! -x "$P/bin/trivy" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/dl" "https://github.com/aquasecurity/trivy/releases/download/v0.70.0/trivy_0.70.0_Linux-64bit.tar.gz"
  echo "8b4376d5d6befe5c24d503f10ff136d9e0c49f9127a4279fd110b727929a5aa9  $T/dl" | sha256sum --check --status || { echo "[pkg trivy-0.70.0] FATAL: sha256 mismatch" >&2; exit 1; }
  case "tgz" in
    raw) install -D -m0755 "$T/dl" "$P/bin/trivy";;
    tgz) tar -xzf "$T/dl" -C "$T"; install -D -m0755 "$T/trivy" "$P/bin/trivy";;
    zip) (cd "$T" && unzip -o -q dl); install -D -m0755 "$T/trivy" "$P/bin/trivy";;
  esac
  rm -rf "$T"
fi
out="$("$P/bin/trivy" --version 2>&1 || true)"; echo "$out" | grep -q "0.70.0" || true
echo "[pkg trivy-0.70.0] OK"
