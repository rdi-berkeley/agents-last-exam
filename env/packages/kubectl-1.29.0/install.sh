#!/usr/bin/env bash
set -euo pipefail
P=/opt/kubectl-1.29.0
if [ ! -x "$P/bin/kubectl" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/dl" "https://dl.k8s.io/release/v1.29.0/bin/linux/amd64/kubectl"
  echo "0e03ab096163f61ab610b33f37f55709d3af8e16e4dcc1eb682882ef80f96fd5  $T/dl" | sha256sum --check --status || { echo "[pkg kubectl-1.29.0] FATAL: sha256 mismatch" >&2; exit 1; }
  case "raw" in
    raw) install -D -m0755 "$T/dl" "$P/bin/kubectl";;
    tgz) tar -xzf "$T/dl" -C "$T"; install -D -m0755 "$T/" "$P/bin/kubectl";;
    zip) (cd "$T" && unzip -o -q dl); install -D -m0755 "$T/" "$P/bin/kubectl";;
  esac
  rm -rf "$T"
fi
out="$("$P/bin/kubectl" version --client=true 2>&1 || true)"; echo "$out" | grep -q "1.29.0" || true
echo "[pkg kubectl-1.29.0] OK"
