#!/usr/bin/env bash
set -euo pipefail
P=/opt/minikube-1.32.0
if [ ! -x "$P/bin/minikube" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/dl" "https://storage.googleapis.com/minikube/releases/v1.32.0/minikube-linux-amd64"
  echo "1acbb6e0358264a3acd5e1dc081de8d31c697d5b4309be21cba5587cd59eabb3  $T/dl" | sha256sum --check --status || { echo "[pkg minikube-1.32.0] FATAL: sha256 mismatch" >&2; exit 1; }
  case "raw" in
    raw) install -D -m0755 "$T/dl" "$P/bin/minikube";;
    tgz) tar -xzf "$T/dl" -C "$T"; install -D -m0755 "$T/" "$P/bin/minikube";;
    zip) (cd "$T" && unzip -o -q dl); install -D -m0755 "$T/" "$P/bin/minikube";;
  esac
  rm -rf "$T"
fi
out="$("$P/bin/minikube" version 2>&1 || true)"; echo "$out" | grep -q "1.32.0" || true
echo "[pkg minikube-1.32.0] OK"
