#!/usr/bin/env bash
set -euo pipefail
P=/opt/helm-3.14.0
if [ ! -x "$P/bin/helm" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/dl" "https://get.helm.sh/helm-v3.14.0-linux-amd64.tar.gz"
  echo "f43e1c3387de24547506ab05d24e5309c0ce0b228c23bd8aa64e9ec4b8206651  $T/dl" | sha256sum --check --status || { echo "[pkg helm-3.14.0] FATAL: sha256 mismatch" >&2; exit 1; }
  case "tgz" in
    raw) install -D -m0755 "$T/dl" "$P/bin/helm";;
    tgz) tar -xzf "$T/dl" -C "$T"; install -D -m0755 "$T/linux-amd64/helm" "$P/bin/helm";;
    zip) (cd "$T" && unzip -o -q dl); install -D -m0755 "$T/linux-amd64/helm" "$P/bin/helm";;
  esac
  rm -rf "$T"
fi
out="$("$P/bin/helm" version --short 2>&1 || true)"; echo "$out" | grep -q "3.14.0" || true
echo "[pkg helm-3.14.0] OK"
