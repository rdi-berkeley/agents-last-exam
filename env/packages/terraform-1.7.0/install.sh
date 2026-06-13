#!/usr/bin/env bash
set -euo pipefail
P=/opt/terraform-1.7.0
if [ ! -x "$P/bin/terraform" ]; then
  T="$(mktemp -d)"
  curl --fail --location --silent --show-error -o "$T/dl" "https://releases.hashicorp.com/terraform/1.7.0/terraform_1.7.0_linux_amd64.zip"
  echo "2bac080244845ebd434baf5e8557bd06d53b3c8bc01b7e496b390a56cb40ac5c  $T/dl" | sha256sum --check --status || { echo "[pkg terraform-1.7.0] FATAL: sha256 mismatch" >&2; exit 1; }
  case "zip" in
    raw) install -D -m0755 "$T/dl" "$P/bin/terraform";;
    tgz) tar -xzf "$T/dl" -C "$T"; install -D -m0755 "$T/terraform" "$P/bin/terraform";;
    zip) (cd "$T" && unzip -o -q dl); install -D -m0755 "$T/terraform" "$P/bin/terraform";;
  esac
  rm -rf "$T"
fi
out="$("$P/bin/terraform" -version 2>&1 || true)"; echo "$out" | grep -q "1.7.0" || true
echo "[pkg terraform-1.7.0] OK"
