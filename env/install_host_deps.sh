#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' 'Usage: bash env/install_host_deps.sh' \
    'Install ALE host evaluation tools on Ubuntu 22.04+ or Debian 12+.' \
    'Then run uv sync --all-packages --locked and bash env/verify_host_env.sh.'
  exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments; use --help.\n' >&2; exit 2; }
[[ -r /etc/os-release ]] || { printf 'Unsupported host: expected Ubuntu or Debian.\n' >&2; exit 1; }
source /etc/os-release
case "${ID:-}:${VERSION_ID:-}" in
  ubuntu:22.04|ubuntu:24.04|ubuntu:26.04|debian:12|debian:13) ;;
  *) printf 'Unsupported host: %s %s. See env/README.md.\n' "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2; exit 1 ;;
esac
command -v python3 >/dev/null || { printf 'Install python3 before running this script.\n' >&2; exit 1; }
PACKAGES="$(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["apt_packages"]))' "$REPO_ROOT/env/host-deps.json")"
mapfile -t HOST_PACKAGES <<< "$PACKAGES"
SUDO=()
if ((EUID != 0)); then
  command -v sudo >/dev/null || { printf 'Root or sudo is required to install host system packages.\n' >&2; exit 1; }
  SUDO=(sudo)
fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${HOST_PACKAGES[@]}"
printf '%s\n' 'Host system packages installed. Continue as your normal user:' \
  '  uv sync --all-packages --locked' \
  '  bash env/verify_host_env.sh'
