#!/usr/bin/env bash
# Create /opt/cua-server/.venv (some task wrappers exec its python). Uses uv
# (no apt). Stdlib-only tasks just need a working interpreter there.
set -euo pipefail
command -v uv >/dev/null 2>&1 || { echo "[pkg cua-server-venv] FATAL: uv required" >&2; exit 1; }
if [ ! -x /opt/cua-server/.venv/bin/python ]; then
  mkdir -p /opt/cua-server; uv venv --python 3.10 /opt/cua-server/.venv
fi
test -x /opt/cua-server/.venv/bin/python || { echo "[pkg cua-server-venv] FATAL: venv python missing" >&2; exit 1; }
echo "[pkg cua-server-venv] OK ($(/opt/cua-server/.venv/bin/python --version 2>&1))"
