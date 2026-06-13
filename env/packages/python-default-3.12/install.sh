#!/usr/bin/env bash
# /usr/bin/python -> python3.12 (for tasks/runtimes pinned to 3.12).
set -euo pipefail
test -x /usr/bin/python3.12 || { echo "[pkg python-default-3.12] FATAL: /usr/bin/python3.12 missing from base" >&2; exit 1; }
[ -e /usr/bin/python ] || ln -sf /usr/bin/python3.12 /usr/bin/python
echo "[pkg python-default-3.12] OK ($(/usr/bin/python --version 2>&1))"
