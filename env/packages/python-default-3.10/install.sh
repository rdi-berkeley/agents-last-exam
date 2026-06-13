#!/usr/bin/env bash
# /usr/bin/python -> python3.10 (the reference VM's python). Many task wrappers
# exec `python` or resolve it via `command -v python`; several RTENVs pin ==3.10.*.
set -euo pipefail
test -x /usr/bin/python3.10 || { echo "[pkg python-default-3.10] FATAL: /usr/bin/python3.10 missing from base" >&2; exit 1; }
[ -e /usr/bin/python ] || ln -sf /usr/bin/python3.10 /usr/bin/python
echo "[pkg python-default-3.10] OK ($(/usr/bin/python --version 2>&1))"
