#!/usr/bin/env bash
# Thin shim: install this task's system deps via the shared package library,
# driven by `requiredSystemPackages` in ../task_card.json. See env/README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
exec "$ROOT/env/install_task_deps.sh" "$HERE/../task_card.json"
