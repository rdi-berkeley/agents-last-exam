#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-sync --project "$HERE/../../../.." python "$HERE/local_build.py" "$@"
