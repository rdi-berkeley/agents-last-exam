#!/usr/bin/env bash
# Install a task's system dependencies from its task_card.json.
# Usage: env/install_task_deps.sh <path/to/task_card.json>
# Reads .requiredSystemPackages[] and runs env/packages/<id>/install.sh for each.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD="${1:?usage: install_task_deps.sh <task_card.json>}"
test -f "$CARD" || { echo "[install_task_deps] FATAL: card not found: $CARD" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { apt-get update && apt-get install -y jq && rm -rf /var/lib/apt/lists/*; }
mapfile -t PKGS < <(jq -r '.requiredSystemPackages[]? // empty' "$CARD")
if [ "${#PKGS[@]}" -eq 0 ]; then
  echo "[install_task_deps] no requiredSystemPackages declared — nothing to install."
  exit 0
fi
echo "[install_task_deps] packages: ${PKGS[*]}"
for p in "${PKGS[@]}"; do
  inst="$SCRIPT_DIR/packages/$p/install.sh"
  test -x "$inst" || { echo "[install_task_deps] FATAL: unknown package '$p' ($inst)" >&2; exit 1; }
  echo "===== installing package: $p ====="
  bash "$inst"
done
echo "[install_task_deps] all packages installed OK"
