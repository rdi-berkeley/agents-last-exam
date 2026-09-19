#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ALE_RUNTIME_MANIFEST:?explicit audited runtime manifest required}"
[ "$(id -u)" = 0 ] || { echo 'export must run as root on the disposable builder' >&2; exit 1; }
LIST="$(mktemp /var/lib/ale-export-manifest.XXXXXX)"
trap 'rm -f "$LIST"' EXIT
python3 "$HERE/rootfs_policy.py" "$ALE_RUNTIME_MANIFEST" > "$LIST"
if [ "${ALE_EXPORT_RAW:-0}" = 1 ]; then
  tar --numeric-owner --no-recursion --null --verbatim-files-from -C / -T "$LIST" -cf -
else
  tar --numeric-owner --no-recursion --null --verbatim-files-from -C / -T "$LIST" -cf - \
    | zstd -T2 -3 -c
fi
