#!/usr/bin/env bash
set -Eeuo pipefail

storage_dir="${STORAGE:-/storage}"
disk_name="${DISK_NAME:-data}"
disk_path="$storage_dir/$disk_name.qcow2"

if [[ ! -s "$disk_path" ]]; then
  echo "ALE QEMU runner requires a non-empty guest disk at $disk_path" >&2
  exit 64
fi

mkdir -p "$storage_dir"

# The upstream dockur startup treats this marker as an already-installed guest.
# ALE always supplies a pre-baked qcow2, for both Ubuntu and Windows.
touch "$storage_dir/windows.boot"

if [[ "${INSTALL_WINARENA_APPS,,}" == "true" ]]; then
  install_winarena_apps=true
else
  install_winarena_apps=false
fi
printf '{"INSTALL_WINARENA_APPS": %s}\n' "$install_winarena_apps" \
  > /oem/install_config.json

echo "Starting ALE QEMU guest from $disk_path"
echo "noVNC is exposed on container port 8006"
echo "CUA is expected on guest ${VM_NET_IP:-172.30.0.2}:${ALE_GUEST_CUA_PORT:-5000}"

# tini becomes PID 1 and forwards signals to the dockur startup process.
# Unlike the inherited entrypoint, no detached readiness loop can keep the
# container alive after QEMU exits.
exec /usr/bin/tini -s -- /run/entry.sh
