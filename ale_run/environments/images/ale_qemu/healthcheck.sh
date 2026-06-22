#!/usr/bin/env bash
set -Eeuo pipefail

guest_ip="${VM_NET_IP:-172.30.0.2}"
guest_port="${ALE_GUEST_CUA_PORT:-5000}"

curl --fail --silent --show-error --max-time 2 \
  "http://$guest_ip:$guest_port/status" >/dev/null
