#!/usr/bin/env bash
# Pre-pull the pinned OpenROAD Flow Scripts image (needs docker-ce + a running
# dockerd; starts one transiently if necessary).
set -euo pipefail
IMG="openroad/orfs@sha256:fd7751659f976aec05129d3272186aa9112ba1a790a7f184449910af4ec4c475"
docker --version >/dev/null 2>&1 || { echo "[pkg openroad-orfs-image] FATAL: docker-ce required first" >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
  nohup dockerd >/var/log/dockerd-install.log 2>&1 &
  for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
docker info >/dev/null 2>&1 || { echo "[pkg openroad-orfs-image] FATAL: dockerd not reachable" >&2; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "[pkg openroad-orfs-image] FATAL: image not present" >&2; exit 1; }
echo "[pkg openroad-orfs-image] OK"
