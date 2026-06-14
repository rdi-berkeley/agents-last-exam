#!/usr/bin/env bash
# Docker Engine + Compose v2 plugin (provides /usr/bin/docker). DinD tasks need
# the sandbox to be privileged; this only installs the engine.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! docker --version >/dev/null 2>&1; then
  apt-get update; apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  [ -s /etc/apt/keyrings/docker.gpg ] || { curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg; chmod a+r /etc/apt/keyrings/docker.gpg; }
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-${VERSION_CODENAME}} stable" > /etc/apt/sources.list.d/docker.list
  apt-get update; apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  rm -rf /var/lib/apt/lists/*
fi
groupadd -f docker; usermod -aG docker kasm-user 2>/dev/null || true
docker --version >/dev/null 2>&1 || { echo "[pkg docker-ce] FATAL: docker missing" >&2; exit 1; }
echo "[pkg docker-ce] OK ($(docker --version))"
