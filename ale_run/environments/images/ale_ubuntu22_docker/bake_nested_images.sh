#!/bin/bash
# bake_nested_images.sh — bake the nested-Docker image store into the image.
#
# A few task evals run `docker` INSIDE the sandbox (see the README "nested
# Docker (DinD)"). Rather than load 5+ GB of image tarballs on every container
# start, we pre-populate the nested daemon's fuse-overlayfs store at
# /var/lib/dind ONCE and `docker commit` it into the image — so a pulled
# container has the nested images already present (zero per-start load).
# (Validated: the fuse-overlayfs store survives commit -> push -> pull intact.)
#
#   ./bake_nested_images.sh [SRC_TAG] [OUT_TAG]
#       SRC_TAG  base image to bake into  (default ale-ubuntu22-docker:base)
#       OUT_TAG  result tag               (default ale-ubuntu22-docker:latest)
#
# Requires: rootless+privileged docker on the host; outbound network (to pull
# the nested images). Run AFTER build.sh has produced the base image.
set -euo pipefail

SRC_TAG="${1:-ale-ubuntu22-docker:base}"
OUT_TAG="${2:-ale-ubuntu22-docker:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
C=ale-bake-$$

# The nested images each consuming task needs. orfs is pinned BY DIGEST (the
# openroad eval does `docker run openroad/orfs@sha256:...`), so it must keep its
# RepoDigest; kicbase (minikube) and flowable (bpmn) are used by tag.
ORFS_DIGEST="sha256:fd7751659f976aec05129d3272186aa9112ba1a790a7f184449910af4ec4c475"
NESTED_IMAGES=(
  "openroad/orfs@${ORFS_DIGEST}"
  "gcr.io/k8s-minikube/kicbase:v0.0.42"
  "flowable/all-in-one:6.5.0"
)

cleanup() { docker rm -f "$C" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> boot $SRC_TAG (privileged, DinD on) as $C"
docker run -d --privileged --name "$C" -e ALE_ENABLE_DIND=1 \
  --entrypoint /bin/bash "$SRC_TAG" -lc 'sleep infinity' >/dev/null

echo "==> start nested dockerd on fuse-overlayfs"
docker exec "$C" bash -lc '
  sudo mkdir -p /run /var/lib/dind
  sudo dockerd --data-root=/var/lib/dind --storage-driver=fuse-overlayfs \
       --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
  docker info --format "    inner storage-driver: {{.Driver}}"
'

echo "==> pull nested images into the store (orfs by digest restores its RepoDigest)"
for ref in "${NESTED_IMAGES[@]}"; do
  echo "    $ref"
  docker exec "$C" bash -lc "export DOCKER_HOST=unix:///var/run/docker.sock; docker pull -q '$ref' >/dev/null"
done
docker exec "$C" bash -lc 'export DOCKER_HOST=unix:///var/run/docker.sock; docker images --digests --format "    baked: {{.Repository}}:{{.Tag}} {{.Digest}}"'

echo "==> stop nested dockerd cleanly (consistent store for commit)"
docker exec "$C" bash -lc 'sudo pkill -TERM dockerd; for _ in $(seq 1 15); do pgrep dockerd >/dev/null || break; sleep 1; done'

echo "==> commit populated /var/lib/dind, then overlay the entrypoint"
docker commit "$C" "${OUT_TAG}-baked-tmp" >/dev/null
DOCKER_BUILDKIT=1 docker build -t "$OUT_TAG" -f - "$HERE" <<DF
FROM ${OUT_TAG}-baked-tmp
COPY --chmod=0755 vnc_startup.sh /dockerstartup/vnc_startup.sh
DF
docker rmi "${OUT_TAG}-baked-tmp" >/dev/null 2>&1 || true

echo "==> done: $OUT_TAG  (nested images baked; entrypoint skips load when /var/lib/dind is populated)"
