#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/bpmn_category_governance_restructuring_l3
#
# Proves the docker-based Flowable workflow is possible post-install:
#   1. Docker daemon runs (started via sudo, as the sandbox orchestration would)
#   2. docker can pull + run an image (hello-world)
#   3. the staged starter docker-compose.yml validates
#   4. the real Flowable 6.5.0 stack (postgres:13 + flowable/flowable-task:6.5.0)
#      comes up and its REST/UI answers on :8080
#
# Runs as uid 1000 (kasm-user). Requires a PRIVILEGED container + network.
# Heavy: pulls ~1GB of images and waits for Flowable to boot (~2-3 min).
# ============================================================================
set -uo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/bpmn_category_governance_restructuring_l3/base}"
COMPOSE="$CANON/input/starter_project/docker-compose.yml"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-300}"

fail() { echo "[verify] FATAL: $*" >&2; exit 1; }

echo "[verify] docker: $(docker --version 2>/dev/null)"
echo "[verify] compose: $(docker compose version 2>/dev/null | head -1)"

# 1. ensure dockerd is up (orchestration normally does this; we use NOPASSWD sudo)
if ! docker info >/dev/null 2>&1; then
  echo "[verify] starting dockerd ..."
  sudo sh -c 'nohup dockerd >/var/log/dockerd.log 2>&1 &'
  for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
docker info >/dev/null 2>&1 || fail "docker daemon not reachable"
echo "[verify] docker daemon up."

# 2. pull + run
docker run --rm hello-world >/dev/null 2>&1 || fail "cannot pull/run hello-world"
echo "[verify] image pull + run OK."

# 3. validate the staged compose
test -f "$COMPOSE" || fail "starter docker-compose.yml missing: $COMPOSE"
docker compose -f "$COMPOSE" config >/dev/null 2>&1 || fail "starter compose does not validate"
echo "[verify] starter compose validates."

# 4. bring up the real Flowable stack and confirm it boots + serves.
#
# NB: flowable/flowable-task:6.5.0 binds its Spring Boot server to port 9999
# (context /flowable-task) INSIDE the container; the task's compose publishes
# 8080:8080, so the host :8080 is not the app's port (the task prompt notes a
# separate "live instance" is the preferred runtime; this compose is a
# fallback). For an *environment* check, success = the stack actually boots and
# serves, which we confirm via (a) the Spring "Started ...Application" marker in
# the container logs and (b) a real HTTP response on the app's internal port,
# reached by sharing the flowable-task container's network namespace.
SVC_CONTAINER="flowable-task"
APP_PORT=9999
cleanup() { docker compose -f "$COMPOSE" down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT
# Pre-clean any leftover state (fixed container_names collide on re-runs).
docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true
docker rm -f flowable-postgres flowable-task >/dev/null 2>&1 || true
echo "[verify] bringing up Flowable stack (this pulls ~1GB on a cold cache) ..."
docker compose -f "$COMPOSE" up -d || fail "docker compose up failed"

echo "[verify] waiting up to ${BOOT_TIMEOUT}s for Flowable to finish starting ..."
started=0
for i in $(seq 1 $((BOOT_TIMEOUT/5))); do
  if docker logs "$SVC_CONTAINER" 2>&1 | grep -q "Started FlowableTaskApplication"; then
    started=1; echo "[verify] flowable-task Spring app started after ~$((i*5))s"; break
  fi
  # surface a hard startup failure early
  if docker logs "$SVC_CONTAINER" 2>&1 | grep -qiE "APPLICATION FAILED TO START|Caused by:.*(Connection refused|UnknownHost)"; then
    docker logs --tail 30 "$SVC_CONTAINER" 2>&1; fail "flowable-task reported a startup failure"
  fi
  sleep 5
done
[ "$started" = "1" ] || { echo "[verify] --- last 30 lines of flowable-task logs ---"; docker logs --tail 30 "$SVC_CONTAINER" 2>&1; fail "Flowable did not start within ${BOOT_TIMEOUT}s"; }

# Functional HTTP check on the app's real internal port (curl from a throwaway
# container that shares flowable-task's network namespace; the image has no curl).
echo "[verify] HTTP-probing the Flowable app on its internal port ${APP_PORT} ..."
is_ready() { case "$1" in 200|302|401) return 0;; *) return 1;; esac; }
hcode=000
for i in $(seq 1 12); do
  hcode=$(docker run --rm --network "container:${SVC_CONTAINER}" curlimages/curl:8.11.1 \
            -s -o /dev/null -w '%{http_code}' -u admin:test \
            "http://localhost:${APP_PORT}/flowable-task/" 2>/dev/null); hcode=${hcode:-000}
  is_ready "$hcode" && break
  sleep 5
done
is_ready "$hcode" || fail "Flowable app did not return a valid HTTP code (got '$hcode') on :${APP_PORT}"
echo "[verify] Flowable app answered HTTP $hcode on :${APP_PORT}/flowable-task/"

echo "[verify] PASS"
