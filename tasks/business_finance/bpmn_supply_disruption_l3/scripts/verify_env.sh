#!/usr/bin/env bash
# ============================================================================
# Environment verification probe
#   task : business_finance/bpmn_supply_disruption_l3
#
# Same docker-based Flowable workflow as the other bpmn L3 task, but this one
# uses the single-container image flowable/all-in-one:6.5.0. Proves:
#   1. Docker daemon runs   2. image pull+run works
#   3. staged compose validates
#   4. flowable/all-in-one:6.5.0 boots and its process REST API answers on :8080
#
# Runs as uid 1000. Requires a PRIVILEGED container + network. Heavy (~1GB pull).
# ============================================================================
set -uo pipefail

CANON="${CANON:-/media/user/data/agenthle/business_finance/bpmn_supply_disruption_l3/base}"
COMPOSE="$CANON/input/starter_project/docker-compose.yml"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-360}"
REST="http://localhost:8080/flowable-task/process-api/repository/process-definitions"

fail() { echo "[verify] FATAL: $*" >&2; exit 1; }

echo "[verify] docker: $(docker --version 2>/dev/null)"
echo "[verify] compose: $(docker compose version 2>/dev/null | head -1)"

if ! docker info >/dev/null 2>&1; then
  echo "[verify] starting dockerd ..."
  sudo sh -c 'nohup dockerd >/var/log/dockerd.log 2>&1 &'
  for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
docker info >/dev/null 2>&1 || fail "docker daemon not reachable"
echo "[verify] docker daemon up."

docker run --rm hello-world >/dev/null 2>&1 || fail "cannot pull/run hello-world"
echo "[verify] image pull + run OK."

test -f "$COMPOSE" || fail "starter docker-compose.yml missing: $COMPOSE"
docker compose -f "$COMPOSE" config >/dev/null 2>&1 || fail "starter compose does not validate"
echo "[verify] starter compose validates."

cleanup() { docker compose -f "$COMPOSE" down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker compose -f "$COMPOSE" down -v --remove-orphans >/dev/null 2>&1 || true
docker rm -f flowable-supply-chain >/dev/null 2>&1 || true
echo "[verify] bringing up flowable/all-in-one:6.5.0 (pulls ~1GB) ..."
docker compose -f "$COMPOSE" up -d || fail "docker compose up failed"

echo "[verify] waiting up to ${BOOT_TIMEOUT}s for Flowable process REST on :8080 ..."
ok=0
for i in $(seq 1 $((BOOT_TIMEOUT/5))); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -u admin:test "$REST" 2>/dev/null); code=${code:-000}
  if [ "$code" = "200" ]; then ok=1; echo "[verify] Flowable REST answered 200 after $((i*5))s"; break; fi
  sleep 5
done
[ "$ok" = "1" ] || { echo "[verify] --- last 30 lines of flowable logs ---"; docker compose -f "$COMPOSE" logs --tail=30 2>/dev/null; fail "Flowable did not become ready within ${BOOT_TIMEOUT}s"; }

echo "[verify] PASS"
