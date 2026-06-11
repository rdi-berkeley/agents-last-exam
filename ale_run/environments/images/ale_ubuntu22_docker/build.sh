#!/usr/bin/env bash
# Build the `ale-ubuntu22-docker` container image from the `ale-ubuntu22` GCE
# sandbox VM, without rebuilding any packages: a container only needs the
# userspace rootfs (the kernel is the host's), so we export the VM's root
# filesystem and import it as a single layer, then bake a container entrypoint.
#
# Phases (resumable — a present rootfs tar is reused unless ALE_FORCE_EXPORT=1):
#   1. export   ssh <vm> tar(/) | zstd            -> $WORKDIR/rootfs.tar.zst
#   2. import   zstd -dc | remap_uids.py | docker import   -> $IMAGE-base
#   3. finalize run base, bake entrypoint + cleanup, commit -> $IMAGE
#   4. smoke    boot it the way the provider does, poll cua-server /status
#
# Config via env (defaults target the dev-ubuntu22 box in us-west2-a):
#   ALE_BUILD_VM / ALE_BUILD_ZONE / ALE_BUILD_SSH_USER / ALE_BUILD_SSH_KEY
#   ALE_BUILD_IMAGE   final tag (default ale-ubuntu22-docker:latest)
#   ALE_BUILD_WORKDIR scratch dir for the rootfs tar (needs ~100GB free)
#   ALE_FORCE_EXPORT=1   re-export even if the tar exists
#   ALE_KEEP_VM=1        do not stop the VM afterwards even if we started it
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VM="${ALE_BUILD_VM:-dev-ubuntu22}"
ZONE="${ALE_BUILD_ZONE:-us-west2-a}"
SSH_USER="${ALE_BUILD_SSH_USER:-weichenzhang}"
SSH_KEY="${ALE_BUILD_SSH_KEY:-$HOME/.ssh/google_compute_engine}"
IMAGE="${ALE_BUILD_IMAGE:-ale-ubuntu22-docker:latest}"
BASE="${IMAGE%%:*}:base"
WORKDIR="${ALE_BUILD_WORKDIR:-$HOME/.cache/ale-docker-build}"
TAR="$WORKDIR/rootfs.tar.zst"

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20)

log() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORKDIR"

# --- VM lifecycle: ensure it is RUNNING, remember whether we started it -------
started=0
status="$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>/dev/null || true)"
[ -n "$status" ] || die "VM $VM not found in zone $ZONE"
if [ "$status" != "RUNNING" ]; then
  log "starting $VM (was $status)"
  gcloud compute instances start "$VM" --zone "$ZONE" >/dev/null
  started=1
fi
IP="$(gcloud compute instances describe "$VM" --zone "$ZONE" \
       --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
[ -n "$IP" ] || die "no external IP for $VM"

restore_vm() {
  if [ "$started" = 1 ] && [ "${ALE_KEEP_VM:-0}" != 1 ]; then
    log "stopping $VM (restore prior state)"
    gcloud compute instances stop "$VM" --zone "$ZONE" >/dev/null || true
  fi
}
trap restore_vm EXIT

# wait for SSH. Prime the key via `gcloud compute ssh` first: it propagates
# our public key to the instance (metadata / OS Login) and waits for sshd, so
# the subsequent raw ssh (used for a clean binary tar stream) can connect.
log "priming SSH on $SSH_USER@$IP"
for _ in $(seq 1 40); do
  gcloud compute ssh "$VM" --zone "$ZONE" --command true >/dev/null 2>&1 && break
  sleep 5
done
ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" true 2>/dev/null \
  || gcloud compute ssh "$VM" --zone "$ZONE" --command true >/dev/null 2>&1 \
  || die "cannot SSH to $VM ($SSH_USER@$IP)"

# --- phase 1: export rootfs --------------------------------------------------
if [ -s "$TAR" ] && [ "${ALE_FORCE_EXPORT:-0}" != 1 ]; then
  log "phase 1 export: reusing $TAR ($(du -h "$TAR" | cut -f1)) — set ALE_FORCE_EXPORT=1 to redo"
else
  log "phase 1 export: $SSH_USER@$IP rootfs -> $TAR"
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'bash -s' < "$HERE/export_rootfs.sh" > "$TAR"
  rc="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" 'cat /tmp/ale_export_tar.rc 2>/dev/null || echo 99')"
  case "$rc" in
    0|1) echo "tar exit $rc (ok)";;
    *)   die "tar exit $rc (fatal) — see /tmp/ale_export_tar.err on $VM";;
  esac
  echo "exported $(du -h "$TAR" | cut -f1) compressed"
fi

# --- phase 2: remap uids + docker import ------------------------------------
log "phase 2 import: $TAR -> docker image $BASE (uid remap inline)"
docker rmi -f "$BASE" 2>/dev/null || true
zstd -dc "$TAR" | python3 "$HERE/remap_uids.py" | docker import - "$BASE"

# --- phase 3: bake entrypoint + cleanup, commit ------------------------------
log "phase 3 finalize: bake entrypoint + cleanup -> commit $IMAGE"
cid="$(docker run -d --user 0 "$BASE" sleep infinity)"
cleanup_build() { docker rm -f "$cid" >/dev/null 2>&1 || true; }
trap 'cleanup_build; restore_vm' EXIT

docker exec "$cid" mkdir -p /dockerstartup
docker cp "$HERE/vnc_startup.sh" "$cid:/dockerstartup/vnc_startup.sh"
docker cp "$HERE/cleanup.sh"     "$cid:/root/cleanup.sh"
docker exec "$cid" bash /root/cleanup.sh

docker commit \
  --change 'USER user' \
  --change 'WORKDIR /home/user' \
  --change 'ENV HOME=/home/user' \
  --change 'ENV PATH=/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --change 'CMD ["/bin/bash"]' \
  "$cid" "$IMAGE" >/dev/null
cleanup_build
trap restore_vm EXIT
echo "committed $IMAGE ($(docker image inspect "$IMAGE" --format '{{.Size}}' | numfmt --to=iec))"

# --- phase 4: smoke — boot it exactly as the docker provider does ------------
log "phase 4 smoke: boot $IMAGE, poll cua-server /status"
sid="$(docker run -d --rm -p 0:5000 --shm-size=2g \
         --entrypoint /dockerstartup/vnc_startup.sh "$IMAGE" --wait)"
smoke_clean() { docker rm -f "$sid" >/dev/null 2>&1 || true; }
trap 'smoke_clean; restore_vm' EXIT
hp="$(docker inspect --format '{{(index (index .NetworkSettings.Ports "5000/tcp") 0).HostPort}}' "$sid")"
ok=
for _ in $(seq 1 30); do
  r="$(curl -s -m 4 "http://localhost:$hp/status" 2>/dev/null || true)"
  [ -n "$r" ] && { ok="$r"; break; }
  sleep 3
done
smoke_clean
trap restore_vm EXIT
[ -n "$ok" ] || die "cua-server did not become ready"
echo "cua-server /status: $ok"

log "DONE: built $IMAGE"
