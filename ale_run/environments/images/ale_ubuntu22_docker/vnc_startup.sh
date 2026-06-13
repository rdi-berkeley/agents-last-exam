#!/bin/bash
# Baked into the image at /dockerstartup/vnc_startup.sh — the container
# entrypoint the ALE docker provider invokes:
#     docker run --entrypoint /dockerstartup/vnc_startup.sh <image> --wait
# (the trailing kasm-style "--wait" arg is ignored here).
#
# Responsibilities (reproduce what the GCE image's systemd units do, since a
# container has no init):
#   1. nested Docker (DinD) for tasks whose eval runs `docker run` (openroad,
#      minikube/k8s, flowable compose) — best effort, see below
#   2. virtual X on :0 (cua-server uses pynput / python-xlib)
#   3. exec cua-computer-server on port 5000 as PID 1
set -u

export HOME=/home/user
export DISPLAY=:0
export XAUTHORITY=/home/user/.Xauthority
export PATH=/opt/cua-server/.venv/bin:/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1
# Disable cua's PostHog/OpenTelemetry phone-home. A benchmark sandbox shouldn't
# emit usage data, and — decisively — the telemetry init does a network call to
# eu.i.posthog.com that, at concurrency, blocks computer_server from binding its
# port until well past the docker provider's 120s cua-ready timeout (every
# acquire then fails). Off → cua-server comes up in seconds regardless of fan-out.
export CUA_TELEMETRY_ENABLED=false
export CUA_TELEMETRY_DISABLED=1

# --- 1. nested Docker (DinD) -------------------------------------------------
# Some task evals run `docker run/build` INSIDE the sandbox. A nested dockerd
# cannot reuse a baked overlay2 /var/lib/docker (overlay2-on-overlay2 is
# unsupported under rootless+privileged), so we run a FRESH daemon on
# fuse-overlayfs at /var/lib/dind.
#
# The nested image store (openroad/orfs, kicbase, flowable) is BAKED into
# /var/lib/dind at image-build time (commit-bake) — verified to survive
# commit/push/pull — so a pulled image needs NO load and NO network: the daemon
# comes up with the images already present (incl. orfs' @sha256 RepoDigest, so
# `docker run openroad/orfs@sha256:...` resolves locally). This is the whole
# point: others pull the image and it just works, zero per-start overhead.
#
# Fallback for an UNBAKED image (empty /var/lib/dind but tarballs present at
# /opt/ale-docker-images/): background-load the tarballs + restore digests via
# pull-refs.txt. Kept so the build/bake step itself can populate the store; the
# load runs in the BACKGROUND so it never blocks cua-server startup past the
# docker provider's 120s ready timeout. Entirely best-effort: without
# --privileged the daemon won't start and non-docker tasks are unaffected.
#
# GATED OFF BY DEFAULT (ALE_ENABLE_DIND != 1). Only the ~4 tasks whose eval runs
# nested docker (openroad, k8s_migration, bpmn ×2) need it, and starting a
# fuse-overlayfs dockerd in EVERY container is pure overhead — at concurrency it
# is an I/O/FUSE storm that starves cua-server startup past the 120s ready
# timeout and fails acquires. The provider sets ALE_ENABLE_DIND=1 only for tasks
# that need it (config knob `enable_dind`). Those 4 tasks are currently excluded
# from the Docker provider — see README "Deferred: DinD tasks".
if [ "${ALE_ENABLE_DIND:-0}" = "1" ] && command -v dockerd >/dev/null; then
  sudo mkdir -p /run /var/lib/dind 2>/dev/null || true
  sudo dockerd --data-root=/var/lib/dind --storage-driver=fuse-overlayfs \
       --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 &
  (
    for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    if docker info >/dev/null 2>&1; then
      if [ -n "$(docker images -q 2>/dev/null)" ]; then
        # Baked store already populated — nothing to load.
        :
      elif [ -d /opt/ale-docker-images ]; then
        for t in /opt/ale-docker-images/*.tar; do
          [ -f "$t" ] && docker load -i "$t" >/dev/null 2>&1
        done
        if [ -f /opt/ale-docker-images/pull-refs.txt ]; then
          while read -r ref; do
            [ -n "$ref" ] && docker pull "$ref" >/dev/null 2>&1
          done < /opt/ale-docker-images/pull-refs.txt
        fi
      fi
    fi
    touch /tmp/dind-images-ready 2>/dev/null || true
  ) >/tmp/dind-load.log 2>&1 &
fi

# --- 2. virtual X on :0 ------------------------------------------------------
# Screen size comes from the docker provider via ALE_SCREEN_RESOLUTION (WxH,
# default 1024x768); 24-bit depth is appended here.
mkdir -p /tmp/.X11-unix 2>/dev/null || true
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
rm -f /tmp/.X0-lock 2>/dev/null || true
touch "$XAUTHORITY" 2>/dev/null || true
Xvfb :0 -screen 0 "${ALE_SCREEN_RESOLUTION:-1024x768}x24" -ac -nolisten tcp >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do
  [ -S /tmp/.X11-unix/X0 ] && break
  sleep 0.2
done

# --- 3. cua-computer-server on :5000 (exec as PID 1) -------------------------
cd /opt/cua-server 2>/dev/null || cd /
exec /opt/cua-server/.venv/bin/python -m computer_server --port 5000
