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

# --- 1. nested Docker (DinD) -------------------------------------------------
# Some task evals run `docker run/build` INSIDE the sandbox. A nested dockerd
# cannot reuse a baked overlay2 /var/lib/docker (overlay2-on-overlay2 is
# unsupported under rootless+privileged), so we run a FRESH daemon on
# fuse-overlayfs and load the image tarballs baked at /opt/ale-docker-images/.
# `docker save` drops the @digest reference, so after loading we re-`pull` each
# digest-pinned ref listed in pull-refs.txt — that only fetches the (tiny)
# manifest and reuses the just-loaded layers, restoring the RepoDigest so an
# eval's `docker run image@sha256:...` resolves locally without a full re-pull.
# Entirely best-effort: if the container isn't --privileged the daemon won't
# start and non-docker tasks are unaffected.
if command -v dockerd >/dev/null && [ -d /opt/ale-docker-images ]; then
  sudo mkdir -p /run /var/lib/dind 2>/dev/null || true
  sudo dockerd --data-root=/var/lib/dind --storage-driver=fuse-overlayfs \
       --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
  if docker info >/dev/null 2>&1; then
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

# --- 2. virtual X on :0 ------------------------------------------------------
mkdir -p /tmp/.X11-unix 2>/dev/null || true
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
rm -f /tmp/.X0-lock 2>/dev/null || true
touch "$XAUTHORITY" 2>/dev/null || true
Xvfb :0 -screen 0 1920x1080x24 -ac -nolisten tcp >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do
  [ -S /tmp/.X11-unix/X0 ] && break
  sleep 0.2
done

# --- 3. cua-computer-server on :5000 (exec as PID 1) -------------------------
cd /opt/cua-server 2>/dev/null || cd /
exec /opt/cua-server/.venv/bin/python -m computer_server --port 5000
