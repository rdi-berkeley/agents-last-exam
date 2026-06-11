#!/bin/bash
# Baked into the image at /dockerstartup/vnc_startup.sh — the container
# entrypoint the ALE docker provider invokes:
#     docker run --entrypoint /dockerstartup/vnc_startup.sh <image> --wait
# (the trailing kasm-style "--wait" arg is ignored here).
#
# The GCE image starts cua-server from a systemd unit; a container has no init,
# so we reproduce that unit's behaviour as PID 1:
#   1. bring up a virtual X server on :0 (cua-server uses pynput / python-xlib)
#   2. exec cua-computer-server on port 5000
set -u

export HOME=/home/user
export DISPLAY=:0
export XAUTHORITY=/home/user/.Xauthority
export PATH=/opt/cua-server/.venv/bin:/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1

mkdir -p /tmp/.X11-unix 2>/dev/null || true
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
rm -f /tmp/.X0-lock 2>/dev/null || true
touch "$XAUTHORITY" 2>/dev/null || true

# Virtual X on :0 (-ac disables access control so local clients connect without
# an xauth cookie; -nolisten tcp keeps it local).
Xvfb :0 -screen 0 1920x1080x24 -ac -nolisten tcp >/tmp/xvfb.log 2>&1 &
for _ in $(seq 1 50); do
  [ -S /tmp/.X11-unix/X0 ] && break
  sleep 0.2
done

cd /opt/cua-server 2>/dev/null || cd /
exec /opt/cua-server/.venv/bin/python -m computer_server --port 5000
