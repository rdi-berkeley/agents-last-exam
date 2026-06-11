#!/bin/bash
# Runs as root INSIDE the freshly-imported base container; the result is
# committed to the final image. Bakes the entrypoint and removes VM-host state
# that is stale or meaningless in a container.
set -u

# --- entrypoint (cua-server on :5000 behind Xvfb :0) ---
chmod +x /dockerstartup/vnc_startup.sh

# --- dirs excluded from the rootfs tar that the runtime needs back, with the
#     sticky perms docker would otherwise recreate them as root:0755 ---
mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
chmod 1777 /tmp
mkdir -p /var/tmp && chmod 1777 /var/tmp

# --- drop VM-host identity / config (regenerated or N/A in a container) ---
: > /etc/fstab                 2>/dev/null || true   # no VM disks to mount
rm -f /etc/netplan/*.yaml      2>/dev/null || true   # docker manages networking
rm -f /etc/ssh/ssh_host_*      2>/dev/null || true   # regen on first sshd start
: > /etc/machine-id            2>/dev/null || true   # regenerated on boot
rm -f /var/lib/dbus/machine-id 2>/dev/null || true
rm -rf /var/lib/cloud          2>/dev/null || true   # cloud-init state, if any

# --- drop baked GCS/gcloud credentials. The docker provider re-injects a fresh
#     SA key per container at runtime (/etc/agenthle/gcs-reader.json) and writes
#     /etc/boto.cfg itself, so nothing credential-bearing needs to ship baked. ---
rm -f  /etc/boto.cfg                                  2>/dev/null || true
rm -rf /home/user/.config/gcloud /root/.config/gcloud 2>/dev/null || true

# --- sanity: paths the ale-ubuntu22-docker Image entry promises must exist ---
echo "--- verify image-promised paths ---"
fail=0
for p in /usr/local/bin/node \
         /opt/cua-server/.venv/bin/python \
         /opt/ale-run/.venv/bin/python \
         /home/user/cua_mcp_server \
         /media/user/data/ale-data; do
  if [ -e "$p" ]; then echo "OK   $p"; else echo "MISS $p"; fail=1; fi
done
command -v Xvfb >/dev/null && echo "OK   Xvfb" || { echo "MISS Xvfb"; fail=1; }
/opt/cua-server/.venv/bin/python -c "import computer_server" 2>/dev/null \
  && echo "OK   computer_server importable" \
  || echo "WARN computer_server import failed without X (expected; entrypoint starts Xvfb)"

[ "$fail" = 0 ] && echo "CLEANUP_OK" || echo "CLEANUP_WARN: missing expected paths above"
