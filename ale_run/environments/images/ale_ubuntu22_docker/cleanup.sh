#!/bin/bash
# Runs as root INSIDE the freshly-imported base container; the result is
# committed to the final image. Bakes the entrypoint and removes VM-host state
# that is stale or meaningless in a container.
set -u

# --- entrypoint (cua-server on :5000 behind Xvfb :0) ---
chmod +x /dockerstartup/entrypoint.sh

# --- desktop: install XFCE so the container has a real window manager + panel.
#     The VM brings its desktop up via gdm/GNOME under systemd; a container has
#     neither, so without a WM the Xvfb display is a bare (black) root window with
#     no window management. XFCE is X11-only and needs no systemd-logind, so the
#     entrypoint can start it directly. Baked here (not in the entrypoint) so it
#     is installed once, not on every per-task container start. ---
export DEBIAN_FRONTEND=noninteractive
if ! command -v startxfce4 >/dev/null 2>&1; then
  # the rootfs export drops /var/cache and /var/log; recreate apt's dirs or it
  # errors ("archives/partial is missing", "/var/log/apt/ missing").
  mkdir -p /var/cache/apt/archives/partial /var/lib/apt/lists/partial /var/log/apt
  apt-get update -qq \
  && apt-get install -y --no-install-recommends \
       xfce4-session xfwm4 xfce4-panel xfdesktop4 xfce4-settings xfconf \
       xfce4-terminal dbus-x11 x11-xserver-utils \
  && apt-get clean && rm -rf /var/lib/apt/lists/* \
  || echo "WARN: XFCE install failed (desktop will fall back to bare Xvfb)"
fi

# --- Chrome in a container: GUI tasks launch google-chrome, but the
#     rootfs-export image breaks it. Chrome's zygote sandbox needs
#     user-namespace/CAP the container doesn't grant, so a bare launch dies
#     (zygote FATAL → defunct zombie, app never opens); it needs --no-sandbox
#     (the container is the boundary). We also skip the first-run wizard and hide
#     the "unsupported flag" infobar (--test-type).
#     Wrap the REAL launcher (/opt/google/chrome/google-chrome) — not a PATH
#     shim — so EVERY caller gets the flags: `google-chrome` by name, the
#     absolute /usr/bin/google-chrome[-stable] symlinks, and the XFCE menu/dock
#     .desktop (exo-open), which all resolve here. ---
rm -f /usr/local/bin/google-chrome /usr/local/bin/google-chrome-stable 2>/dev/null || true  # old PATH shim
if [ -e /opt/google/chrome/google-chrome ] && [ ! -e /opt/google/chrome/google-chrome.real ]; then
  mv /opt/google/chrome/google-chrome /opt/google/chrome/google-chrome.real
  cat > /opt/google/chrome/google-chrome <<'CHROME'
#!/bin/bash
exec /opt/google/chrome/google-chrome.real --no-sandbox --no-first-run --no-default-browser-check --disable-gpu --test-type "$@"
CHROME
  chmod +x /opt/google/chrome/google-chrome
fi

# --- stale dev-VM runtime locks baked by the rootfs export (apps were running
#     on the dev VM at export time): a Chrome SingletonLock and a LibreOffice
#     .lock both point at the VM hostname, making a fresh launch think another
#     instance owns the profile. Drop them. ---
rm -f /home/user/.config/google-chrome/Singleton*  2>/dev/null || true
rm -f /home/user/.config/libreoffice/*/.lock        2>/dev/null || true

# --- file manager + default-application helpers: the bare XFCE install ships no
#     file manager and no "preferred application" config, so the panel/menu
#     category shortcuts (Web Browser / Terminal Emulator / File Manager) fail
#     with "Failed to execute default …" / "Choose Preferred Application". Install
#     Thunar and wire the exo-open defaults to the apps that ARE installed. ---
if ! command -v thunar >/dev/null 2>&1; then
  mkdir -p /var/cache/apt/archives/partial /var/lib/apt/lists/partial /var/log/apt
  apt-get update -qq && apt-get install -y --no-install-recommends thunar \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    || echo "WARN: thunar install failed"
fi
install -d -o user -g user /home/user/.config/xfce4
cat > /home/user/.config/xfce4/helpers.rc <<'HELPERS'
TerminalEmulator=xfce4-terminal
FileManager=thunar
WebBrowser=google-chrome
HELPERS
chown user:user /home/user/.config/xfce4/helpers.rc

# --- dirs excluded from the rootfs tar that the runtime needs back, with the
#     sticky perms docker would otherwise recreate them as root:0755 ---
mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
chmod 1777 /tmp
mkdir -p /var/tmp && chmod 1777 /var/tmp

# --- task_data_root: this is a DATA-LESS image (the ~146GB of task data is NOT
#     baked — excluded from the rootfs tar). Task data is supplied at runtime by
#     the `local:<dir>` task_data source (docker cp from the host) into this dir,
#     so ship it as an empty mount point. ---
mkdir -p /media/user/data/agenthle && chown -R user:user /media/user/data

# --- drop VM-host identity / config (regenerated or N/A in a container) ---
: > /etc/fstab                 2>/dev/null || true   # no VM disks to mount
rm -f /etc/netplan/*.yaml      2>/dev/null || true   # docker manages networking
rm -f /etc/ssh/ssh_host_*      2>/dev/null || true   # regen on first sshd start
# machine-id: on the VM systemd regenerates this at boot, but a container has no
# systemd — an empty/missing id makes D-Bus/dconf/desktop apps warn or misbehave.
# Bake a fresh valid one now (dbus-uuidgen, no systemd needed) + the dbus symlink.
# rm first: `--ensure` only creates when ABSENT (it won't replace an empty file).
rm -f /etc/machine-id /var/lib/dbus/machine-id
dbus-uuidgen --ensure=/etc/machine-id
mkdir -p /var/lib/dbus && ln -sf /etc/machine-id /var/lib/dbus/machine-id
rm -rf /var/lib/cloud          2>/dev/null || true   # cloud-init state, if any

# --- drop baked GCS/gcloud credentials. The docker provider re-injects a fresh
#     SA key per container at runtime (/etc/agenthle/gcs-reader.json) and writes
#     /etc/boto.cfg itself, so nothing credential-bearing needs to ship baked. ---
rm -f  /etc/boto.cfg                                  2>/dev/null || true
rm -rf /home/user/.config/gcloud /root/.config/gcloud 2>/dev/null || true

# Belt-and-suspenders: scrub any baked credentials/secrets that should only ever
# be injected at runtime (the rootfs tar already excludes these; this is a second
# line of defence so a public image never ships a real key). The framework/
# deployer re-injects fresh creds per run.
rm -rf /home/user/.config/agenthle-artifacts /root/.config/agenthle-artifacts 2>/dev/null || true
rm -f  /home/user/.openhands/.env /home/user/.hermes/.env /home/user/.openclaw/.env 2>/dev/null || true
rm -rf /home/user/ale-test /home/user/.ale-src 2>/dev/null || true

# --- scrub dev-VM cruft: the image is a rootfs export of a shared dev VM, so it
#     carries per-user config and other files that don't belong in the benchmark
#     image. Remove them WITHOUT touching the prebaked agent harnesses (codex in
#     /usr/local, openclaw, hermes/gemini/claude in ~/.local, .openclaw/.grok/
#     .bun/.cargo/.rustup/.npm-global) or the /opt task software. ---
#
#   per-user config / state that shouldn't ship in a published image:
rm -rf /home/user/.config/gcloud-agenthle-artifacts 2>/dev/null || true
rm -f  /home/user/.netrc 2>/dev/null || true
rm -f  /home/user/.hermes/auth.json 2>/dev/null || true
rm -rf /home/user/.hermes/sessions 2>/dev/null || true
rm -rf /home/user/.config/google-chrome/Default 2>/dev/null || true       # dev browser profile
rm -f  /home/user/.config/google-chrome/Singleton* 2>/dev/null || true
rm -f  /home/user/.bash_history /home/user/.python_history \
       /home/user/.zsh_history /home/user/.node_repl_history /root/.bash_history 2>/dev/null || true
rm -rf /root/.ssh 2>/dev/null || true
#
#   extra user accounts carried over from the shared dev VM (homes + accounts):
for u in weichenzhang bytedance User; do userdel -r "$u" 2>/dev/null || rm -rf "/home/$u"; done
rm -rf /home/{{your_email* 2>/dev/null || true   # a stray templated home dir
for u in weichenzhang bytedance User; do
  sed -i "\|^$u:|d" /etc/passwd /etc/shadow /etc/group /etc/gshadow 2>/dev/null || true
done
sed -i '/^{{your_email/d' /etc/passwd /etc/shadow /etc/group /etc/gshadow 2>/dev/null || true
#
#   benchmark-irrelevant dev cruft / build workspaces / caches / stray files:
rm -rf /home/user/codex-build 2>/dev/null || true                  # codex BUILD dir (codex runs from /usr/local)
rm -f  /home/user/reference.frc 2>/dev/null || true                # ~580MB stray simulation output
rm -rf /home/user/.config/Code /home/user/.config/Sabaki 2>/dev/null || true   # dev editor / app state
rm -rf /home/user/.agenthle_hidden_eval_assets 2>/dev/null || true # leftover per-task eval asset (no Linux task should rely on a baked home-dir copy)
rm -rf /home/user/.cache /home/user/.npm /root/.cache 2>/dev/null || true       # package/build caches

# --- sanity: paths the ale-ubuntu22-docker Image entry promises must exist ---
echo "--- verify image-promised paths ---"
fail=0
for p in /usr/local/bin/node \
         /opt/cua-server/.venv/bin/python \
         /opt/ale-run/.venv/bin/python \
         /home/user/cua_mcp_server \
         /media/user/data/agenthle; do
  if [ -e "$p" ]; then echo "OK   $p"; else echo "MISS $p"; fail=1; fi
done
command -v Xvfb >/dev/null && echo "OK   Xvfb" || { echo "MISS Xvfb"; fail=1; }
command -v startxfce4 >/dev/null && echo "OK   startxfce4 (XFCE desktop)" || echo "WARN startxfce4 missing (bare Xvfb, no WM)"
# the scrub must NOT have touched the prebaked agent harnesses
for b in /usr/local/bin/codex /usr/local/bin/openclaw \
         /home/user/.local/bin/claude /home/user/.local/bin/gemini /home/user/.local/bin/hermes; do
  if [ -e "$b" ]; then echo "OK   agent: $b"; else echo "MISS agent: $b"; fail=1; fi
done
# the removed dev-VM files/accounts must be GONE
for s in /home/user/.config/gcloud-agenthle-artifacts /home/user/.netrc \
         /home/weichenzhang /home/bytedance /root/.ssh; do
  [ -e "$s" ] && { echo "LEFTOVER (should be gone): $s"; fail=1; } || echo "OK   scrubbed: $s"
done
/opt/cua-server/.venv/bin/python -c "import computer_server" 2>/dev/null \
  && echo "OK   computer_server importable" \
  || echo "WARN computer_server import failed without X (expected; entrypoint starts Xvfb)"

[ "$fail" = 0 ] && echo "CLEANUP_OK" || echo "CLEANUP_WARN: missing expected paths above"
