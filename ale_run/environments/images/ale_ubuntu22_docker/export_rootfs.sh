#!/bin/bash
# Runs ON the source VM. Two ways to drive it:
#   * SSH-pull (default):     ssh <vm> 'bash -s' < export_rootfs.sh  > rootfs.tar.zst
#       -> a zstd-compressed tar streamed to stdout, imported on the LOCAL host.
#   * On-VM (ALE_EXPORT_RAW=1): ALE_EXPORT_RAW=1 bash export_rootfs.sh | docker import - <ref>
#       -> a RAW (uncompressed) tar to stdout, imported by docker ON THE VM, so
#          the ~37GB rootfs never crosses the network (no slow SSH transfer).
#
# Either way the SAME excludes apply, so the two paths produce identical images.
# Drops everything a container shares with / gets from the host (kernel, init,
# devices, logs, caches, snap/flatpak desktop bits, swap) + the task data and the
# dev-VM-specific files (see excludes). A container only needs the userspace rootfs.
#
# tar's own exit code is recorded to /tmp/ale_export_tar.rc so the caller can
# tell a benign "files changed while reading" (1) from a fatal error (2): in a
# pipe the shell only sees the downstream stage's status, not tar's.

# Clamp out-of-range owners to root here, on the VM, so the local `docker
# import` can be a plain stream (no per-file Python remap pass — that was the
# ~2h bottleneck). Rootless docker maps every in-image uid through a 65536-wide
# subuid range, so a few stray corp/LDAP-owned files (e.g. under /opt/subread,
# uid ~629M) would otherwise fail `lchown`. Only a handful match; cheap.
sudo find / -xdev \( -uid +65535 -o -gid +65535 \) -exec chown -h 0:0 {} + 2>/dev/null || true

{
  sudo tar --numeric-owner \
    --warning=no-file-changed --warning=no-file-removed --warning=no-file-ignored \
    --exclude=./proc --exclude=./sys --exclude=./dev --exclude=./run \
    --exclude=./tmp --exclude=./var/tmp \
    --exclude=./boot --exclude=./snap --exclude=./var/snap \
    --exclude=./var/lib/snapd --exclude=./var/lib/flatpak \
    --exclude=./var/log --exclude=./var/cache \
    --exclude=./swapfile --exclude=./cdrom --exclude=./lost+found \
    --exclude=./media/floppy0 --exclude=./srv --exclude=./.cache \
    \
    --exclude=./media/user/data \
    --exclude=./opt/ale-docker-images \
    \
    --exclude=./home/user/ale-test --exclude=./home/user/.ale-src \
    --exclude=./home/user/.config/agenthle-artifacts \
    --exclude='./home/*/.env' --exclude='./home/*/*/.env' \
    --exclude=./root/.config/agenthle-artifacts \
    \
    --exclude=./home/weichenzhang --exclude=./home/bytedance \
    --exclude=./home/User --exclude='./home/{{*' \
    --exclude=./root/.ssh \
    --exclude=./home/user/.config/gcloud-agenthle-artifacts \
    --exclude=./home/user/.netrc \
    --exclude=./home/user/.hermes/auth.json --exclude=./home/user/.hermes/sessions \
    --exclude=./home/user/.config/google-chrome/Default \
    --exclude='./home/*/.bash_history' --exclude='./home/*/.python_history' \
    --exclude='./home/*/.zsh_history' --exclude=./root/.bash_history \
    --exclude=./home/user/codex-build --exclude=./home/user/reference.frc \
    --exclude=./home/user/.config/Code --exclude=./home/user/.config/Sabaki \
    --exclude=./home/user/.agenthle_hidden_eval_assets \
    --exclude=./home/user/.cache --exclude=./home/user/.npm --exclude=./root/.cache \
    -cf - -C / . 2>/tmp/ale_export_tar.err
  echo $? > /tmp/ale_export_tar.rc
} | if [ "${ALE_EXPORT_RAW:-0}" = 1 ]; then cat; else zstd -T0 -3 -c; fi
