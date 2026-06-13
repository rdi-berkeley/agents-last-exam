#!/bin/bash
# Runs ON the source VM, fed over SSH as:  ssh <vm> 'bash -s' < export_rootfs.sh
#
# Streams a zstd-compressed tar of the VM's root filesystem to stdout, dropping
# everything a container shares with / gets from the host (kernel, init, devices,
# logs, caches, snap/flatpak desktop bits, swap). A container only needs the
# userspace rootfs; the host kernel is shared.
#
# tar's own exit code is recorded to /tmp/ale_export_tar.rc so the caller can
# tell a benign "files changed while reading" (1) from a fatal error (2): in a
# pipe the shell only sees zstd's status, not tar's.

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
    -cf - -C / . 2>/tmp/ale_export_tar.err
  echo $? > /tmp/ale_export_tar.rc
} | zstd -T0 -3 -c
