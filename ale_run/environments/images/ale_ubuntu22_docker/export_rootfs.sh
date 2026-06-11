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
