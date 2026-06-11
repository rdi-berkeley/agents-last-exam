#!/usr/bin/env python3
"""Stream a tar on stdin -> tar on stdout, clamping any uid/gid that falls
outside the rootless-Docker subuid range (>65535) down to 0 (root).

Why: rootless `docker import` maps every in-image uid through /etc/subuid,
which is only 65536 ids wide. The VM rootfs carries a few files owned by huge
corp/LDAP uids (e.g. ~629M under /opt/subread); importing them fails with
`lchown ... invalid argument`. Clamping ONLY the out-of-range ids to 0 fixes
the import while preserving every normal owner — crucially uid 1000 (`user`),
which must keep owning /home/user, /opt/cua-server, /media/user/data, etc.

Constant memory; large copy buffer for throughput.
"""
import sys
import tarfile

MAXID = 65535
remapped = 0
total = 0

tin = tarfile.open(fileobj=sys.stdin.buffer, mode="r|*")
tout = tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.GNU_FORMAT)
tout.copybufsize = 8 * 1024 * 1024  # 8 MiB blocks vs the 64 KiB default

for m in tin:
    total += 1
    changed = False
    if m.uid > MAXID:
        m.uid, m.uname, changed = 0, "", True
    if m.gid > MAXID:
        m.gid, m.gname, changed = 0, "", True
    if changed:
        remapped += 1
    if m.isreg():
        tout.addfile(m, tin.extractfile(m))
    else:
        tout.addfile(m)

tout.close()
tin.close()
sys.stderr.write(
    f"remap_uids: clamped {remapped}/{total} entries with out-of-range uid/gid to 0\n"
)
