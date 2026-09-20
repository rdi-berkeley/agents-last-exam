"""Provision a public export key inside the disposable CUA-ready builder only."""

from __future__ import annotations

import base64
import os
import pwd
import subprocess
import sys
from pathlib import Path

HOST_PUBLIC_KEY = Path("/etc/ssh/ssh_host_ed25519_key.pub")


def public_key(value: str) -> str:
    parts = value.strip().split()
    if len(parts) not in (2, 3) or parts[0] not in {
        "ssh-ed25519",
        "ssh-rsa",
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
    }:
        raise ValueError("invalid SSH public key")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as error:
        raise ValueError("invalid SSH public key encoding") from error
    name = parts[0].encode("ascii")
    if len(blob) < 4 + len(name) + 16 or int.from_bytes(blob[:4], "big") != len(name):
        raise ValueError("invalid SSH public key blob")
    if blob[4 : 4 + len(name)] != name:
        raise ValueError("SSH public key type mismatch")
    return " ".join(parts[:2])


def provision(user: str, key: str) -> str:
    key = public_key(key)
    account = pwd.getpwnam(user)
    home = Path(account.pw_dir)
    if not home.is_absolute() or home.resolve(strict=True) != home:
        raise ValueError("builder home must be an existing canonical directory")
    ssh_dir = home / ".ssh"
    if ssh_dir.is_symlink():
        raise ValueError("refusing a symlinked builder .ssh directory")
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    ssh_dir.chmod(0o700)
    os.chown(ssh_dir, account.pw_uid, account.pw_gid)
    authorized = ssh_dir / "authorized_keys"
    authorized.unlink(missing_ok=True)
    with authorized.open("x") as output:
        output.write("restrict " + key + "\n")
    authorized.chmod(0o600)
    os.chown(authorized, account.pw_uid, account.pw_gid)
    subprocess.run(["/usr/bin/ssh-keygen", "-A"], check=True, timeout=15)
    subprocess.run(["/usr/bin/systemctl", "start", "ssh"], check=True, timeout=30)
    return public_key(HOST_PUBLIC_KEY.read_text())


if __name__ == "__main__":
    print(provision(sys.argv[1], sys.argv[2]))
