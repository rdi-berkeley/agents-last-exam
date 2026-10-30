"""In-guest check that a sealed sandbox reaches only its egress gateway."""

from __future__ import annotations

import base64
import json
import shlex
import subprocess
from collections.abc import Sequence
from typing import Any

from ..base_interface import SandboxHandle

_MARKER = "ALE_NETWORK_PROBE:"

_PROBE_PY = r"""
import json, os, socket, sys

proxy_host, proxy_port, allowlist = sys.argv[1], int(sys.argv[2]), sys.argv[3:]


def direct_tcp(host, port):
    try:
        socket.create_connection((host, port), timeout=5).close()
        return True
    except OSError:
        return False


def dns_answers(server):
    labels = b"".join(bytes([len(p)]) + p.encode() for p in ("example", "com"))
    query = os.urandom(2) + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00" + labels + b"\x00\x00\x01\x00\x01"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(3)
        try:
            sock.sendto(query, (server, 53))
            sock.recvfrom(512)
            return True
        except OSError:
            return False


def connect_status(authority):
    try:
        with socket.create_connection((proxy_host, proxy_port), timeout=10) as sock:
            sock.settimeout(20)
            sock.sendall(("CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n" % (authority, authority)).encode())
            data = b""
            while b"\r\n" not in data and len(data) < 4096:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        return int(data.split(None, 2)[1])
    except (OSError, ValueError, IndexError):
        return None


print("ALE_NETWORK_PROBE:" + json.dumps({
    "direct_tcp": {h: direct_tcp(h, 443) for h in ("1.1.1.1", "8.8.8.8")},
    "dns": {s: dns_answers(s) for s in (proxy_host, "1.1.1.1", "8.8.8.8")},
    "unlisted_status": connect_status("example.com:443"),
    "allowlisted_status": {h: connect_status(h + ":443") for h in allowlist},
}))
"""


def _failures(report: dict[str, Any], allowlist: Sequence[str]) -> list[str]:
    failures = [f"direct TCP to {h} succeeded" for h, ok in report["direct_tcp"].items() if ok]
    failures += [f"DNS server {s} answered" for s, ok in report["dns"].items() if ok]
    if report["unlisted_status"] != 403:
        failures.append(f"gateway answered {report['unlisted_status']} for an unlisted host")
    failures += [
        f"gateway answered {report['allowlisted_status'].get(h)} for allowlisted {h}"
        for h in allowlist
        if report["allowlisted_status"].get(h) != 200
    ]
    return failures


async def probe_sealed_guest(
    sandbox: SandboxHandle,
    *,
    proxy_host: str,
    proxy_port: int,
    allowlist: Sequence[str],
) -> dict[str, Any]:
    """Run the probe inside the guest; return its report or raise ``RuntimeError``.

    Passes only if direct TCP and DNS fail, the gateway refuses an unlisted
    host with 403, and it opens a tunnel to every allowlisted host.
    """
    encoded = base64.b64encode(_PROBE_PY.encode()).decode("ascii")
    argv = [
        sandbox.python, "-B", "-c", f"import base64;exec(base64.b64decode('{encoded}'))",
        proxy_host, str(proxy_port), *allowlist,
    ]
    command = shlex.join(argv) if sandbox.is_linux else subprocess.list2cmdline(argv)
    result = await sandbox.run_command(command, timeout=120)
    line = next(
        (x for x in reversed((result.stdout or "").splitlines()) if x.startswith(_MARKER)),
        None,
    )
    if result.returncode != 0 or line is None:
        raise RuntimeError(
            f"network probe did not run (rc={result.returncode}): "
            f"{(result.stderr or result.stdout or '').strip()[:500]}"
        )
    report = json.loads(line[len(_MARKER):])
    if failures := _failures(report, allowlist):
        raise RuntimeError(f"sealed sandbox failed the network probe: {'; '.join(failures)}")
    return report
