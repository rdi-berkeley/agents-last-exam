"""netguard proxy — a cross-platform TLS-passthrough allow-list proxy.

The whole egress design is deliberately simple and identical on Linux and
Windows:

1. The guest's ``hosts`` file maps each allowed hostname → ``127.0.0.1``, so any
   client (Node, curl, a browser — no cooperation needed) that connects to the
   model endpoint lands on this proxy.
2. This proxy peeks the **TLS SNI** (plaintext, no decryption), confirms the
   hostname is allow-listed, connects to the endpoint's **real** IP (resolved at
   setup, before the hosts override, and handed to us in a map), and splices the
   bytes — the TLS handshake stays end-to-end between the client and the real
   server.
3. A default-deny egress firewall (nftables owner-match on Linux, Windows
   Firewall program-rule on Windows) lets ONLY this proxy process + DNS out, so
   any *other* host the agent tries — resolved to its real IP, not 127.0.0.1 —
   is dropped.

So the agent can reach exactly the hostnames on the allow-list and nothing else,
with no in-kernel packet redirect and no client proxy-awareness. Stdlib-only.

Run (on the guest)::

    python3 proxy.py --map /etc/netguard/map.json --log /var/log/netguard.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import struct
import sys

_LISTEN_PORT = 443
_PEEK = 8192
_PEEK_TIMEOUT = 8.0
_CONNECT_TIMEOUT = 10.0
_IO = 65536

log = logging.getLogger("netguard")


def extract_sni(data: bytes) -> str | None:
    """SNI hostname from a TLS ClientHello, else None (fail-closed)."""
    try:
        if len(data) < 5 or data[0] != 0x16:
            return None
        pos = 5
        if len(data) < pos + 4 or data[pos] != 0x01:
            return None
        pos += 4 + 2 + 32  # hs type+len, client_version, random
        if pos >= len(data):
            return None
        pos += 1 + data[pos]  # session_id
        if pos + 2 > len(data):
            return None
        pos += 2 + struct.unpack("!H", data[pos:pos + 2])[0]  # cipher_suites
        if pos >= len(data):
            return None
        pos += 1 + data[pos]  # compression
        if pos + 2 > len(data):
            return None
        ext_end = min(len(data), pos + 2 + struct.unpack("!H", data[pos:pos + 2])[0])
        pos += 2
        while pos + 4 <= ext_end:
            etype, elen = struct.unpack("!HH", data[pos:pos + 4])
            pos += 4
            if etype == 0x0000:
                if pos + 5 > len(data):
                    return None
                nlen = struct.unpack("!H", data[pos + 3:pos + 5])[0]
                name = data[pos + 5:pos + 5 + nlen]
                try:
                    return name.decode("idna").lower() or None
                except Exception:
                    return name.decode("ascii", "ignore").lower() or None
            pos += elen
    except Exception:
        return None
    return None


def resolve_ip(host: str | None, amap: dict) -> str | None:
    """Return a real IP for *host* from the setup-resolved map (exact or
    subdomain match), else None."""
    if not host:
        return None
    h = host.strip().rstrip(".").lower()
    for name, ips in amap.items():
        n = name.strip().rstrip(".").lower()
        if ips and (h == n or h.endswith("." + n)):
            return ips[0]
    return None


class Proxy:
    def __init__(self, amap: dict, audit: logging.Logger):
        self.amap = amap
        self.audit = audit

    async def handle(self, reader, writer):
        try:
            head = await asyncio.wait_for(reader.read(_PEEK), _PEEK_TIMEOUT)
        except (asyncio.TimeoutError, OSError):
            return _close(writer)
        if not head:
            return _close(writer)
        host = extract_sni(head)
        ip = resolve_ip(host, self.amap)
        if not ip:
            self.audit.info("DENY %s", host)
            return _close(writer)
        self.audit.info("ALLOW %s -> %s", host, ip)
        try:
            up_r, up_w = await asyncio.wait_for(
                asyncio.open_connection(ip, 443, family=socket.AF_INET), _CONNECT_TIMEOUT)
        except (asyncio.TimeoutError, OSError) as e:
            self.audit.info("UPSTREAM-FAIL %s %s %s", host, ip, e)
            return _close(writer)
        up_w.write(head)
        try:
            await up_w.drain()
        except OSError:
            pass
        await asyncio.gather(_pump(reader, up_w), _pump(up_r, writer),
                             return_exceptions=True)


async def _pump(src, dst):
    try:
        while True:
            b = await src.read(_IO)
            if not b:
                break
            dst.write(b)
            await dst.drain()
    except OSError:
        pass
    finally:
        _close(dst)


def _close(w):
    try:
        w.close()
    except OSError:
        pass


async def _serve(amap: dict, audit: logging.Logger):
    srv = await asyncio.start_server(Proxy(amap, audit).handle, "127.0.0.1", _LISTEN_PORT)
    audit.info("netguard proxy on 127.0.0.1:%d allow=%s", _LISTEN_PORT, sorted(amap))
    async with srv:
        await srv.serve_forever()


def _do_resolve(hosts: list[str], map_path: str) -> int:
    """Resolve real IPs for *hosts* and write the map. MUST run BEFORE the hosts
    file is overridden (else the names resolve to 127.0.0.1). Fails if any host
    can't be resolved — better to abort than to strand the agent."""
    amap: dict[str, list[str]] = {}
    for host in hosts:
        host = host.strip().lower()
        if not host:
            continue
        try:
            infos = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except OSError as e:
            sys.stderr.write(f"netguard: cannot resolve {host!r}: {e}\n")
            return 2
        ips = sorted({i[4][0] for i in infos})
        if not ips:
            sys.stderr.write(f"netguard: no A record for {host!r}\n")
            return 2
        amap[host] = ips
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(amap, f)
    sys.stdout.write("NETGUARD_RESOLVED " + json.dumps(amap) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--resolve", default="", help="comma-separated hosts: resolve+write map, then exit")
    ap.add_argument("--log", default="")
    a = ap.parse_args(argv)
    if a.resolve:
        return _do_resolve(a.resolve.split(","), a.map)
    audit = logging.getLogger("netguard.audit")
    audit.setLevel(logging.INFO)
    h = logging.FileHandler(a.log) if a.log else logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit.addHandler(h)
    try:
        amap = json.load(open(a.map, encoding="utf-8"))
    except OSError:
        amap = {}
    try:
        asyncio.run(_serve(amap, audit))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
