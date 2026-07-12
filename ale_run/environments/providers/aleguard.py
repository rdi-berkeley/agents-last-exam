"""aleguard — a tiny transparent egress allow-list proxy for a task sandbox.

Runs *inside* the sandbox VM (shipped there by the gcloud provider). All
outbound TCP to :80/:443 from every user except aleguard's own is REDIRECTed
to this proxy by an nftables rule; the proxy reads the connection's original
destination (``SO_ORIGINAL_DST``), extracts the target **hostname** (TLS SNI
for 443, HTTP ``Host`` for 80) *without decrypting anything*, and:

* forwards the connection to its real destination iff the hostname matches the
  allow-list (suffix match on the registrable domain), else
* drops it and logs the denied host.

Because filtering is on the plaintext SNI / Host, it is immune to CDN IP churn
(the model endpoint's Cloudflare IPs can rotate freely) and never needs the
agent to honour an ``HTTPS_PROXY`` env var — any HTTP client is transparently
governed.

This module is import-safe and dependency-free (stdlib only, Python 3.8+) so
its pure helpers (:func:`extract_sni`, :func:`extract_http_host`,
:func:`host_allowed`) can be unit-tested on the host without a VM.

Run (on the VM, as the dedicated ``aleproxy`` user)::

    python3 aleguard.py --port 15599 --allow-file /etc/aleguard/allow \
        --log-file /var/log/aleguard.log
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import struct
import sys

SO_ORIGINAL_DST = 80  # <linux/netfilter_ipv4.h>
_PEEK_BYTES = 8192
_PEEK_TIMEOUT_S = 10.0
_IO_CHUNK = 65536

logger = logging.getLogger("aleguard")


# ───────────────────────────── hostname extraction ─────────────────────────

def extract_sni(data: bytes) -> str | None:
    """Extract the SNI hostname from a TLS ClientHello. None if not present /
    unparseable (fail-closed: caller denies). No decryption — the SNI is sent
    in the clear before the handshake completes."""
    try:
        # TLS record: type(1)=0x16 handshake, version(2), length(2)
        if len(data) < 5 or data[0] != 0x16:
            return None
        # handshake: msg_type(1)=0x01 ClientHello, length(3), version(2),
        # random(32), then session_id / cipher_suites / compression / extensions
        pos = 5
        if len(data) < pos + 4 or data[pos] != 0x01:
            return None
        pos += 4  # skip handshake type + length
        pos += 2 + 32  # client_version + random
        if pos >= len(data):
            return None
        sid_len = data[pos]; pos += 1 + sid_len
        if pos + 2 > len(data):
            return None
        cs_len = struct.unpack("!H", data[pos:pos + 2])[0]; pos += 2 + cs_len
        if pos >= len(data):
            return None
        comp_len = data[pos]; pos += 1 + comp_len
        if pos + 2 > len(data):
            return None
        ext_total = struct.unpack("!H", data[pos:pos + 2])[0]; pos += 2
        end = min(len(data), pos + ext_total)
        while pos + 4 <= end:
            etype, elen = struct.unpack("!HH", data[pos:pos + 4]); pos += 4
            if etype == 0x0000:  # server_name
                # server_name_list len(2), name_type(1)=0 host, name_len(2), name
                if pos + 5 > len(data):
                    return None
                nlen = struct.unpack("!H", data[pos + 3:pos + 5])[0]
                name = data[pos + 5:pos + 5 + nlen]
                try:
                    return name.decode("idna") if name else None
                except Exception:
                    return name.decode("ascii", "ignore").lower() or None
            pos += elen
    except Exception:
        return None
    return None


def extract_http_host(data: bytes) -> str | None:
    """Extract the ``Host`` header from a plaintext HTTP request. None if
    absent."""
    try:
        head = data.split(b"\r\n\r\n", 1)[0]
        for line in head.split(b"\r\n")[1:]:
            if line[:5].lower() == b"host:":
                host = line[5:].strip().decode("ascii", "ignore")
                return host.rsplit(":", 1)[0].lower() or None
    except Exception:
        return None
    return None


def host_allowed(host: str | None, allow: frozenset[str]) -> bool:
    """True iff *host* equals or is a subdomain of an allow-list entry.
    ``api.openrouter.ai`` matches allow entry ``openrouter.ai``. Case- and
    trailing-dot-insensitive. Empty host / empty allow → False (fail-closed)."""
    if not host or not allow:
        return False
    h = host.strip().rstrip(".").lower()
    for entry in allow:
        e = entry.strip().rstrip(".").lower()
        if e and (h == e or h.endswith("." + e)):
            return True
    return False


def _original_dst(sock: socket.socket) -> tuple[str, int] | None:
    try:
        raw = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
        port = struct.unpack("!H", raw[2:4])[0]
        ip = socket.inet_ntoa(raw[4:8])
        return ip, port
    except OSError:
        return None


# ───────────────────────────── proxy core ──────────────────────────────────

class Proxy:
    def __init__(self, allow: frozenset[str], audit: logging.Logger):
        self.allow = allow
        self.audit = audit

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer_sock = writer.get_extra_info("socket")
        dst = _original_dst(peer_sock) if peer_sock is not None else None
        if dst is None:
            writer.close()
            return
        dst_ip, dst_port = dst
        try:
            head = await asyncio.wait_for(reader.read(_PEEK_BYTES), _PEEK_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError):
            writer.close()
            return
        if not head:
            writer.close()
            return

        host = extract_sni(head) if dst_port == 443 else extract_http_host(head)
        if not host_allowed(host, self.allow):
            self.audit.info("DENY %s:%d host=%s", dst_ip, dst_port, host)
            try:
                writer.close()
            except OSError:
                pass
            return
        self.audit.info("ALLOW %s:%d host=%s", dst_ip, dst_port, host)

        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(dst_ip, dst_port), _PEEK_TIMEOUT_S
            )
        except (asyncio.TimeoutError, OSError) as exc:
            self.audit.info("UPSTREAM-FAIL %s:%d host=%s err=%s", dst_ip, dst_port, host, exc)
            writer.close()
            return

        up_writer.write(head)  # replay the bytes we peeked
        try:
            await up_writer.drain()
        except OSError:
            pass

        await asyncio.gather(
            self._pump(reader, up_writer),
            self._pump(up_reader, writer),
            return_exceptions=True,
        )

    @staticmethod
    async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                chunk = await src.read(_IO_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
                await dst.drain()
        except OSError:
            pass
        finally:
            try:
                dst.close()
            except OSError:
                pass


# ───────────────────────── VM setup / teardown builders ────────────────────
# Pure string builders (host-side, unit-testable) that the gcloud provider runs
# over the cua channel as root to install/remove enforcement on a Linux guest.

PROXY_PORT = 15599
_ALLOW_FILE = "/etc/aleguard/allow"
_PROXY_PY = "/opt/aleguard/aleguard.py"
_LOG_FILE = "/var/log/aleguard.log"


def build_nft_ruleset(mode: str) -> str:
    """nftables ruleset for a Linux guest. ``__UID__`` is substituted with the
    aleproxy uid by the setup script.

    allowlist: redirect all :80/:443 (except aleproxy's own egress) to the local
      transparent proxy, which allows by hostname; permit DNS + loopback +
      established (cua replies) + aleproxy egress; drop everything else. IPv6
      egress is rejected so happy-eyeballs clients fall back to the governed v4.
    off: air-gap — only loopback + established (cua) survive; even DNS/model die.
    """
    v6 = (
        "table ip6 aleguard6 {\n"
        "  chain out {\n"
        "    type filter hook output priority 0; policy drop;\n"
        "    oif lo accept\n"
        "    ct state established,related accept\n"
        "    meta l4proto { tcp, udp } reject\n"
        "  }\n"
        "}\n"
    )
    if mode == "off":
        return (
            "flush ruleset\n"
            "table ip aleguard {\n"
            "  chain out {\n"
            "    type filter hook output priority 0; policy drop;\n"
            "    oif lo accept\n"
            "    ct state established,related accept\n"
            "  }\n"
            "}\n" + v6
        )
    # allowlist
    return (
        "flush ruleset\n"
        "table ip aleguard {\n"
        "  chain nat_out {\n"
        "    type nat hook output priority -100; policy accept;\n"
        "    meta skuid __UID__ return\n"
        f"    tcp dport {{ 80, 443 }} redirect to :{PROXY_PORT}\n"
        "  }\n"
        "  chain out {\n"
        "    type filter hook output priority 0; policy drop;\n"
        "    oif lo accept\n"
        "    ct state established,related accept\n"
        "    meta skuid __UID__ accept\n"
        "    udp dport 53 accept\n"
        "    tcp dport 53 accept\n"
        "  }\n"
        "}\n" + v6
    )


def build_setup_script(mode: str, allow_hosts: list[str], proxy_py_b64: str) -> str:
    """Bash (run as root over cua) that installs enforcement. Idempotent."""
    import shlex
    allow_lines = "\n".join(sorted(set(allow_hosts)))
    nft = build_nft_ruleset(mode)
    start_proxy = ""
    if mode != "off":
        start_proxy = (
            "pkill -f aleguard.py 2>/dev/null || true\n"
            f"base64 -d > {_PROXY_PY} <<'ALEB64'\n{proxy_py_b64}\nALEB64\n"
            f"setsid runuser -u aleproxy -- python3 {_PROXY_PY} "
            f"--port {PROXY_PORT} --allow-file {_ALLOW_FILE} --log-file {_LOG_FILE} "
            ">/dev/null 2>&1 &\n"
            f"for i in $(seq 1 100); do ss -ltn 2>/dev/null | grep -q ':{PROXY_PORT} ' "
            "&& break; sleep 0.1; done\n"
        )
    return (
        "set -e\n"
        "mkdir -p /opt/aleguard /etc/aleguard\n"
        "id -u aleproxy >/dev/null 2>&1 || useradd -r -M -s /usr/sbin/nologin aleproxy\n"
        "ALEUID=$(id -u aleproxy)\n"
        f"cat > {_ALLOW_FILE} <<'ALEALLOW'\n{allow_lines}\nALEALLOW\n"
        + start_proxy +
        "cat > /etc/aleguard/rules.nft <<ALENFT\n"
        + nft.replace("__UID__", "$ALEUID") +
        "ALENFT\n"
        "nft -f /etc/aleguard/rules.nft\n"
        "echo ALEGUARD_OK\n"
    )


def build_teardown_script() -> str:
    """Bash that removes enforcement and restores full connectivity."""
    return (
        "nft delete table ip aleguard 2>/dev/null || true\n"
        "nft delete table ip6 aleguard6 2>/dev/null || true\n"
        "pkill -f aleguard.py 2>/dev/null || true\n"
        "echo ALEGUARD_CLEARED\n"
    )


def _load_allow(path: str) -> frozenset[str]:
    hosts: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    hosts.add(s.lower())
    except OSError:
        pass
    return frozenset(hosts)


async def _serve(port: int, allow: frozenset[str], audit: logging.Logger) -> None:
    proxy = Proxy(allow, audit)
    server = await asyncio.start_server(proxy.handle, "0.0.0.0", port)
    audit.info("aleguard listening on :%d allow=%s", port, sorted(allow))
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="transparent egress allow-list proxy")
    ap.add_argument("--port", type=int, default=15599)
    ap.add_argument("--allow-file", required=True)
    ap.add_argument("--log-file", default="")
    args = ap.parse_args(argv)

    audit = logging.getLogger("aleguard.audit")
    audit.setLevel(logging.INFO)
    handler: logging.Handler = (
        logging.FileHandler(args.log_file) if args.log_file else logging.StreamHandler(sys.stderr)
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit.addHandler(handler)

    allow = _load_allow(args.allow_file)
    try:
        asyncio.run(_serve(args.port, allow, audit))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
