"""Host-side egress gateway for sealed sandboxes.

A sealed sandbox can open one kind of connection: ``CONNECT <host>:443`` to
this gateway, for a host on the run's allowlist. The provider drops every
other packet the guest sends, so traffic that ignores the proxy has nowhere
to go. The tunnel is relayed without decrypting it.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import selectors
import socket
import socketserver
import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MAX_HEADER_BYTES = 16 * 1024
_HANDSHAKE_TIMEOUT_S = 15
_UPSTREAM_CONNECT_TIMEOUT_S = 15
# A model may think for minutes between stream events.
_IDLE_TIMEOUT_S = 1800


def canonical_host(value: Any) -> str:
    """Return a lowercase fully qualified DNS name, or raise ``ValueError``.

    URLs, ports, wildcards and IP literals are rejected: an allowlist entry
    names one host.
    """
    host = str(value).strip().rstrip(".").lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"allowlist entry must be a DNS name, not an IP literal: {value!r}")
    labels = host.split(".")
    if len(host) > 253 or len(labels) < 2 or not all(_DNS_LABEL_RE.fullmatch(x) for x in labels):
        raise ValueError(f"allowlist entry must be an exact DNS name like api.openai.com: {value!r}")
    return host


def allowlist_host_from_url(url: str) -> str:
    """Host of an ``https://`` endpoint URL, as an allowlist entry."""
    parsed = urlsplit(str(url))
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"network-off needs an https:// model endpoint, got {url!r}")
    if parsed.port not in (None, 443):
        raise ValueError(f"network-off gateway relays port 443 only, got {url!r}")
    return canonical_host(parsed.hostname)


def _usable_upstream(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, gateway: EgressGateway, address: tuple[str, int]):
        self.gateway = gateway
        super().__init__(address, _Handler)


class _Handler(socketserver.BaseRequestHandler):
    server: _Server

    def handle(self) -> None:
        client: socket.socket = self.request
        client.settimeout(_HANDSHAKE_TIMEOUT_S)
        try:
            head, rest = _read_head(client)
            parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
        except (OSError, ValueError):
            return
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.server.gateway.record(False, parts[1] if len(parts) > 1 else "")
            _reply(client, 405, "this gateway accepts CONNECT only")
            return
        authority = parts[1].lower()
        host, _, port = authority.rpartition(":")
        if port != "443" or host.rstrip(".") not in self.server.gateway.allowlist:
            self.server.gateway.record(False, authority)
            _reply(client, 403, f"{authority} is not allowlisted for this run")
            return
        try:
            upstream = _connect_upstream(host.rstrip("."))
        except OSError as exc:
            self.server.gateway.record(False, authority)
            _reply(client, 502, f"cannot reach {authority}: {exc}")
            return
        self.server.gateway.record(True, authority)
        with upstream:
            try:
                client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if rest:
                    upstream.sendall(rest)
                client.settimeout(None)
                _relay(client, upstream)
            except OSError:
                pass


def _read_head(sock: socket.socket) -> tuple[bytes, bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk or len(data) + len(chunk) > _MAX_HEADER_BYTES:
            raise ValueError("incomplete or oversized proxy request")
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def _reply(sock: socket.socket, status: int, message: str) -> None:
    body = (message + "\n").encode()
    try:
        sock.sendall(
            f"HTTP/1.1 {status} {message}\r\nContent-Type: text/plain\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
    except OSError:
        pass


def _connect_upstream(host: str) -> socket.socket:
    last_error: OSError | None = None
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if not _usable_upstream(sockaddr[0]):
            last_error = OSError(f"{host} resolved to non-routable {sockaddr[0]}")
            continue
        try:
            return socket.create_connection(sockaddr[:2], timeout=_UPSTREAM_CONNECT_TIMEOUT_S)
        except OSError as exc:
            last_error = exc
    raise last_error or OSError(f"{host} did not resolve")


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(client, selectors.EVENT_READ, upstream)
        selector.register(upstream, selectors.EVENT_READ, client)
        while ready := selector.select(_IDLE_TIMEOUT_S):
            for key, _ in ready:
                data = key.fileobj.recv(64 * 1024)
                if not data:
                    return
                key.data.sendall(data)


class EgressGateway:
    """Allowlist-only CONNECT proxy for one sealed sandbox."""

    def __init__(self, allowlist: Iterable[str], *, bind_host: str, port: int = 0):
        self.allowlist = frozenset(canonical_host(host) for host in allowlist)
        self._server = _Server(self, (bind_host, port))
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._allowed: Counter[str] = Counter()
        self._denied: Counter[str] = Counter()

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name=f"egress-gateway-{self.address[1]}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()

    def record(self, allowed: bool, authority: str) -> None:
        with self._lock:
            (self._allowed if allowed else self._denied)[authority[:255]] += 1
        if not allowed:
            logger.warning("egress gateway %s:%s denied %r", *self.address, authority)

    def summary(self) -> dict[str, dict[str, int]]:
        """Connection counts by requested authority, for run records."""
        with self._lock:
            return {"allowed": dict(self._allowed), "denied": dict(self._denied)}
