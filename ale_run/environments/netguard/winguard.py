"""winguard — a tiny forward (CONNECT) allow-list proxy for a Windows sandbox.

Windows has no in-kernel transparent-REDIRECT equivalent of Linux nftables (that
would need a signed WFP/WinDivert driver, not baked on the image), so the
Windows egress design is:

* **Windows Firewall default-deny outbound**, allowing only this proxy process +
  DNS + loopback — the *hard* guarantee that nothing escapes, for any client
  (Node, curl, browser), whether or not it honours a proxy setting.
* **this CONNECT proxy** does the **hostname** allow-listing: proxy-aware clients
  send ``CONNECT host:443`` (or an absolute-form HTTP request) and the proxy
  forwards iff the host is allow-listed, else 403 + logs DENY. No decryption —
  the target host is in the CONNECT line / Host header in the clear.

So a client that honours the system/HTTPS_PROXY reaches allow-listed hosts and
is refused others; a client that ignores the proxy and dials out directly is
dropped by the firewall. Either way only the model endpoint is reachable.

Stdlib-only (Python 3.8+), cross-platform; the pure helpers are unit-testable on
the host. The build_setup_ps1 installer ships + launches this on the guest.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

_IO_CHUNK = 65536
_HEAD_TIMEOUT_S = 10.0
_CONNECT_TIMEOUT_S = 10.0

logger = logging.getLogger("winguard")


# ───────────────────────────── hostname parsing ────────────────────────────

def parse_target(head: bytes) -> tuple[str | None, str | None, int]:
    """From the first HTTP request bytes return (method, host, port).

    Handles ``CONNECT host:port`` (HTTPS tunnel) and plain HTTP where the host
    comes from an absolute request-URI or the ``Host`` header. Returns
    (None, None, 0) if unparseable (caller denies — fail-closed)."""
    try:
        line0 = head.split(b"\r\n", 1)[0].decode("latin-1", "ignore")
        parts = line0.split()
        if len(parts) < 2:
            return None, None, 0
        method, target = parts[0].upper(), parts[1]
        if method == "CONNECT":
            host, _, port = target.partition(":")
            return "CONNECT", host.lower() or None, int(port or 443)
        # plain HTTP: absolute-form (http://host/..) or Host header
        host = None
        if "://" in target:
            host = target.split("://", 1)[1].split("/", 1)[0]
        if not host:
            for ln in head.split(b"\r\n")[1:]:
                if ln[:5].lower() == b"host:":
                    host = ln[5:].strip().decode("latin-1", "ignore")
                    break
        if not host:
            return method, None, 0
        h, _, p = host.partition(":")
        return method, h.lower() or None, int(p or 80)
    except Exception:
        return None, None, 0


def host_allowed(host: str | None, allow: "frozenset[str]") -> bool:
    """True iff *host* equals or is a subdomain of an allow-list entry.
    Case/trailing-dot-insensitive; empty host or empty allow → False."""
    if not host or not allow:
        return False
    h = host.strip().rstrip(".").lower()
    for entry in allow:
        e = entry.strip().rstrip(".").lower()
        if e and (h == e or h.endswith("." + e)):
            return True
    return False


# ───────────────────────────── proxy core ──────────────────────────────────

class Proxy:
    def __init__(self, allow: "frozenset[str]", audit: logging.Logger):
        self.allow = allow
        self.audit = audit

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), _HEAD_TIMEOUT_S)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, OSError):
            _close(writer)
            return
        method, host, port = parse_target(head)
        if not host_allowed(host, self.allow):
            self.audit.info("DENY %s %s:%s", method, host, port)
            try:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
            except OSError:
                pass
            _close(writer)
            return
        self.audit.info("ALLOW %s %s:%s", method, host, port)

        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), _CONNECT_TIMEOUT_S
            )
        except (asyncio.TimeoutError, OSError) as exc:
            self.audit.info("UPSTREAM-FAIL %s:%s %s", host, port, exc)
            _close(writer)
            return

        if method == "CONNECT":
            # tunnel: tell the client the tunnel is up, then splice raw bytes
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            try:
                await writer.drain()
            except OSError:
                pass
        else:
            # plain HTTP: replay the request head we already consumed
            up_writer.write(head)
            try:
                await up_writer.drain()
            except OSError:
                pass

        await asyncio.gather(
            _pump(reader, up_writer), _pump(up_reader, writer),
            return_exceptions=True,
        )


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
        _close(dst)


def _close(w: asyncio.StreamWriter) -> None:
    try:
        w.close()
    except OSError:
        pass


def _load_allow(path: str) -> "frozenset[str]":
    hosts = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    hosts.add(s.lower())
    except OSError:
        pass
    return frozenset(hosts)


async def _serve(port: int, allow: "frozenset[str]", audit: logging.Logger) -> None:
    proxy = Proxy(allow, audit)
    server = await asyncio.start_server(proxy.handle, "127.0.0.1", port)
    audit.info("winguard listening on 127.0.0.1:%d allow=%s", port, sorted(allow))
    async with server:
        await server.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward CONNECT allow-list proxy")
    ap.add_argument("--port", type=int, default=15599)
    ap.add_argument("--allow-file", required=True)
    ap.add_argument("--log-file", default="")
    args = ap.parse_args(argv)

    audit = logging.getLogger("winguard.audit")
    audit.setLevel(logging.INFO)
    handler = logging.FileHandler(args.log_file) if args.log_file else logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    audit.addHandler(handler)

    try:
        asyncio.run(_serve(args.port, _load_allow(args.allow_file), audit))
    except KeyboardInterrupt:
        return 0
    return 0


# ───────────────────────── Windows setup builder ───────────────────────────

PROXY_PORT = 15599


def build_setup_ps1(mode: str, allow_hosts: "list[str]", proxy_b64: str) -> str:
    """PowerShell (run elevated) that installs Windows egress enforcement.

    allowlist: default-deny outbound firewall + allow (proxy process, DNS);
      launch winguard; point the system/WinHTTP/env proxy at it.
    off: default-deny outbound with no proxy/allow — full air-gap (loopback and
      the stateful reply path for the inbound cua channel still work)."""
    allow_lines = "\n".join(sorted(set(h.strip().lower() for h in allow_hosts if h.strip())))
    port = PROXY_PORT
    launch = ""
    proxy_cfg = ""
    if mode != "off":
        launch = f"""
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {{ $py = 'C:\\ale-run\\.venv\\Scripts\\python.exe' }}
[IO.File]::WriteAllBytes('C:\\aleguard\\winguard.py', [Convert]::FromBase64String('{proxy_b64}'))
Set-Content -Path 'C:\\aleguard\\allow' -Value @'
{allow_lines}
'@ -Encoding ascii
Get-Process python -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -eq $py }} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Process -FilePath $py -ArgumentList '"C:\\aleguard\\winguard.py"','--port','{port}','--allow-file','"C:\\aleguard\\allow"','--log-file','"C:\\aleguard\\winguard.log"' -WindowStyle Hidden
Start-Sleep -Milliseconds 1500
$listening = $false
for ($i=0; $i -lt 40; $i++) {{ if (Test-NetConnection -ComputerName 127.0.0.1 -Port {port} -InformationLevel Quiet -WarningAction SilentlyContinue) {{ $listening = $true; break }}; Start-Sleep -Milliseconds 250 }}
if (-not $listening) {{ Write-Output 'ALEGUARD_PROXY_FAILED'; Get-Content 'C:\\aleguard\\winguard.log' -ErrorAction SilentlyContinue; exit 1 }}
New-NetFirewallRule -DisplayName 'aleguard-proxy' -Direction Outbound -Program $py -Action Allow -Profile Any | Out-Null
"""
        proxy_cfg = f"""
netsh winhttp set proxy proxy-server="127.0.0.1:{port}" bypass-list="localhost;127.0.0.1" | Out-Null
$ienv = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings'
Set-ItemProperty -Path $ienv -Name ProxyEnable -Value 1
Set-ItemProperty -Path $ienv -Name ProxyServer -Value '127.0.0.1:{port}'
[Environment]::SetEnvironmentVariable('HTTPS_PROXY','http://127.0.0.1:{port}','Machine')
[Environment]::SetEnvironmentVariable('HTTP_PROXY','http://127.0.0.1:{port}','Machine')
[Environment]::SetEnvironmentVariable('NODE_USE_ENV_PROXY','1','Machine')
"""
    return f"""
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path 'C:\\aleguard' | Out-Null
# DNS allow (before default-deny so resolution keeps working)
New-NetFirewallRule -DisplayName 'aleguard-dns-udp' -Direction Outbound -Protocol UDP -RemotePort 53 -Action Allow -Profile Any | Out-Null
New-NetFirewallRule -DisplayName 'aleguard-dns-tcp' -Direction Outbound -Protocol TCP -RemotePort 53 -Action Allow -Profile Any | Out-Null
{launch}
# Default-deny all other outbound. Keep inbound DEFAULT-ALLOW so the cua control
# channel (host->VM ingress) is never severed — inbound is already gated by the
# cloud network firewall. This is the hard, client-agnostic blocking guarantee.
Set-NetFirewallProfile -All -Enabled True -DefaultInboundAction Allow -DefaultOutboundAction Block
{proxy_cfg}
Write-Output 'ALEGUARD_OK'
"""


def build_teardown_ps1() -> str:
    return (
        "Set-NetFirewallProfile -All -DefaultOutboundAction Allow -ErrorAction SilentlyContinue\n"
        "Remove-NetFirewallRule -DisplayName 'aleguard-*' -ErrorAction SilentlyContinue\n"
        "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*python*' -and (Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.Id)\").CommandLine -like '*winguard.py*' } | Stop-Process -Force -ErrorAction SilentlyContinue\n"
        "netsh winhttp reset proxy | Out-Null\n"
        "Write-Output 'ALEGUARD_CLEARED'\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
