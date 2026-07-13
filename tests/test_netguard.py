"""Unit tests for the unified netguard proxy + setup builders (host, not VM).
Run: ``python tests/test_netguard.py``.
"""
import os
import ssl
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.environments.netguard.proxy import extract_sni, resolve_ip
from ale_run.environments.netguard.setup import (
    build_linux,
    build_linux_teardown,
    build_windows,
    build_windows_teardown,
)


def _client_hello(host: str) -> bytes:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    inb, outb = ssl.MemoryBIO(), ssl.MemoryBIO()
    obj = ctx.wrap_bio(inb, outb, server_hostname=host)
    try:
        obj.do_handshake()
    except ssl.SSLWantReadError:
        pass
    return outb.read()


def test_extract_sni_real():
    for h in ("openrouter.ai", "api.anthropic.com", "yunwu.ai"):
        assert extract_sni(_client_hello(h)) == h


def test_extract_sni_garbage():
    assert extract_sni(b"") is None
    assert extract_sni(b"not tls") is None


def test_resolve_ip_exact_and_subdomain_failclosed():
    amap = {"openrouter.ai": ["1.2.3.4"], "api.anthropic.com": ["5.6.7.8"]}
    assert resolve_ip("openrouter.ai", amap) == "1.2.3.4"
    assert resolve_ip("cdn.openrouter.ai", amap) == "1.2.3.4"   # subdomain
    assert resolve_ip("OPENROUTER.AI.", amap) == "1.2.3.4"       # case/dot
    assert resolve_ip("google.com", amap) is None               # not allowed
    assert resolve_ip("evil-openrouter.ai", amap) is None       # not a suffix
    assert resolve_ip(None, amap) is None
    assert resolve_ip("openrouter.ai", {}) is None


def test_linux_allowlist_shape():
    s = build_linux("allowlist", ["openrouter.ai"], "UE9D")
    assert "ip_unprivileged_port_start=443" in s                 # non-root bind 443
    assert "--resolve 'openrouter.ai' --map" in s                # resolve before hosts
    assert "127.0.0.1 $h # netguard" in s                        # hosts override
    assert "127.0.0.1:443" in s and "runuser -u netguard" in s   # proxy launch
    assert "policy drop" in s and "meta skuid $NGUID accept" in s  # deny-all except proxy
    assert "udp dport 53 accept" in s                            # DNS
    assert "NETGUARD_OK" in s


def test_linux_off_is_airgap():
    s = build_linux("off", [], "")
    assert "proxy.py" not in s and "127.0.0.1 $h" not in s        # no proxy/hosts
    assert "meta skuid" not in s and "dport 53" not in s          # nothing allowed out
    assert "policy drop" in s and "ct state established,related accept" in s  # cua survives


def test_windows_allowlist_shape():
    s = build_windows("allowlist", ["yunwu.ai"], "UE9D")
    assert "--resolve 'yunwu.ai' --map" in s
    assert "127.0.0.1 yunwu.ai # netguard" in s and "flushdns" in s
    assert "netguard-proxy" in s and "127.0.0.1 -Port 443" in s
    assert "DefaultOutboundAction Block" in s and "DefaultInboundAction Allow" in s
    assert "RemotePort 53 -Action Allow" in s
    assert "NETGUARD_OK" in s


def test_windows_off_has_no_proxy():
    s = build_windows("off", [], "")
    assert "proxy.py" not in s and "flushdns" not in s
    assert "DefaultOutboundAction Block" in s and "NETGUARD_OK" in s


def test_teardowns():
    assert "nft delete table inet netguard" in build_linux_teardown()
    assert "DefaultOutboundAction Allow" in build_windows_teardown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
