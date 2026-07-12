"""Unit tests for aleguard's pure helpers (host, not VM):
- extract_sni against a REAL TLS ClientHello (generated via ssl.MemoryBIO)
- extract_http_host against a real HTTP request head
- host_allowed suffix/fail-closed semantics

Run: ``python tests/test_aleguard.py``.
"""
import os
import ssl
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.environments.netguard.aleguard import (
    PROXY_PORT,
    build_nft_ruleset,
    build_setup_script,
    build_teardown_script,
    extract_http_host,
    extract_sni,
    host_allowed,
)


def _real_client_hello(server_hostname: str) -> bytes:
    """Drive a TLS handshake through memory BIOs far enough to capture the
    ClientHello bytes the client emits (with SNI = server_hostname)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    inb, outb = ssl.MemoryBIO(), ssl.MemoryBIO()
    sslobj = ctx.wrap_bio(inb, outb, server_hostname=server_hostname)
    try:
        sslobj.do_handshake()
    except ssl.SSLWantReadError:
        pass
    return outb.read()


def test_extract_sni_real_clienthello():
    for host in ("openrouter.ai", "api.anthropic.com", "yunwu.ai"):
        ch = _real_client_hello(host)
        got = extract_sni(ch)
        assert got == host, f"expected {host!r}, got {got!r}"


def test_extract_sni_garbage_is_none():
    assert extract_sni(b"") is None
    assert extract_sni(b"\x16\x03\x01\x00\x05hello") is None
    assert extract_sni(b"not a tls record at all") is None


def test_extract_http_host():
    req = b"GET /search?q=answer HTTP/1.1\r\nHost: www.google.com\r\nAccept: */*\r\n\r\n"
    assert extract_http_host(req) == "www.google.com"
    req2 = b"GET / HTTP/1.1\r\nHost: example.com:8080\r\n\r\n"
    assert extract_http_host(req2) == "example.com"
    assert extract_http_host(b"GET / HTTP/1.1\r\n\r\n") is None


def test_host_allowed_suffix_and_failclosed():
    allow = frozenset({"openrouter.ai", "api.anthropic.com"})
    assert host_allowed("openrouter.ai", allow)
    assert host_allowed("api.openrouter.ai", allow)      # subdomain
    assert host_allowed("API.ANTHROPIC.COM", allow)       # case-insensitive
    assert host_allowed("api.anthropic.com.", allow)      # trailing dot
    assert not host_allowed("google.com", allow)
    assert not host_allowed("notopenrouter.ai", allow)    # not a real suffix
    assert not host_allowed("evil-openrouter.ai", allow)
    assert not host_allowed(None, allow)                  # fail-closed
    assert not host_allowed("openrouter.ai", frozenset()) # empty allow → deny


def test_nft_allowlist_shape():
    r = build_nft_ruleset("allowlist")
    assert "flush ruleset" in r
    assert f"redirect to :{PROXY_PORT}" in r          # 80/443 → proxy
    assert "meta skuid __UID__ return" in r            # proxy's own egress not redirected
    assert "meta skuid __UID__ accept" in r            # proxy egress permitted
    assert "policy drop" in r                          # default-deny output
    assert "ip daddr 127.0.0.0/8 accept" in r          # redirected pkts reach the proxy
    assert "udp dport 53 accept" in r                  # DNS resolution allowed
    assert "table ip6 aleguard6" in r and "reject" in r  # IPv6 killed


def test_nft_off_is_airgap():
    r = build_nft_ruleset("off")
    assert "redirect" not in r                         # no proxy path
    assert "dport 53" not in r                          # even DNS blocked
    assert "policy drop" in r
    assert "ct state established,related accept" in r   # cua replies survive


def test_setup_script_allowlist():
    s = build_setup_script("allowlist", ["openrouter.ai", "openrouter.ai"], "UE9DPQ==")
    assert "useradd -r -M -s /usr/sbin/nologin aleproxy" in s
    assert "ALEUID=$(id -u aleproxy)" in s
    assert "openrouter.ai" in s and s.count("openrouter.ai") == 1  # deduped
    assert "runuser -u aleproxy" in s and "aleguard.py" in s        # proxy launched
    assert "nft -f /etc/aleguard/rules.nft" in s
    assert "$ALEUID" in s and "__UID__" not in s                    # uid substituted
    assert "ALEGUARD_OK" in s


def test_setup_script_off_has_no_proxy():
    s = build_setup_script("off", [], "")
    assert "runuser" not in s                            # air-gap: no proxy process
    assert "nft -f" in s and "ALEGUARD_OK" in s


def test_teardown_restores():
    t = build_teardown_script()
    assert "nft delete table ip aleguard" in t
    assert "nft delete table ip6 aleguard6" in t
    assert "pkill -f aleguard.py" in t


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
