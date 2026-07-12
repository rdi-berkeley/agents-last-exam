"""Unit tests for winguard's pure helpers (host, not VM).
Run: ``python tests/test_winguard.py``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.environments.netguard.winguard import (
    PROXY_PORT,
    build_setup_ps1,
    build_teardown_ps1,
    host_allowed,
    parse_target,
)


def test_parse_connect():
    m, h, p = parse_target(b"CONNECT openrouter.ai:443 HTTP/1.1\r\nHost: openrouter.ai\r\n\r\n")
    assert (m, h, p) == ("CONNECT", "openrouter.ai", 443)


def test_parse_connect_default_port():
    m, h, p = parse_target(b"CONNECT api.anthropic.com HTTP/1.1\r\n\r\n")
    assert (m, h, p) == ("CONNECT", "api.anthropic.com", 443)


def test_parse_plain_http_host_header():
    m, h, p = parse_target(b"GET /x HTTP/1.1\r\nHost: www.google.com\r\n\r\n")
    assert (m, h, p) == ("GET", "www.google.com", 80)


def test_parse_plain_http_absolute():
    m, h, p = parse_target(b"GET http://example.com:8080/y HTTP/1.1\r\n\r\n")
    assert (m, h, p) == ("GET", "example.com", 8080)


def test_parse_garbage():
    assert parse_target(b"") == (None, None, 0)
    assert parse_target(b"garbage\r\n\r\n") == (None, None, 0)


def test_host_allowed_suffix_failclosed():
    allow = frozenset({"openrouter.ai"})
    assert host_allowed("openrouter.ai", allow)
    assert host_allowed("api.openrouter.ai", allow)
    assert host_allowed("OPENROUTER.AI.", allow)
    assert not host_allowed("google.com", allow)
    assert not host_allowed("evil-openrouter.ai", allow)
    assert not host_allowed(None, allow)
    assert not host_allowed("openrouter.ai", frozenset())


def test_setup_ps1_allowlist_shape():
    s = build_setup_ps1("allowlist", ["openrouter.ai"], "UE9DPQ==")
    assert "DefaultOutboundAction Block" in s              # hard deny-all outbound
    assert "aleguard-proxy" in s and "-Action Allow" in s   # proxy allowed out
    assert "RemotePort 53 -Action Allow" in s               # DNS allowed
    assert "winguard.py" in s and str(PROXY_PORT) in s      # proxy launched
    assert "netsh winhttp set proxy" in s                   # system proxy
    assert "NODE_USE_ENV_PROXY" in s                        # node proxy opt-in
    assert "openrouter.ai" in s
    assert "ALEGUARD_OK" in s


def test_setup_ps1_off_has_no_proxy():
    s = build_setup_ps1("off", [], "")
    assert "DefaultOutboundAction Block" in s               # air-gap
    assert "winguard.py" not in s and "winhttp set proxy" not in s
    assert "ALEGUARD_OK" in s


def test_teardown_ps1():
    t = build_teardown_ps1()
    assert "DefaultOutboundAction Allow" in t
    assert "Remove-NetFirewallRule" in t


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
