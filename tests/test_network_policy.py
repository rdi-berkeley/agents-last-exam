"""Unit tests for the per-task network policy (task_card ``vm.network``).

Covers:
- NetworkPolicy.from_card parsing + validation (absent → open, bad values raise)
- effective_allow: model host unioned only in allowlist mode, empty for off
- model_host_from_env: derive endpoint host from injected base-URL vars
- Provider.assert_network_supported: fail CLOSED on a non-open policy when the
  provider does not enforce, and pass when open or when enforcement is declared.

Run: ``python tests/test_network_policy.py``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ale_run.base_interface.sandbox import (
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
    Provider,
    SandboxSpec,
)


# ─────────────────────────── from_card ───────────────────────────

def test_from_card_absent_is_open():
    assert NetworkPolicy.from_card(None) == NetworkPolicy()
    assert NetworkPolicy.from_card(None).mode == "open"


def test_from_card_allowlist_with_hosts():
    p = NetworkPolicy.from_card({"mode": "allowlist", "allow": ["pypi.org", " x.io "]})
    assert p.mode == "allowlist"
    assert p.allow == ("pypi.org", "x.io")  # trimmed


def test_from_card_off():
    assert NetworkPolicy.from_card({"mode": "off"}).mode == "off"


def test_from_card_bad_mode_raises():
    for bad in ({"mode": "deny"}, {"mode": "blacklist"}, {"mode": 1}):
        try:
            NetworkPolicy.from_card(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_from_card_bad_allow_raises():
    for bad in ({"allow": "pypi.org"}, {"allow": [""]}, {"allow": [123]}):
        try:
            NetworkPolicy.from_card(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_from_card_non_object_raises():
    try:
        NetworkPolicy.from_card(["allowlist"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-object network")


# ─────────────────────────── effective_allow ───────────────────────────

def test_effective_allow_unions_model_host_in_allowlist():
    p = NetworkPolicy(mode="allowlist", allow=("pypi.org",))
    assert p.effective_allow("openrouter.ai") == frozenset({"pypi.org", "openrouter.ai"})


def test_effective_allow_off_is_airgap():
    p = NetworkPolicy(mode="off", allow=("pypi.org",))
    # off never adds the model host — true air-gap for host-side agents.
    assert p.effective_allow("openrouter.ai") == frozenset({"pypi.org"})


def test_effective_allow_no_model_host():
    p = NetworkPolicy(mode="allowlist", allow=("pypi.org",))
    assert p.effective_allow(None) == frozenset({"pypi.org"})


# ─────────────────────────── model_host_from_env ───────────────────────────

def test_model_host_from_env_anthropic():
    env = {"ANTHROPIC_BASE_URL": "https://openrouter.ai/api"}
    assert NetworkPolicy.model_host_from_env(env) == "openrouter.ai"


def test_model_host_from_env_precedence_and_absent():
    assert NetworkPolicy.model_host_from_env({}) is None
    env = {"OPENAI_BASE_URL": "https://api.openai.com/v1"}
    assert NetworkPolicy.model_host_from_env(env) == "api.openai.com"


# ─────────────────────────── fail-closed guard ───────────────────────────

class _DummyProvider(Provider):
    enforces_network_policy = False

    async def acquire(self, spec):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def release(self, sandbox, *, mode="delete"):  # pragma: no cover
        raise NotImplementedError

    def open_session(self, sandbox):  # pragma: no cover
        raise NotImplementedError


class _EnforcingProvider(_DummyProvider):
    enforces_network_policy = True


def _spec(mode):
    return SandboxSpec(snapshot="cpu-free", network=NetworkPolicy(mode=mode))


def test_open_policy_always_ok():
    _DummyProvider().assert_network_supported(_spec("open"))  # no raise


def test_non_open_on_unsupported_provider_fails_closed():
    for mode in ("allowlist", "off"):
        try:
            _DummyProvider().assert_network_supported(_spec(mode))
        except NetworkPolicyUnsupportedError:
            pass
        else:
            raise AssertionError(f"expected fail-closed for mode={mode}")


def test_non_open_on_enforcing_provider_ok():
    _EnforcingProvider().assert_network_supported(_spec("allowlist"))  # no raise


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
