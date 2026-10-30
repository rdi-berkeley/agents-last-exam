from __future__ import annotations

import asyncio
import base64
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import qemu as qemu_module
from ale_run.environments.providers.qemu import QemuProvider
from ale_run.executors import _sandbox_entry
from ale_run.executors.sandbox import SandboxExecutor
from ale_run.networking import egress_gateway, probe
from ale_run.networking.egress_gateway import (
    EgressGateway,
    allowlist_host_from_url,
    canonical_host,
)
from ale_run.orchestration.config_loader import load_experiment
from ale_run.orchestration.lifecycle import _build_env_spec, _plan_network
from ale_run.tasks.loader import TaskLoader


def _task(tmp_path: Path, vm: dict) -> TaskLoader:
    task = tmp_path / "task"
    task.mkdir()
    (task / "main.py").write_text(
        "class _Config:\n"
        "    task_description = 'solve'\n\n"
        "    def to_metadata(self):\n"
        "        return {}\n\n"
        "config = _Config()\n",
        encoding="utf-8",
    )
    (task / "task_card.json").write_text(json.dumps({"vm": vm}), encoding="utf-8")
    return TaskLoader(str(task))


@pytest.mark.parametrize(
    ("network", "expected"),
    [(None, "on"), ({"mode": "on"}, "on"), ({"mode": "off"}, "off")],
)
def test_task_card_network_mode(tmp_path: Path, network: dict | None, expected: str) -> None:
    vm = {"snapshot": "cpu-free-ubuntu"} | ({"network": network} if network else {})

    meta = _task(tmp_path, vm).load()

    assert meta["network_mode"] == expected
    assert _build_env_spec(meta).network_mode == expected


@pytest.mark.parametrize("network", [{"mode": "Off"}, {}, "off", None])
def test_task_card_rejects_malformed_network(tmp_path: Path, network: object) -> None:
    with pytest.raises(ValueError, match="vm.network"):
        _task(tmp_path, {"snapshot": "cpu-free-ubuntu", "network": network}).load()


def _experiment(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "agent.yaml").write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    (tmp_path / "environment.yaml").write_text(
        "provider: static\nendpoint: http://127.0.0.1:5000\n", encoding="utf-8",
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "name: network-off\nagent: agent.yaml\nenvironment: environment.yaml\n"
        "tasks:\n  - path: demo/hello\n" + extra,
        encoding="utf-8",
    )
    return path


def test_experiment_network_defaults_leave_runs_unsealed(tmp_path: Path) -> None:
    spec = load_experiment(_experiment(tmp_path))

    assert spec.network_off_policy == "allow_unenforced"
    assert spec.network_allowlist == ()


def test_experiment_network_allowlist_is_canonical(tmp_path: Path) -> None:
    spec = load_experiment(_experiment(
        tmp_path,
        "network_off_policy: enforce\nnetwork_allowlist: [OpenRouter.AI., api.openai.com]\n",
    ))

    assert spec.network_off_policy == "enforce"
    assert spec.network_allowlist == ("api.openai.com", "openrouter.ai")


@pytest.mark.parametrize(
    "extra",
    [
        "network_off_policy: strict\n",
        "network_allowlist: [https://api.openai.com]\n",
        "network_allowlist: api.openai.com\n",
    ],
)
def test_experiment_rejects_bad_network_settings(tmp_path: Path, extra: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        load_experiment(_experiment(tmp_path, extra))


@pytest.mark.parametrize(
    "value",
    ["1.1.1.1", "::1", "localhost", "*.openai.com", "api.openai.com:443", "https://a.com", ""],
)
def test_canonical_host_accepts_only_exact_names(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_host(value)


def test_allowlist_host_from_url() -> None:
    assert (
        allowlist_host_from_url("https://ale-super-server-1.tail8363db.ts.net/v1")
        == "ale-super-server-1.tail8363db.ts.net"
    )
    for url in ("http://api.openai.com/v1", "https://api.openai.com:8443/v1"):
        with pytest.raises(ValueError):
            allowlist_host_from_url(url)


@pytest.mark.parametrize(
    ("address", "usable"),
    [
        ("104.18.2.1", True),
        ("100.101.102.103", True),
        ("127.0.0.1", False),
        ("169.254.169.254", False),
        ("::ffff:127.0.0.1", False),
        ("::1", False),
    ],
)
def test_gateway_upstream_address_filter(address: str, usable: bool) -> None:
    assert egress_gateway._usable_upstream(address) is usable


def _echo_upstream() -> socket.socket:
    server = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                while data := conn.recv(1024):
                    conn.sendall(data.upper())

    threading.Thread(target=serve, daemon=True).start()
    return server


def _send(address: tuple[str, int], request: bytes) -> tuple[bytes, socket.socket]:
    sock = socket.create_connection(address, timeout=5)
    sock.sendall(request)
    head = b""
    while b"\r\n\r\n" not in head and (chunk := sock.recv(1024)):
        head += chunk
    return head, sock


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch):
    upstream = _echo_upstream()
    monkeypatch.setattr(
        egress_gateway, "_connect_upstream",
        lambda host: socket.create_connection(upstream.getsockname()),
    )
    gateway = EgressGateway(["api.example.com"], bind_host="127.0.0.1")
    gateway.start()
    yield gateway
    gateway.close()
    upstream.close()


def test_gateway_tunnels_only_allowlisted_hosts(gateway: EgressGateway) -> None:
    head, sock = _send(gateway.address, b"CONNECT API.example.com:443 HTTP/1.1\r\n\r\n")
    with sock:
        assert head.startswith(b"HTTP/1.1 200")
        sock.sendall(b"hello")
        assert sock.recv(1024) == b"HELLO"

    for request, status in (
        (b"CONNECT example.com:443 HTTP/1.1\r\n\r\n", b"403"),
        (b"CONNECT api.example.com:80 HTTP/1.1\r\n\r\n", b"403"),
        (b"GET http://api.example.com/ HTTP/1.1\r\n\r\n", b"405"),
    ):
        head, sock = _send(gateway.address, request)
        sock.close()
        assert head.split()[1] == status

    assert gateway.summary() == {
        "allowed": {"api.example.com:443": 1},
        "denied": {
            "example.com:443": 1,
            "api.example.com:80": 1,
            "http://api.example.com/": 1,
        },
    }


def test_probe_script_reads_gateway_answers(gateway: EgressGateway) -> None:
    host, port = gateway.address
    encoded = base64.b64encode(probe._PROBE_PY.encode()).decode()
    result = subprocess.run(
        [sys.executable, "-c", f"import base64;exec(base64.b64decode('{encoded}'))",
         host, str(port), "api.example.com"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    report = json.loads(result.stdout.split("ALE_NETWORK_PROBE:", 1)[1])

    assert report["unlisted_status"] == 403
    assert report["allowlisted_status"] == {"api.example.com": 200}
    assert set(report["direct_tcp"]) == {"1.1.1.1", "8.8.8.8"}
    assert set(report["dns"]) == {host, "1.1.1.1", "8.8.8.8"}


_SEALED_REPORT = {
    "direct_tcp": {"1.1.1.1": False, "8.8.8.8": False},
    "dns": {"172.30.0.1": False, "1.1.1.1": False, "8.8.8.8": False},
    "unlisted_status": 403,
    "allowlisted_status": {"api.example.com": 200},
}


class _ProbeSandbox:
    python = "/usr/bin/python3"
    is_linux = True

    def __init__(self, report: dict) -> None:
        self.report = report
        self.commands: list[str] = []

    async def run_command(self, command: str, timeout: float | None = None):
        self.commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="noise\nALE_NETWORK_PROBE:" + json.dumps(self.report),
            stderr="",
        )


def _run_probe(sandbox: _ProbeSandbox) -> dict:
    return asyncio.run(probe.probe_sealed_guest(
        sandbox, proxy_host="172.30.0.1", proxy_port=3128, allowlist=["api.example.com"],
    ))


def test_probe_accepts_a_sealed_guest() -> None:
    sandbox = _ProbeSandbox(_SEALED_REPORT)

    assert _run_probe(sandbox) == _SEALED_REPORT
    assert sandbox.commands[0].endswith(" 172.30.0.1 3128 api.example.com")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"direct_tcp": {"1.1.1.1": True, "8.8.8.8": False}}, "direct TCP to 1.1.1.1"),
        ({"dns": {"172.30.0.1": True, "1.1.1.1": False, "8.8.8.8": False}}, "DNS server 172.30.0.1"),
        ({"unlisted_status": 200}, "unlisted host"),
        ({"allowlisted_status": {"api.example.com": 502}}, "allowlisted api.example.com"),
    ],
)
def test_probe_rejects_any_other_path(change: dict, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _run_probe(_ProbeSandbox(_SEALED_REPORT | change))


def test_qemu_seal_network_routes_guest_only_to_its_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str, check: bool = True) -> tuple[int, str, str]:
        calls.append(args)
        return 0, "127.0.0.1 \n" if args[0] == "inspect" else "", ""

    async def fake_probe(sandbox, *, proxy_host, proxy_port, allowlist):
        return {"seen": [proxy_host, proxy_port, list(allowlist)]}

    monkeypatch.setattr(qemu_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(qemu_module, "probe_sealed_guest", fake_probe)
    provider = QemuProvider({"snapshots": {"cpu-free": {
        "image": "ale-win10", "disk_source": str(tmp_path / "disk.qcow2"),
        "root": str(tmp_path / "qemu"),
    }}})
    sandbox = SimpleNamespace(id="ale-qemu-x", metadata={"container_name": "ale-qemu-x"})

    report = asyncio.run(provider.seal_network(sandbox, ["api.example.com"]))

    gateway_port = int(report["gateway"].rsplit(":", 1)[1])
    assert report["guest_proxy_url"] == "http://172.30.0.1:3128"
    assert report["probe"] == {"seen": ["172.30.0.1", 3128, ["api.example.com"]]}
    script = next(call[-1] for call in calls if call[0] == "exec")
    assert f"--dport 3128 -j DNAT --to-destination 127.0.0.1:{gateway_port}" in script
    assert "iptables -I FORWARD 1 -i docker -j DROP" in script
    assert "ip6tables -I FORWARD 1 -i docker -j DROP" in script
    assert provider.network_summary(sandbox) == {"allowed": {}, "denied": {}}
    socket.create_connection(("127.0.0.1", gateway_port), timeout=2).close()

    asyncio.run(provider.release(sandbox, mode="keep"))

    assert provider.network_summary(sandbox) is None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", gateway_port), timeout=2)


class _Writer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit_event(self, name: str, **fields) -> None:
        self.events.append((name, fields))


class _SealingProvider:
    async def seal_network(self, sandbox, allowlist):
        raise NotImplementedError


def _plan(**overrides):
    kwargs = {
        "env_spec": SandboxSpec(snapshot="cpu-free-ubuntu", network_mode="off"),
        "provider": _SealingProvider(),
        "executor_type": "sandbox",
        "config": SimpleNamespace(base_url="https://Gateway.Example.com/v1"),
        "policy": "enforce",
        "extra_allowlist": ("api.example.com",),
        "writer": _Writer(),
    } | overrides
    return _plan_network(**kwargs)


def test_plan_network_leaves_network_on_units_alone() -> None:
    assert _plan(env_spec=SandboxSpec(snapshot="cpu-free-ubuntu")) is None


def test_plan_network_records_unenforced_runs() -> None:
    writer = _Writer()

    assert _plan(policy="allow_unenforced", writer=writer) == {"mode": "off", "enforced": False}
    assert writer.events == [
        ("network_off_unenforced", {"reason": "network_off_policy=allow_unenforced"}),
    ]


def test_plan_network_allowlists_model_endpoint_and_extras() -> None:
    assert _plan() == {
        "mode": "off",
        "enforced": True,
        "allowlist": ["api.example.com", "gateway.example.com"],
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"executor_type": "local"}, "outside the sandbox"),
        ({"provider": object()}, "cannot seal"),
        ({"config": SimpleNamespace(base_url="http://10.0.0.5:8000/v1")}, "https://"),
        ({"config": SimpleNamespace(), "extra_allowlist": ()}, "network_allowlist"),
    ],
)
def test_plan_network_enforce_fails_before_any_vm(overrides: dict, message: str) -> None:
    with pytest.raises((RuntimeError, ValueError), match=message):
        _plan(**overrides)


def test_entry_waits_for_seal_and_returns_proxy_env(tmp_path: Path) -> None:
    (tmp_path / "_egress_sealed").write_text(json.dumps({"proxy_url": "http://172.30.0.1:3128"}))

    env = _sandbox_entry._await_egress_seal(tmp_path)

    assert (tmp_path / "_egress_request").exists()
    assert env["HTTPS_PROXY"] == env["http_proxy"] == "http://172.30.0.1:3128"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert env["NODE_USE_ENV_PROXY"] == "1"


def test_entry_aborts_when_seal_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "_egress_failed").write_text("RuntimeError: probe failed")
    with pytest.raises(RuntimeError, match="probe failed"):
        _sandbox_entry._await_egress_seal(tmp_path)

    (tmp_path / "_egress_failed").unlink()
    monkeypatch.setattr(_sandbox_entry, "_EGRESS_SEAL_TIMEOUT_S", 0.0)
    with pytest.raises(TimeoutError):
        _sandbox_entry._await_egress_seal(tmp_path)


class _HandshakeSandbox:
    def __init__(self, requested: bool) -> None:
        self.requested = requested
        self.files: dict[str, str] = {}

    async def exists(self, path: str) -> bool:
        return self.requested

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def _executor(sandbox: _HandshakeSandbox, seal) -> SandboxExecutor:
    executor = SandboxExecutor(config=SimpleNamespace(), work_dir="/w", sandbox=sandbox)
    executor.on_egress_request = seal
    return executor


def test_executor_seals_once_the_entry_asks() -> None:
    async def seal() -> str:
        return "http://172.30.0.1:3128"

    waiting = _HandshakeSandbox(requested=False)
    assert asyncio.run(_executor(waiting, seal)._answer_egress_request("/w/_egress_request", "/w", "/"))
    assert waiting.files == {}

    asked = _HandshakeSandbox(requested=True)
    assert not asyncio.run(_executor(asked, seal)._answer_egress_request("/w/_egress_request", "/w", "/"))
    assert asked.files == {"/w/_egress_sealed": '{"proxy_url": "http://172.30.0.1:3128"}'}


def test_executor_reports_seal_failure_to_the_entry() -> None:
    async def seal() -> str:
        raise RuntimeError("probe failed")

    asked = _HandshakeSandbox(requested=True)
    assert not asyncio.run(_executor(asked, seal)._answer_egress_request("/w/_egress_request", "/w", "/"))
    assert asked.files == {"/w/_egress_failed": "RuntimeError: probe failed"}
