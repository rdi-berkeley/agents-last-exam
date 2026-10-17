import asyncio
import dataclasses
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.experiment_spec import AgentSpec
from ale_run.orchestration.factory import build_config, resolve_agent


HARNESSES = [
    ("pi_cli", "PiCli"),
    ("qwen_code", "QwenCode"),
    ("zcode", "ZCode"),
]


def adapter(name, stem, tmp_path, **kwargs):
    module = importlib.import_module(f"ale_run.agents.{name}.deployer")
    config_cls = getattr(importlib.import_module(f"ale_run.agents.{name}.config"), f"{stem}Config")
    config = config_cls(**kwargs)
    executor = SimpleNamespace(
        work_dir=str(tmp_path),
        config=config,
        env={},
        sandbox=SimpleNamespace(is_linux=True, node="node", mcp_server_dir="/absent-mcp"),
        cua_bridge_url=lambda: "http://127.0.0.1:8000",
    )
    return module, getattr(module, f"{stem}Deployer")(executor)


def parse(name, stem, tmp_path, events, **kwargs):
    module, deployer = adapter(name, stem, tmp_path, **kwargs)
    filename = "transcript.jsonl" if name == "pi_cli" else "stdout.log"
    (tmp_path / filename).write_text("\n".join(json.dumps(event) for event in events))
    builder = TrajectoryBuilder(agent_name=name, task_path="audit/no-task", variant_index=0)
    deployer.parse_artifacts(
        work_dir=tmp_path,
        config=deployer.config,
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )
    return builder.finalize(reward=None)


@pytest.mark.parametrize("name,stem", HARNESSES)
def test_public_package_and_factory_contract(name, stem):
    package = importlib.import_module(f"ale_run.agents.{name}")
    deployer_cls, config_cls = resolve_agent(
        AgentSpec(
            id=name,
            class_=f"ale_run.agents.{name}.deployer.{stem}Deployer",
        )
    )
    assert deployer_cls is getattr(package, f"{stem}Deployer")
    assert config_cls is getattr(package, f"{stem}Config")
    assert dataclasses.is_dataclass(build_config(config_cls, {"model": "audit-model"}))
    assert deployer_cls.supported_executors == frozenset({"sandbox"})


@pytest.mark.parametrize("name,stem", HARNESSES)
def test_api_key_is_not_in_config_repr(name, stem, tmp_path):
    _, deployer = adapter(name, stem, tmp_path, api_key="audit-private-key")
    assert "audit-private-key" not in repr(deployer.config)


def test_qwen_cumulative_usage_is_not_added_twice(tmp_path):
    trajectory = parse(
        "qwen_code",
        "QwenCode",
        tmp_path,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                },
            },
        ],
    )
    assert trajectory.final_metrics.total_input_tokens == 10
    assert trajectory.final_metrics.total_output_tokens == 3


def test_qwen_summary_only_preserves_response_and_zero_cost(tmp_path):
    trajectory = parse(
        "qwen_code",
        "QwenCode",
        tmp_path,
        [
            {
                "type": "result",
                "result": "answer",
                "total_cost_usd": 0.0,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        ],
    )
    assert any(step.message == "answer" for step in trajectory.steps)
    assert trajectory.final_metrics.total_output_tokens == 0


def test_qwen_tool_images_are_typed(tmp_path):
    trajectory = parse(
        "qwen_code",
        "QwenCode",
        tmp_path,
        [
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aW1hZ2U=",
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
        ],
    )
    result = trajectory.steps[0].observation.results[0]
    assert result.content[0].type == "image"
    assert result.content[0].image.data == "aW1hZ2U="


def test_pi_native_tool_result_images_and_no_duplicate_message(tmp_path):
    content = [{"type": "image", "mimeType": "image/png", "data": "aW1hZ2U="}]
    trajectory = parse(
        "pi_cli",
        "PiCli",
        tmp_path,
        [
            {"type": "tool_execution_end", "toolCallId": "tool1", "result": {"content": content}},
            {"type": "message_end", "message": {"role": "toolResult", "content": content}},
        ],
    )
    assert len(trajectory.steps) == 1
    image = trajectory.steps[0].observation.results[0].content[0]
    assert image.type == "image"
    assert image.image.data == "aW1hZ2U="


def test_pi_thinking_is_not_lost(tmp_path):
    trajectory = parse(
        "pi_cli",
        "PiCli",
        tmp_path,
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reason"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            },
        ],
    )
    assert any(step.reasoning == "reason" for step in trajectory.steps)


def test_zcode_summary_only_camel_case_usage_and_zero_values(tmp_path):
    trajectory = parse(
        "zcode",
        "ZCode",
        tmp_path,
        [
            {
                "type": "result",
                "response": "answer",
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 0,
                    "cacheReadTokens": 2,
                    "cacheWriteTokens": 0,
                },
            },
        ],
    )
    assert any(step.message == "answer" for step in trajectory.steps)
    assert trajectory.final_metrics.total_input_tokens == 10
    assert trajectory.final_metrics.total_output_tokens == 0
    assert trajectory.final_metrics.total_cache_read_tokens == 2


def test_zcode_summary_overrides_request_metrics_without_duplicate_answer(tmp_path):
    trajectory = parse(
        "zcode",
        "ZCode",
        tmp_path,
        [
            {"type": "model.streaming", "payload": {"kind": "text_delta", "delta": "answer"}},
            {"type": "model.streaming", "payload": {"kind": "finish"}},
            {
                "type": "session.updated",
                "payload": {
                    "type": "model_request_completed",
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 3,
                    },
                },
            },
            {
                "type": "result",
                "response": "answer",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                },
            },
        ],
    )
    assert trajectory.final_metrics.total_input_tokens == 10
    assert sum(step.message == "answer" for step in trajectory.steps) == 1


def test_zcode_partial_stream_survives_timeout(tmp_path):
    trajectory = parse(
        "zcode",
        "ZCode",
        tmp_path,
        [
            {
                "type": "model.streaming",
                "payload": {
                    "kind": "text_delta",
                    "assistantMessageId": "a1",
                    "delta": "unfinished",
                },
            },
        ],
    )
    assert any(step.message == "unfinished" for step in trajectory.steps)


@pytest.mark.parametrize("name,stem", HARNESSES)
def test_known_secret_is_redacted_from_parsed_artifacts(name, stem, tmp_path):
    token = "audit-private-key"
    event = {"type": "unexpected", "error": token}
    if name == "pi_cli":
        event = {"type": "message_end", "message": {"role": "assistant", "content": token}}
    trajectory = parse(name, stem, tmp_path, [event], api_key=token)
    assert token not in trajectory.model_dump_json()


def process_alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("name,stem", HARNESSES)
async def test_cancellation_kills_real_tool_process_tree(name, stem, tmp_path, monkeypatch):
    module, deployer = adapter(name, stem, tmp_path)
    fixture = Path(__file__).parent / "fixtures/process_tree.py"
    command = [sys.executable, str(fixture)]
    child_file = tmp_path / "child.pid"
    environment = {**os.environ, "AUDIT_CHILD_PID": str(child_file)}
    monkeypatch.setattr(deployer, "_build_env", lambda config: environment)
    monkeypatch.setattr(module, "_TERM_GRACE_S", 0.1)
    if name == "pi_cli":
        monkeypatch.setattr(deployer, "_build_argv", lambda *args: command)
        pid_name = "pi.pid"
    elif name == "qwen_code":
        monkeypatch.setattr(deployer, "_build_argv", lambda *args: command)
        pid_name = "qwen_code.pid"
    else:
        deployer._node_path = sys.executable
        deployer._cli_path = fixture
        pid_name = "zcode.pid"
    task = asyncio.create_task(deployer.launch("audit"))
    child_pid = parent_pid = None
    try:
        for attempt in range(200):
            if child_file.exists():
                break
            await asyncio.sleep(0.01)
        assert child_file.exists(), "fixture child was not started"
        child_pid = int(child_file.read_text())
        parent_pid = int((tmp_path / pid_name).read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        for attempt in range(100):
            if not process_alive(child_pid):
                break
            await asyncio.sleep(0.01)
        assert not process_alive(parent_pid), "CLI parent not reaped"
        assert not process_alive(child_pid), "tool subprocess survived cancellation"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for pid in (parent_pid, child_pid):
            if pid and process_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.asyncio
async def test_zcode_cached_ref_and_sha_are_resolved(tmp_path, monkeypatch):
    module, deployer = adapter("zcode", "ZCode", tmp_path)
    repo_dir = tmp_path / "build"
    (repo_dir / ".git").mkdir(parents=True)
    commands = []

    async def command(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "abc123\n", "")

    if hasattr(module, "run_command"):
        monkeypatch.setattr(module, "run_command", command)
    await deployer._clone_or_update(repo_dir, "1" * 40)
    assert any("fetch" in argv and "1" * 40 in argv for argv in commands)
    assert deployer.version == "src:abc123"


@pytest.mark.asyncio
async def test_qwen_wrong_installed_version_is_replaced(tmp_path, monkeypatch):
    module, deployer = adapter("qwen_code", "QwenCode", tmp_path, api_key="audit-key")
    bootstrap = importlib.import_module("ale_run.agents._bootstrap")
    monkeypatch.setattr(bootstrap, "ensure_node_npm", AsyncMock(return_value=("node", "npm")))
    monkeypatch.setattr(bootstrap, "ensure_cua_mcp_server", AsyncMock())
    monkeypatch.setattr(module, "shutil", SimpleNamespace(which=lambda name: "/mock/qwen"))
    installed = AsyncMock()
    monkeypatch.setattr(deployer, "_install_cli", installed)
    if hasattr(module, "ensure_cli_node"):
        monkeypatch.setattr(module, "ensure_cli_node", AsyncMock(return_value="node"))

        async def command(argv, **kwargs):
            version = "0.1.0" if installed.await_count == 0 else deployer.config.cli_version
            return subprocess.CompletedProcess(argv, 0, version, "")

        monkeypatch.setattr(module, "run_command", command)
    else:
        monkeypatch.setattr(
            module,
            "subprocess",
            SimpleNamespace(
                run=lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "0.1.0", ""),
                TimeoutExpired=subprocess.TimeoutExpired,
            ),
        )
    await deployer.install()
    installed.assert_awaited_once_with("npm", deployer.config.cli_version)


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [["text", "image"], ["text"]])
async def test_pi_custom_model_preserves_input_capabilities(tmp_path, monkeypatch, modalities):
    module, deployer = adapter(
        "pi_cli",
        "PiCli",
        tmp_path,
        provider="custom",
        api_key="audit-key",
        base_url="https://example.invalid/v1",
        input_modalities=modalities,
    )
    bootstrap = importlib.import_module("ale_run.agents._bootstrap")
    monkeypatch.setattr(bootstrap, "ensure_node_npm", AsyncMock(return_value=("node", "npm")))
    monkeypatch.setattr(bootstrap, "ensure_cua_mcp_server", AsyncMock())
    monkeypatch.setattr(module, "ensure_cli_node", AsyncMock(return_value="node"))
    monkeypatch.setattr(
        module, "_installed_version", AsyncMock(return_value=deployer.config.cli_version)
    )
    monkeypatch.setattr(module, "shutil", SimpleNamespace(which=lambda name: "/mock/pi"))
    await deployer.install()
    models_file = Path(deployer._home_dir) / ".pi/agent/models.json"
    provider = json.loads(models_file.read_text())["providers"]["custom"]
    assert provider["models"][0]["input"] == modalities
    assert provider["apiKey"] == "${ALE_PI_API_KEY}"
    assert "audit-key" not in models_file.read_text()


@pytest.mark.parametrize("version", ["0.1.0", None])
@pytest.mark.parametrize("replaced", [False, True])
async def test_pi_checks_cli_after_install(tmp_path, monkeypatch, version, replaced):
    module, deployer = adapter("pi_cli", "PiCli", tmp_path, api_key="audit-key")
    bootstrap = importlib.import_module("ale_run.agents._bootstrap")
    monkeypatch.setattr(bootstrap, "ensure_node_npm", AsyncMock(return_value=("node", "npm")))
    monkeypatch.setattr(bootstrap, "ensure_cua_mcp_server", AsyncMock())
    monkeypatch.setattr(module, "ensure_cli_node", AsyncMock(return_value="node"))
    refreshed_version = deployer.config.cli_version if replaced else version
    probe = AsyncMock(side_effect=[version, refreshed_version])
    monkeypatch.setattr(module, "_installed_version", probe)
    monkeypatch.setattr(module, "shutil", SimpleNamespace(which=lambda name: "/mock/pi"))
    command = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(module, "run_command", command)
    if replaced:
        await deployer.install()
        assert deployer._pi_path == "/mock/pi"
    else:
        with pytest.raises(RuntimeError, match="installed CLI does not match cli_version"):
            await deployer.install()
    assert probe.await_count == 2
    assert any(
        "install" in call.args[0] and "-g" in call.args[0] for call in command.await_args_list
    )


@pytest.mark.parametrize("failure", ["exit", "timeout"])
async def test_pi_version_probe_failure_is_not_an_installed_version(monkeypatch, failure):
    module = importlib.import_module("ale_run.agents.pi_cli.deployer")
    command = AsyncMock(
        return_value=subprocess.CompletedProcess([], 1, "0.85.1", "failed to load CLI")
    )
    if failure == "timeout":
        command.side_effect = TimeoutError("CLI command timed out")
    monkeypatch.setattr(module, "run_command", command)
    assert await module._installed_version("/mock/pi") is None


@pytest.fixture
def zcode_builtin():
    return {
        "schemaVersion": 1,
        "revision": 42,
        "config": {
            "providerConfigRules": {"templateRules": [], "providerRules": []},
            "modelConfigRules": {
                "modelRules": [
                    {
                        "modelMatch": ".*",
                        "config": {"optionSpecs": {"maxOutputTokens": {"max": 9137}}},
                    },
                ],
                "modelApiRules": [
                    {
                        "modelMatch": ".*",
                        "apiTypeMatch": "openai-chat-completions",
                        "config": {
                            "optionSpecs": {
                                "maxOutputTokens": {"map": "{'max_tokens': maxOutputTokens}"},
                            },
                        },
                    },
                ],
                "providerSiteRules": [],
                "templateModelRules": [],
                "builtinProviderModelRules": [],
            },
        },
    }


def test_zcode_builtin_defaults_preserve_upstream_output_limit(zcode_builtin):
    from ale_run.agents.zcode import ZCodeConfig, ZCodeDeployer

    config = ZCodeConfig()
    assert config.max_output_tokens is None
    original = json.dumps(zcode_builtin)
    builtin = ZCodeDeployer._build_builtin_config(config, zcode_builtin)
    rules = builtin["config"]["modelConfigRules"]
    override = rules["builtinProviderModelRules"].pop()
    assert override["providerId"] == config.provider_id
    assert override["modelId"] == config.model
    assert "maxOutputTokens" not in json.dumps(override)
    assert "max_completion_tokens" not in json.dumps(builtin)
    assert builtin == zcode_builtin
    assert json.dumps(zcode_builtin) == original


@pytest.mark.parametrize("overrides", [{}, {"max_output_tokens": None}])
@pytest.mark.parametrize("context_window", [64000, 1048576])
def test_zcode_unset_output_limit_preserves_context_and_reasoning(
    overrides, context_window, zcode_builtin
):
    from ale_run.agents.zcode import ZCodeConfig, ZCodeDeployer

    config = build_config(
        ZCodeConfig,
        {
            "context_window": context_window,
            "reasoning_request": {"reasoning_effort": "low"},
            **overrides,
        },
    )
    builtin = ZCodeDeployer._build_builtin_config(config, zcode_builtin)
    personal = ZCodeDeployer._build_personal_config(config)
    rules = builtin["config"]["modelConfigRules"]
    override = rules["builtinProviderModelRules"][-1]
    serialized = json.dumps([override, personal])
    assert "maxOutputTokens" not in serialized
    assert "max_completion_tokens" not in serialized
    model = override["config"]
    assert model["properties"]["contextWindow"] == context_window
    assert model["optionSpecs"]["reasoningLevel"]["values"] == ["disabled", "enabled"]
    options = model["optionSpecs"]
    assert json.loads(options["reasoningLevel"]["map"]) == {"reasoning_effort": "low"}


@pytest.mark.parametrize("max_output_tokens", [128, 131072])
def test_zcode_explicit_model_limits_and_reasoning_request(
    tmp_path, max_output_tokens, zcode_builtin
):
    _, deployer = adapter(
        "zcode",
        "ZCode",
        tmp_path,
        context_window=64000,
        max_output_tokens=max_output_tokens,
        reasoning_request={"reasoning_effort": "low"},
    )
    rules = deployer._build_builtin_config(deployer.config, zcode_builtin)["config"][
        "modelConfigRules"
    ]
    model = rules["builtinProviderModelRules"][-1]["config"]
    assert model["properties"]["contextWindow"] == 64000
    assert model["optionSpecs"]["maxOutputTokens"]["max"] == max_output_tokens
    options = model["optionSpecs"]
    assert options["maxOutputTokens"] == {
        "max": max_output_tokens,
        "map": "{'max_completion_tokens': maxOutputTokens}",
    }
    assert json.loads(options["reasoningLevel"]["map"]) == {"reasoning_effort": "low"}
    assert "thinking" not in options["reasoningLevel"]["map"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "failure", "timeout"])
async def test_wait_process_background_ownership(mode, tmp_path):
    from ale_run.agents._cli_runtime import wait_process

    fixture = Path(__file__).parent / "fixtures/process_tree.py"
    child_file = tmp_path / "child.pid"
    source = (
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "subprocess.Popen([sys.executable,sys.argv[1],'--child'])\n"
        "while not Path(os.environ['AUDIT_CHILD_PID']).exists(): time.sleep(0.01)\n"
        + ("time.sleep(60)\n" if mode == "timeout" else f"sys.exit({int(mode == 'failure')})\n")
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", source, str(fixture)],
        env={**os.environ, "AUDIT_CHILD_PID": str(child_file)},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        if mode == "timeout":
            with pytest.raises(TimeoutError):
                await wait_process(proc, timeout=0.3, grace_s=0.05)
        else:
            assert await wait_process(proc, timeout=2, grace_s=0.05) == int(mode == "failure")
        assert child_file.exists()
        child_pid = int(child_file.read_text())
        if mode != "success":
            for attempt in range(100):
                if not process_alive(child_pid):
                    break
                await asyncio.sleep(0.01)
        assert process_alive(child_pid) is (mode == "success")
        assert proc.returncode is not None
    finally:
        for pid in (proc.pid, child_pid):
            if pid and process_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_cache", [False, True])
async def test_old_node_installs_requested_private_runtime(tmp_path, monkeypatch, stale_cache):
    from ale_run.agents import _cli_runtime as runtime

    target = tmp_path / ".ale-node/24.14.0/node_modules/.bin/node"
    if stale_cache:
        target.parent.mkdir(parents=True)
        target.touch()
    monkeypatch.setattr(runtime.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", "/system/bin")
    commands = []
    installed = False

    async def command(argv, **kwargs):
        nonlocal installed
        commands.append(argv)
        if argv[0] == "/system/npm":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
            installed = True
            output = ""
        else:
            output = "v24.14.0\n" if installed else "v24.12.0\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(runtime, "run_command", command)
    chosen = await runtime.ensure_cli_node("/system/node", "/system/npm", (24, 14, 0))
    assert chosen == str(target)
    assert any(argv[0] == "/system/npm" and "node@24.14.0" in argv for argv in commands)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(target.parent)


@pytest.mark.parametrize("name", ["qwen_code_hf", "zcode_hf"])
def test_hf_presets_use_separate_inference_credential(name):
    import yaml

    root = Path(__file__).resolve().parents[1]
    preset = root / "candidate/configs/agents" / f"{name}.yaml"
    if not preset.exists():
        preset = Path(__file__).resolve().parents[2] / "configs/agents" / f"{name}.yaml"
    config = yaml.safe_load(preset.read_text())
    assert config["config"]["api_key"] == "${env:HF_INFERENCE_TOKEN}"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [True, False])
async def test_run_command_captures_output_without_thread_shutdown(text):
    from ale_run.agents._cli_runtime import run_command

    result = await run_command(
        [sys.executable, "-c", "import sys; print('ok'); print('err', file=sys.stderr)"],
        text=text,
        timeout=2,
    )
    assert result.returncode == 0
    assert result.stdout == ("ok\n" if text else b"ok\n")
    assert result.stderr == ("err\n" if text else b"err\n")


@pytest.mark.parametrize("name,stem", HARNESSES)
def test_selected_node_takes_priority_over_executor_path(name, stem, tmp_path):
    _, deployer = adapter(name, stem, tmp_path, api_key="audit-key")
    deployer.executor.env = {"PATH": "/old-runtime/bin"}
    deployer._node_path = "/private-runtime/bin/node"
    deployer._provider_env = {}
    assert (
        deployer._build_env(deployer.config)["PATH"].split(os.pathsep)[0] == "/private-runtime/bin"
    )


@pytest.mark.parametrize("recovered", [False, True])
async def test_pi_reports_stream_failure_even_when_cli_exits_zero(tmp_path, monkeypatch, recovered):
    _, deployer = adapter("pi_cli", "PiCli", tmp_path, api_key="audit-private-key")
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "401 audit-private-key",
            },
        }
    ]
    if recovered:
        events.append(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "stopReason": "stop",
                },
            }
        )
    source = "print(" + repr("\n".join(json.dumps(event) for event in events)) + ")"
    monkeypatch.setattr(deployer, "_build_argv", lambda *args: [sys.executable, "-c", source])
    result = await deployer.launch("smoke")
    assert result.exit_code == 0
    assert result.status == ("completed" if recovered else "failed")
    if not recovered:
        assert "401" in result.error
        assert "audit-private-key" not in result.error
