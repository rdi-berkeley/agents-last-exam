import ast
import asyncio
import inspect
import json
from unittest.mock import AsyncMock

import pytest

from ale_run.environments.providers import docker
from ale_run.orchestration import lifecycle
from ale_run.orchestration.lifecycle import _append_prompt_suffix


@pytest.mark.parametrize(
    "nano_cpus,memory_bytes,cpu_quota",
    [
        (4_000_000_000, 15 * 1024 ** 3, 4.0),
        (1_500_000_000, 1536 * 1024 ** 2, 1.5),
        (250_000_000, 123456789, 0.25),
        (0, 0, 0.0),
    ],
)
def test_resource_limits_use_numeric_inspect_values(
    monkeypatch, nano_cpus, memory_bytes, cpu_quota,
):
    commands = AsyncMock(
        return_value=(0, json.dumps({"NanoCpus": nano_cpus, "Memory": memory_bytes}), "")
    )
    monkeypatch.setattr(docker, "_run_docker", commands)

    limits = asyncio.run(docker._get_resource_limits("sandbox-name"))

    assert limits == {"cpu_quota": cpu_quota, "memory_limit_bytes": memory_bytes}
    assert isinstance(limits["cpu_quota"], float)
    assert isinstance(limits["memory_limit_bytes"], int)
    commands.assert_awaited_once_with(
        "inspect", "--format", "{{json .HostConfig}}", "sandbox-name",
    )


def test_resource_inspect_failure_is_not_replaced_with_defaults(monkeypatch):
    monkeypatch.setattr(docker, "_run_docker", AsyncMock(return_value=(1, "", "unavailable")))

    with pytest.raises(RuntimeError, match="sandbox-name limits: unavailable"):
        asyncio.run(docker._get_resource_limits("sandbox-name"))


@pytest.mark.parametrize(
    "cpu_quota,memory_bytes,cpu_text,memory_text",
    [
        (4.0, 15 * 1024 ** 3, "4 CPU equivalents", "15 GiB (16106127360 bytes)"),
        (1.5, 1536 * 1024 ** 2, "1.5 CPU equivalents", "1.5 GiB (1610612736 bytes)"),
    ],
)
def test_shared_suffix_preserves_custom_text_and_adds_one_budget(
    cpu_quota, memory_bytes, cpu_text, memory_text,
):
    task_meta = {"description": "Task instructions.\n"}
    metadata = {
        "provider": "docker",
        "resource_limits": {"cpu_quota": cpu_quota, "memory_limit_bytes": memory_bytes},
    }

    _append_prompt_suffix(task_meta, "\nCustom suffix.\n", sandbox_metadata=metadata)

    prompt = task_meta["description"]
    assert prompt.startswith("Task instructions.\n\nCustom suffix.\n\n")
    assert prompt.count("Task sandbox configured per-container ceilings:") == 1
    assert cpu_text in prompt
    assert memory_text in prompt
    assert "not guaranteed free memory or exclusive CPU capacity" in prompt
    assert "/proc, nproc, os.cpu_count(), and free may report host resources" in prompt
    assert metadata["resource_limits"]["memory_limit_bytes"] == memory_bytes


def test_budget_is_added_without_custom_suffix():
    task_meta = {"description": "Task instructions."}
    _append_prompt_suffix(
        task_meta,
        "",
        sandbox_metadata={
            "provider": "docker",
            "resource_limits": {"cpu_quota": 4.0, "memory_limit_bytes": 0},
        },
    )

    assert "CPU quota 4 CPU equivalents" in task_meta["description"]
    assert "memory limit" not in task_meta["description"]


@pytest.mark.parametrize("provider", ["qemu", "gcloud", "static", None])
@pytest.mark.parametrize("suffix", ["", " \n", "\nCustom suffix.\n"])
def test_other_providers_keep_existing_suffix_behavior(provider, suffix):
    task_meta = {"description": "Task instructions.\n"}
    _append_prompt_suffix(
        task_meta,
        suffix,
        sandbox_metadata={
            "provider": provider,
            "resource_limits": {"cpu_quota": 4.0, "memory_limit_bytes": 15 * 1024 ** 3},
        },
    )

    expected = "Task instructions.\n\nCustom suffix." if suffix.strip() else "Task instructions.\n"
    assert task_meta["description"] == expected


@pytest.mark.parametrize("metadata", [None, {}, {"provider": "docker"}, {
    "provider": "docker", "resource_limits": {"cpu_quota": 0, "memory_limit_bytes": 0},
}])
def test_missing_or_unlimited_values_do_not_claim_capacity(metadata):
    task_meta = {"description": "Task instructions."}
    _append_prompt_suffix(task_meta, "Custom suffix.", sandbox_metadata=metadata)

    assert task_meta["description"] == "Task instructions.\n\nCustom suffix."


def test_per_run_metadata_does_not_leak_between_prompts():
    first = {"description": "First task."}
    second = {"description": "Second task."}
    _append_prompt_suffix(
        first, "", sandbox_metadata={
            "provider": "docker", "resource_limits": {"cpu_quota": 1.5},
        },
    )
    _append_prompt_suffix(
        second, "", sandbox_metadata={
            "provider": "docker", "resource_limits": {"cpu_quota": 4.0},
        },
    )

    assert "1.5 CPU equivalents" in first["description"]
    assert "4 CPU equivalents" in second["description"]
    assert "1.5 CPU equivalents" not in second["description"]


def test_lifecycle_uses_one_shared_prompt_for_recording_and_launch():
    tree = ast.parse(inspect.getsource(lifecycle.run_one_unit))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    suffix_calls = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "_append_prompt_suffix"
    ]
    assert len(suffix_calls) == 1
    suffix_call = suffix_calls[0]
    assert any(
        keyword.arg == "sandbox_metadata"
        and ast.unparse(keyword.value) == "env.sandbox.metadata"
        for keyword in suffix_call.keywords
    )
    assert not any(
        isinstance(node, (ast.If, ast.Match)) and suffix_call in ast.walk(node)
        for node in ast.walk(tree)
    )
    builder_call = next(
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "TrajectoryBuilder"
    )
    launch_call = next(
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "run_deployer"
    )
    for call, field in [(builder_call, "instruction"), (launch_call, "prompt")]:
        assert suffix_call.lineno < call.lineno
        assert any(
            keyword.arg == field and ast.unparse(keyword.value) == "task_meta['description']"
            for keyword in call.keywords
        )
