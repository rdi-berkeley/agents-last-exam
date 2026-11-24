"""Helper calls preserve usage without retaining prompts or credentials."""

import asyncio
import json
from types import SimpleNamespace

from ale_claw.harness.model.usage import (
    record_helper_usage,
    start_helper_usage,
    stop_helper_usage,
)


def test_helper_usage_propagates_to_thread_and_prefers_provider_cost(tmp_path):
    path = tmp_path / "usage.jsonl"
    response = SimpleNamespace(
        usage={
            "prompt_tokens": 40,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 9},
            "cost": 0.25,
        },
        _hidden_params={"response_cost": 0.5},
    )

    async def exercise():
        token = start_helper_usage(path)
        try:
            await asyncio.to_thread(record_helper_usage, response, model="model", purpose="vision")
        finally:
            stop_helper_usage(token)
        record_helper_usage(response, model="model", purpose="ignored")

    asyncio.run(exercise())
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["input_tokens"] == 40
    assert entries[0]["cache_read_input_tokens"] == 9
    assert entries[0]["cost_usd"] == 0.25
    assert set(entries[0]) == {
        "timestamp",
        "purpose",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "cost_usd",
    }


def test_helper_call_with_no_usage_is_recorded_as_unknown(tmp_path):
    path = tmp_path / "usage.jsonl"
    token = start_helper_usage(path)
    try:
        record_helper_usage(SimpleNamespace(usage=None), model="model", purpose="vision")
    finally:
        stop_helper_usage(token)
    entry = json.loads(path.read_text())
    assert entry["input_tokens"] is None
    assert entry["output_tokens"] is None
    assert entry["cost_usd"] is None
