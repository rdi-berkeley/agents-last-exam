"""Regression tests for a reviewer-caught bug: ATIF's prompt_tokens is the
FULL input count (cache-inclusive), not uncached-only. Several deployers,
written against ALE's old (pre-ATIF-rename) convention where input_tokens
meant "uncached", were subtracting cached_tokens back out when their native
harness usage was already cache-inclusive -- silently under-reporting.
"""
from __future__ import annotations

import json

from ale_run.agents.forgecode.deployer import ForgecodeDeployer
from ale_run.agents.gemini_cli.deployer import GeminiCliDeployer
from ale_run.agents.terminus_2.deployer import Terminus2Deployer
from ale_run.base_interface.trajectory import TrajectoryBuilder


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent_name="x", model="m", task_path="demo/hello", variant_index=0
    )


def test_terminus_2_prompt_tokens_is_not_subtracted():
    # terminus_2's own trajectory.json is already real ATIF: prompt_tokens=100
    # already includes the 80 cached tokens.
    metrics = Terminus2Deployer._step_metrics(
        {"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 10}
    )
    assert metrics.prompt_tokens == 100
    assert metrics.cached_tokens == 80


def test_gemini_cli_prompt_tokens_is_not_subtracted():
    # Gemini's own usage metadata (promptTokenCount) is already cache-inclusive.
    metrics = GeminiCliDeployer._stats_to_metrics(
        {"input_tokens": 100, "cached": 80, "output_tokens": 10}
    )
    assert metrics.prompt_tokens == 100
    assert metrics.cached_tokens == 80


def test_forgecode_prompt_tokens_is_not_subtracted(tmp_path):
    dump = {
        "conversation": {
            "context": {
                "messages": [
                    {
                        "text": {"role": "assistant", "content": "hi"},
                        "usage": {
                            "prompt_tokens": 100,
                            "cached_tokens": 80,
                            "completion_tokens": 10,
                        },
                    }
                ]
            }
        }
    }
    dump_file = tmp_path / "dump.json"
    dump_file.write_text(json.dumps(dump), encoding="utf-8")

    builder = _builder()
    ForgecodeDeployer._parse_dump_json(dump_file, builder)

    metrics_steps = [s for s in builder.trajectory.steps if s.metrics is not None]
    assert len(metrics_steps) == 1
    assert metrics_steps[0].metrics.prompt_tokens == 100
    assert metrics_steps[0].metrics.cached_tokens == 80
