"""Offline smoke: CAR run-task transcript -> ATIF Steps.

No VM, no `car` binary, no API key. Feeds a synthetic transcript through the
translator and asserts the resulting ATIF trajectory shape. Run:

    uv run python tests/smoke_car_transcript.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ale_run.agents.car.transcript_to_trajectory import parse_transcript_into
from ale_run.base_interface import TrajectoryBuilder

TRANSCRIPT = [
    {"type": "run_start", "goal": "list /tmp then write done.txt",
     "model": "anthropic/claude-sonnet-4.6", "max_turns": 100,
     "servers": ["vm", "cua"], "tool_count": 9},
    {"type": "agent_turn", "turn": 1, "text": "Listing the directory.",
     "tool_calls": [{"id": "call_0_0", "name": "mcp_vm_exec",
                     "arguments": {"command": "ls /tmp"}}],
     "usage": {"input_tokens": 1200, "output_tokens": 40}},
    {"type": "observation", "turn": 1,
     "results": [{"tool_call_id": "call_0_0", "content": "a.txt\nb.txt", "is_error": False}]},
    {"type": "agent_turn", "turn": 2, "text": "Writing the marker file.",
     "tool_calls": [{"id": "call_1_0", "name": "mcp_vm_write",
                     "arguments": {"path": "/tmp/done.txt", "content": "ok"}}],
     "usage": {"input_tokens": 1300, "output_tokens": 30}},
    {"type": "observation", "turn": 2,
     "results": [{"tool_call_id": "call_1_0", "content": "[FAILED] permission denied", "is_error": True}]},
    {"type": "agent_turn", "turn": 3, "text": "Done.", "tool_calls": [],
     "usage": {"input_tokens": 1400, "output_tokens": 5}},
    {"type": "run_end", "status": "completed", "turns": 3, "answer": "Done.", "error": None},
]


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        work_dir = Path(td)
        tpath = work_dir / "car" / "transcript.jsonl"
        tpath.parent.mkdir(parents=True)
        tpath.write_text("\n".join(json.dumps(r) for r in TRANSCRIPT), encoding="utf-8")

        builder = TrajectoryBuilder(
            agent_name="car", model="anthropic/claude-sonnet-4.6",
            task_path="demo/helloworld", variant_index=0,
            instruction="list /tmp then write done.txt",
        )
        parse_transcript_into(work_dir, builder)
        traj = builder.finalize(reward=1.0, status="completed")

        steps = traj.steps
        sources = [s.source for s in steps]
        assert sources == ["agent", "environment", "agent", "environment", "agent"], sources

        # First agent step: one tool call, token metrics captured.
        assert steps[0].tool_calls[0].name == "mcp_vm_exec"
        assert steps[0].tool_calls[0].arguments == {"command": "ls /tmp"}
        assert steps[0].metrics.input_tokens == 1200

        # Observations correlate by tool_call_id and carry the error flag.
        obs1 = steps[1].observation.results[0]
        assert obs1.tool_call_id == "call_0_0" and obs1.is_error is False
        assert obs1.content[0].text == "a.txt\nb.txt"
        obs2 = steps[3].observation.results[0]
        assert obs2.is_error is True

        # Final agent step is the done signal: text, no tool calls.
        assert steps[4].message == "Done." and steps[4].tool_calls == []

        # Token totals summed across agent turns; run metadata on extra.
        assert traj.final_metrics.total_input_tokens == 3900
        assert traj.extra["car"]["run_end"]["status"] == "completed"
        assert traj.extra["car"]["run_start"]["servers"] == ["vm", "cua"]

    print("smoke_car_transcript: OK")


if __name__ == "__main__":
    main()
