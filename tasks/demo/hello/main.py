"""Demo task: ``demo/hello`` — minimal but exercises the full surface.

Tests:
  - input/ files agent must read (visible during solve)
  - reference/ file invisible during setup, visible during eval
  - software/ helper script the agent MAY use (or not)
  - output/ dir the agent writes to
  - 3 variants: tests variant params injection + load() iteration
  - partial scoring: exact match → 1.0, partial → 0.0..0.5, missing → 0.0

Agent's fastest solving path (claude-code typical):
  - read input/note_request.txt  (or input/spec.json for json_kv variant)
  - bash software/write_answer.sh  OR  echo ... > output/answer.txt
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)


VARIANTS: list[tuple[str, dict]] = [
    ("simple",    {"expected": "hello world\n"}),
    ("json_kv",   {"expected_kv": {"greeting": "hello", "target": "world"}}),
    ("multiline", {"expected_lines": [
        "line 1: hello",
        "line 2: world",
        "line 3: done",
    ]}),
]


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "hello"
    VARIANT_NAME: str = "simple"
    # variant-specific payload, injected via load() below
    expected_payload: dict | None = None

    @property
    def answer_path(self) -> str:
        return f"{self.remote_output_dir}/answer.txt"

    @property
    def input_request_path(self) -> str:
        return f"{self.input_dir}/note_request.txt"

    @property
    def input_spec_path(self) -> str:
        return f"{self.input_dir}/spec.json"

    @property
    def reference_path(self) -> str:
        return f"{self.reference_dir}/expected.txt"

    @property
    def software_script_path(self) -> str:
        return f"{self.software_dir}/write_answer.sh"

    @property
    def task_description(self) -> str:
        return (
            f"Goal: write a specific string to {self.answer_path}.\n\n"
            f"Steps:\n"
            f"1. Read {self.input_request_path} for the exact content required "
            f"for variant '{self.VARIANT_NAME}'.\n"
            f"2. (Optional) Read {self.input_spec_path} for any variant params.\n"
            f"3. Write the exact required content to {self.answer_path}. "
            f"You may use the helper {self.software_script_path} or write the "
            f"file yourself with any shell command (e.g. printf > file).\n\n"
            f"Verification: file content must match the reference exactly. "
            f"Partial credit for partial matches. Do not write anything else."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "answer_path": self.answer_path,
            "input_request_path": self.input_request_path,
            "input_spec_path": self.input_spec_path,
            "reference_path": self.reference_path,
            "software_script_path": self.software_script_path,
            "expected_payload": self.expected_payload,
        })
        return m


def _expected_text(variant: str, payload: dict) -> str:
    """The reference content for each variant. This is what evaluate compares against."""
    if variant == "simple":
        return payload["expected"]
    if variant == "json_kv":
        kv = payload["expected_kv"]
        return json.dumps(kv, separators=(",", ":")) + "\n"
    if variant == "multiline":
        return "\n".join(payload["expected_lines"]) + "\n"
    raise ValueError(f"unknown variant: {variant}")


def _request_text(variant: str, payload: dict, output_path: str) -> str:
    """The agent-visible instructions in input/note_request.txt for this variant."""
    if variant == "simple":
        return (
            f"Variant: simple\n"
            f"Write exactly this text to {output_path}:\n"
            f"---\n{payload['expected']}---\n"
        )
    if variant == "json_kv":
        kv = payload["expected_kv"]
        return (
            f"Variant: json_kv\n"
            f"Write a single-line JSON object to {output_path} followed by a newline.\n"
            f"Keys and values (order matters):\n"
            + "\n".join(f"  {k!r}: {v!r}" for k, v in kv.items())
            + f"\nUse compact JSON (no spaces). See {payload.get('_spec_hint','spec.json')}\n"
        )
    if variant == "multiline":
        lines = payload["expected_lines"]
        return (
            f"Variant: multiline\n"
            f"Write these 3 lines to {output_path} (Unix newlines, trailing newline):\n"
            + "\n".join(f"  {l}" for l in lines)
            + "\n"
        )
    raise ValueError(variant)


@cb.tasks_config(split="train")
def load():
    """Build one cb.Task per variant. Each gets its own metadata payload."""
    out = []
    for variant_name, payload in VARIANTS:
        cfg = TaskConfig(VARIANT_NAME=variant_name, expected_payload=payload)
        out.append(cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": cfg.OS_TYPE},
            },
        ))
    return out


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Stage input/ + software/ on VM; verify reference/ NOT visible yet."""
    meta = task_cfg.metadata
    variant = meta["variant_name"]
    payload = meta["expected_payload"]

    # Create directories.
    for d in (meta["input_dir"], meta["software_dir"], meta["remote_output_dir"]):
        await session.run_command(f"mkdir -p {d!r}", check=False)
    # Reset output between runs.
    await session.run_command(f"rm -f {meta['answer_path']!r}", check=False)
    # Reset reference too — on persistent dev VMs the previous run's evaluate
    # leaves it staged; we want the visibility check below to be meaningful.
    # In production (cua-house ephemeral VMs) this is a no-op.
    await session.run_command(f"rm -rf {meta['reference_dir']!r}", check=False)

    # Stage input: request + spec.
    await session.write_file(
        meta["input_request_path"],
        _request_text(variant, payload, meta["answer_path"]),
    )
    await session.write_file(
        meta["input_spec_path"],
        json.dumps({"variant": variant, "payload": payload}, indent=2),
    )

    # Stage a helper script the agent MAY use.
    expected = _expected_text(variant, payload)
    # The helper writes the expected text directly — for testing convenience.
    # Real agents would compose the content themselves; this is a shortcut.
    helper = (
        "#!/bin/bash\n"
        "set -eu\n"
        f"mkdir -p {meta['remote_output_dir']!r}\n"
        f"cat > {meta['answer_path']!r} <<'__ALE_EOF__'\n"
        f"{expected}"
        f"__ALE_EOF__\n"
    )
    await session.write_file(meta["software_script_path"], helper)
    await session.run_command(f"chmod +x {meta['software_script_path']!r}", check=False)

    # Visibility check: reference MUST NOT be readable yet (eval staging unlocks).
    try:
        await session.read_file(meta["reference_path"])
    except Exception:
        logger.info("[%s] reference correctly hidden during setup", variant)
    else:
        raise RuntimeError(
            f"reference leaked during setup: {meta['reference_path']} was readable"
        )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score: exact 1.0; partial 0.0–0.5; missing 0.0."""
    meta = task_cfg.metadata
    variant = meta["variant_name"]
    payload = meta["expected_payload"]
    out_path = meta["answer_path"]
    ref_path = meta["reference_path"]

    expected = _expected_text(variant, payload)

    # Write the reference at eval time (simulates cua-house unlocking).
    await session.run_command(f"mkdir -p {meta['reference_dir']!r}", check=False)
    await session.write_file(ref_path, expected)

    try:
        actual = await session.read_file(out_path)
    except Exception as exc:
        logger.info("[%s] output unreadable: %s", variant, exc)
        return [0.0]

    actual_norm = actual.replace("\r\n", "\n")
    expected_norm = expected.replace("\r\n", "\n")
    if actual_norm == expected_norm:
        return [1.0]

    # Partial: how many expected lines are present in output?
    expected_lines = [l for l in expected_norm.split("\n") if l.strip()]
    if not expected_lines:
        return [0.0]
    hits = sum(1 for l in expected_lines if l in actual_norm)
    partial = hits / len(expected_lines)
    return [round(partial * 0.5, 3)]
