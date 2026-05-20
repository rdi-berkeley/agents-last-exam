"""Demo task: ``demo/hello_win`` — Windows counterpart of ``demo/hello``.

Tests:
  - input/ files agent must read (visible during solve)
  - reference/ file invisible during setup, visible during eval
  - software/ helper script the agent MAY use (or not)
  - output/ dir the agent writes to
  - 3 variants: tests variant params injection + load() iteration
  - partial scoring: exact match → 1.0, partial → 0.0..0.5, missing → 0.0

Agent's fastest solving path (claude-code typical):
  - read input\\note_request.txt  (or input\\spec.json for json_kv variant)
  - cmd /c software\\write_answer.cmd  OR  cmd /c "echo ... > output\\answer.txt"
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig

logger = logging.getLogger(__name__)


VARIANTS: list[tuple[str, dict]] = [
    ("simple",    {"expected": "hello world\r\n"}),
    ("json_kv",   {"expected_kv": {"greeting": "hello", "target": "world"}}),
    ("multiline", {"expected_lines": [
        "line 1: hello",
        "line 2: world",
        "line 3: done",
    ]}),
]


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "hello_win"
    VARIANT_NAME: str = "simple"
    OS_TYPE: str = "windows"
    expected_payload: dict | None = None

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def answer_path(self) -> str:
        return rf"{self.remote_output_dir}\answer.txt"

    @property
    def input_request_path(self) -> str:
        return rf"{self.input_dir}\note_request.txt"

    @property
    def input_spec_path(self) -> str:
        return rf"{self.input_dir}\spec.json"

    @property
    def reference_path(self) -> str:
        return rf"{self.reference_dir}\expected.txt"

    @property
    def software_script_path(self) -> str:
        return rf"{self.software_dir}\write_answer.cmd"

    @property
    def task_description(self) -> str:
        return (
            f"Goal: write a specific string to {self.answer_path}.\n\n"
            f"Steps:\n"
            f"1. Read {self.input_request_path} for the exact content required "
            f"for variant '{self.VARIANT_NAME}'.\n"
            f"2. (Optional) Read {self.input_spec_path} for any variant params.\n"
            f"3. Write the exact required content to {self.answer_path}. "
            f"You may use the helper {self.software_script_path} (run "
            f"`cmd /c {self.software_script_path}`) or write the file yourself "
            f"with any shell command.\n\n"
            f"Verification: file content must match the reference exactly. "
            f"Partial credit for partial matches. Do not write anything else."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "input_dir": self.input_dir,
            "answer_path": self.answer_path,
            "input_request_path": self.input_request_path,
            "input_spec_path": self.input_spec_path,
            "reference_path": self.reference_path,
            "software_script_path": self.software_script_path,
            "expected_payload": self.expected_payload,
        })
        return m


def _expected_text(variant: str, payload: dict) -> str:
    """The reference content for each variant. This is what evaluate compares against.

    All variants use CRLF newlines so the byte-level reference matches what
    ``cmd``-style helpers produce on Windows.
    """
    if variant == "simple":
        return payload["expected"]
    if variant == "json_kv":
        kv = payload["expected_kv"]
        return json.dumps(kv, separators=(",", ":")) + "\r\n"
    if variant == "multiline":
        return "\r\n".join(payload["expected_lines"]) + "\r\n"
    raise ValueError(f"unknown variant: {variant}")


def _request_text(variant: str, payload: dict, output_path: str) -> str:
    """The agent-visible instructions in input\\note_request.txt for this variant."""
    if variant == "simple":
        return (
            f"Variant: simple\r\n"
            f"Write exactly this text to {output_path}:\r\n"
            f"---\r\n{payload['expected']}---\r\n"
        )
    if variant == "json_kv":
        kv = payload["expected_kv"]
        return (
            f"Variant: json_kv\r\n"
            f"Write a single-line JSON object to {output_path} followed by a newline.\r\n"
            f"Keys and values (order matters):\r\n"
            + "\r\n".join(f"  {k!r}: {v!r}" for k, v in kv.items())
            + f"\r\nUse compact JSON (no spaces). See {payload.get('_spec_hint','spec.json')}\r\n"
        )
    if variant == "multiline":
        lines = payload["expected_lines"]
        return (
            f"Variant: multiline\r\n"
            f"Write these 3 lines to {output_path} (Windows CRLF, trailing newline):\r\n"
            + "\r\n".join(f"  {l}" for l in lines)
            + "\r\n"
        )
    raise ValueError(variant)


def _helper_script(expected: str, answer_path: str, output_dir: str) -> str:
    """Build a `.cmd` script that writes the variant's expected text to answer_path.

    Uses a temp file to capture exact bytes (including CRLF) without relying on
    ``echo`` escaping quirks. Encoded as latin-1 → all variant payloads here are
    ASCII so the round-trip is byte-stable.
    """
    payload_b64 = _to_b64(expected)
    # PowerShell decodes base64 → bytes → writes file as raw bytes.
    return (
        "@echo off\r\n"
        "setlocal\r\n"
        f'if not exist "{output_dir}" mkdir "{output_dir}"\r\n'
        f'set "ALE_PAYLOAD={payload_b64}"\r\n'
        'powershell -NoProfile -Command '
        '"[IO.File]::WriteAllBytes('
        f"'{answer_path}', "
        '[Convert]::FromBase64String($env:ALE_PAYLOAD))"\r\n'
        "endlocal\r\n"
    )


def _to_b64(s: str) -> str:
    import base64
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


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
    """Stage input\\ + software\\ on VM; verify reference\\ NOT visible yet."""
    meta = task_cfg.metadata
    variant = meta["variant_name"]
    payload = meta["expected_payload"]

    # Create directories (PowerShell's New-Item -Force is idempotent).
    for d in (meta["input_dir"], meta["software_dir"], meta["remote_output_dir"]):
        await session.run_command(
            f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path \'{d}\' | Out-Null"',
            check=False,
        )
    # Reset output between runs (Remove-Item -Force is no-op if missing).
    await session.run_command(
        f'powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue \'{meta["answer_path"]}\'"',
        check=False,
    )
    # Reset reference too — on persistent dev VMs the previous run's evaluate
    # leaves it staged; we want the visibility check below to be meaningful.
    await session.run_command(
        f'powershell -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue \'{meta["reference_dir"]}\'"',
        check=False,
    )

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
    await session.write_file(
        meta["software_script_path"],
        _helper_script(expected, meta["answer_path"], meta["remote_output_dir"]),
    )

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
    await session.run_command(
        f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path \'{meta["reference_dir"]}\' | Out-Null"',
        check=False,
    )
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

    expected_lines = [l for l in expected_norm.split("\n") if l.strip()]
    if not expected_lines:
        return [0.0]
    hits = sum(1 for l in expected_lines if l in actual_norm)
    partial = hits / len(expected_lines)
    return [round(partial * 0.5, 3)]
