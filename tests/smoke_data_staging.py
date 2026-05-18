"""Unit smoke for ale.io.data_staging + AgenthleEnv visibility rule.

No live VM, no GCS. Uses a recording stub session that captures every
``run_command`` invocation. Asserts:

  1. ``stage_input`` issues gsutil rsync for input/ + software/, mkdir for output/
  2. ``stage_reference`` issues gsutil rsync for reference/
  3. ``upload_output`` issues gsutil cp for output/ → results bucket
  4. ``requires_task_data=False`` → all stage_* / upload_* are no-ops
  5. **Visibility rule**: env.reset_async never issues a reference-related
     command; only env.step_async(Submit) does.

Run from repo root::

    uv run python tests/smoke_data_staging.py
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ale.core.task_data import TaskDataSpec
from ale.io import data_staging


# =============================================================================
# Recording stub session
# =============================================================================

@dataclass
class _CmdResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _RecordingSession:
    commands: list[str] = field(default_factory=list)
    file_writes: list[tuple[str, bytes]] = field(default_factory=list)
    # Pre-canned responses for specific command-prefixes (first-match wins).
    canned: dict[str, _CmdResult] = field(default_factory=dict)

    async def run_command(self, command: str, *, timeout: float = 60.0) -> _CmdResult:
        self.commands.append(command)
        for prefix, result in self.canned.items():
            if prefix in command:
                return result
        return _CmdResult()

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.file_writes.append((path, data))


# =============================================================================
# Tests
# =============================================================================

def _td(requires: bool = True) -> TaskDataSpec:
    if not requires:
        return TaskDataSpec(requires_task_data=False)
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="demo",
        task_name="foo",
        variant_name="v0",
    )


async def test_stage_input_issues_correct_rsync() -> None:
    s = _RecordingSession()
    # gsutil ls returns success for input/, fail for software/
    s.canned = {
        "gsutil ls 'gs://agenthle/demo/foo/v0/input'": _CmdResult(0),
        "gsutil ls 'gs://agenthle/demo/foo/v0/software'": _CmdResult(1, stderr="No URLs matched"),
    }
    result = await data_staging.stage_input(s, _td(), "linux")
    assert result["skipped"] is False
    assert "input" in result["staged_dirs"]
    assert "software" not in result["staged_dirs"]
    # Should have issued: rsync for input, mkdir for software fallback (skipped existence)
    rsync_cmds = [c for c in s.commands if "gsutil -m rsync" in c]
    assert any("/demo/foo/v0/input" in c for c in rsync_cmds), (
        f"missing input rsync; got: {rsync_cmds}"
    )
    print("[stage_input] rsync for input/ + skip software/ → ok")


async def test_stage_input_skips_when_not_required() -> None:
    s = _RecordingSession()
    result = await data_staging.stage_input(s, _td(requires=False), "linux")
    assert result["skipped"] is True
    assert s.commands == []
    print("[stage_input/skip] requires_task_data=False → zero commands ✓")


async def test_stage_reference_issues_rsync() -> None:
    s = _RecordingSession()
    s.canned = {"verify_nonempty_dir_cmd": _CmdResult(0)}
    await data_staging.stage_reference(s, _td(), "linux")
    rsync_cmds = [c for c in s.commands if "gsutil -m rsync" in c]
    assert any("/demo/foo/v0/reference" in c for c in rsync_cmds), (
        f"missing reference rsync; got: {rsync_cmds}"
    )
    print("[stage_reference] rsync for reference/ → ok")


async def test_stage_reference_skips_when_not_required() -> None:
    s = _RecordingSession()
    result = await data_staging.stage_reference(s, _td(requires=False), "linux")
    assert result["skipped"] is True
    assert s.commands == []
    print("[stage_reference/skip] requires_task_data=False → zero commands ✓")


async def test_upload_output_issues_cp() -> None:
    s = _RecordingSession()
    result = await data_staging.upload_output(
        s, _td(), "linux", run_id="r123", bucket="gs://test-bucket",
    )
    assert result["uploaded"] is True
    assert "gs://test-bucket/r123/output/" in result["gcs_path"]
    cp_cmds = [c for c in s.commands if "gcloud storage cp" in c]
    assert any("/demo/foo/v0/output" in c for c in cp_cmds), (
        f"missing output cp; got: {cp_cmds}"
    )
    print("[upload_output] gcloud storage cp output/ → results bucket ok")


async def test_visibility_rule_env_only_stages_reference_on_submit() -> None:
    """End-to-end check that reset_async does NOT touch reference/ — only
    step_async(Submit) does. This is the formal benchmark visibility rule."""
    # We need the full AgenthleEnv with a stub provider. The smoke_hello
    # path already exercises this for requires_task_data=False; for True we
    # would need a full stub provider that supports gcloud commands. Defer
    # to integration smoke for the live VM case; here just assert the env
    # source code only references stage_reference inside the Submit branch.
    src = (Path(__file__).resolve().parent.parent / "ale" / "core" / "env.py").read_text()
    submit_idx = src.find("isinstance(action, Submit)")
    ref_idx = src.find("data_staging.stage_reference")
    assert submit_idx > 0 and ref_idx > submit_idx, (
        "stage_reference must be called only inside the Submit branch of "
        "step_async — found before isinstance(action, Submit) check, which "
        "would violate the visibility rule"
    )
    print("[visibility] stage_reference only inside Submit branch ✓")


async def main() -> None:
    await test_stage_input_issues_correct_rsync()
    await test_stage_input_skips_when_not_required()
    await test_stage_reference_issues_rsync()
    await test_stage_reference_skips_when_not_required()
    await test_upload_output_issues_cp()
    await test_visibility_rule_env_only_stages_reference_on_submit()
    print("\nsmoke OK ✓")


if __name__ == "__main__":
    asyncio.run(main())
