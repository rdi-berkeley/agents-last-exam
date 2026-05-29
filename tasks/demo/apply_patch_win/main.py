"""Demo task: ``demo/apply_patch_win`` — proves the patched Windows codex.exe.

Why this task exists
--------------------
The stock codex Windows shim writes ``apply_patch.bat``; when PowerShell/cmd
invoke it, cmd.exe re-tokenizes the patch body via ``%*`` expansion, which
corrupts cmd-toxic characters (``< > & | ( ) ^ %``) and trips the 8191-char
command-line cap. The patched codex.exe (cua-verse/codex @ ee3e51688) instead
ships ``apply_patch.exe`` (a hardlink of codex.exe) so the OS spawns it via
CreateProcessW directly, bypassing cmd.exe entirely.

So: a task that forces the agent to create a file *through the apply_patch
tool* whose body is full of cmd-toxic characters is a direct, behavioral proof.
If the patched binary is in place the bytes round-trip exactly → 1.0. If the
stock ``.bat`` shim is used the toxic chars get mangled → the byte-exact
compare fails → <1.0. There is no normalization in scoring of the very
characters under test, by design.

The agent is told to use apply_patch and NOT to use echo/redirection (which
would sidestep the very codepath we are validating).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig

logger = logging.getLogger(__name__)


# The payload deliberately packs every cmd-toxic metacharacter the .bat shim
# corrupts: < > & | ( ) ^ % plus doubled %% and a long line approaching the
# old command-line cap. ASCII-only so byte round-trip is unambiguous; LF
# (not CRLF) so the apply_patch-created file content is exactly these bytes.
_TOXIC_LINES = [
    "redirect: a < b > c",
    "pipe-and: x | y & z",
    "group: (alpha) ^ (beta)",
    "percent: 100% done, %PATH% %%literal",
    "mix: <<&&||(())^^%%>> end",
    "amp-cluster: && & &&& | || |||",
    "caret-escapes: ^< ^> ^& ^| ^( ^) ^^ ^%",
]
_EXPECTED = "\n".join(_TOXIC_LINES) + "\n"


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "apply_patch_win"
    VARIANT_NAME: str = "toxic"
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def answer_path(self) -> str:
        return rf"{self.remote_output_dir}\patched.txt"

    @property
    def input_request_path(self) -> str:
        return rf"{self.input_dir}\required_content.txt"

    @property
    def reference_path(self) -> str:
        return rf"{self.reference_dir}\expected.txt"

    @property
    def task_description(self) -> str:
        return (
            f"Goal: create the file {self.answer_path} with an EXACT required "
            f"content, using ONLY your apply_patch tool.\n\n"
            f"Steps:\n"
            f"1. Read {self.input_request_path}. It contains, between the two "
            f"'---' fences, the exact bytes the file must contain (it includes "
            f"shell metacharacters like < > & | ( ) ^ % — these are literal "
            f"content, not commands).\n"
            f"2. Use the apply_patch tool to ADD the file {self.answer_path} "
            f"with that exact content (a trailing newline after the last "
            f"line, LF line endings).\n\n"
            f"Constraints:\n"
            f"- You MUST create the file via apply_patch. Do NOT use echo, "
            f"redirection (>, >>), Set-Content, here-strings, or any other "
            f"shell write — only apply_patch.\n"
            f"- Reproduce every character literally. Do not escape, drop, or "
            f"alter any of the metacharacters.\n\n"
            f"Verification: the file must match the reference byte-for-byte "
            f"(after LF/CRLF normalization of line endings only — the "
            f"metacharacters themselves are compared literally)."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "input_dir": self.input_dir,
            "answer_path": self.answer_path,
            "input_request_path": self.input_request_path,
            "reference_path": self.reference_path,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={
            "provider": "computer",
            "setup_config": {"os_type": cfg.OS_TYPE},
        },
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Stage the required-content spec; ensure reference is hidden + output clean."""
    meta = task_cfg.metadata

    for d in (meta["input_dir"], meta["remote_output_dir"]):
        await session.run_command(
            f'powershell -NoProfile -Command "New-Item -ItemType Directory '
            f"-Force -Path '{d}' | Out-Null\"",
            check=False,
        )
    await session.run_command(
        f'powershell -NoProfile -Command "Remove-Item -Force -ErrorAction '
        f"SilentlyContinue '{meta['answer_path']}'\"",
        check=False,
    )
    await session.run_command(
        f'powershell -NoProfile -Command "Remove-Item -Recurse -Force '
        f"-ErrorAction SilentlyContinue '{meta['reference_dir']}'\"",
        check=False,
    )

    # The agent-visible spec: the exact required bytes between '---' fences.
    request = (
        "Create the output file with EXACTLY the content between the fences "
        "below (literal characters, trailing newline, LF line endings). Use "
        "your apply_patch tool only.\r\n"
        "---\r\n"
        + _EXPECTED.replace("\n", "\r\n")
        + "---\r\n"
    )
    await session.write_file(meta["input_request_path"], request)

    # Reference must NOT be visible during setup.
    try:
        await session.read_file(meta["reference_path"])
    except Exception:
        logger.info("reference correctly hidden during setup")
    else:
        raise RuntimeError(
            f"reference leaked during setup: {meta['reference_path']} readable"
        )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score: byte-exact (LF-normalized) → 1.0; any corruption/partial → 0.0–0.5."""
    meta = task_cfg.metadata
    out_path = meta["answer_path"]
    ref_path = meta["reference_path"]

    await session.run_command(
        f'powershell -NoProfile -Command "New-Item -ItemType Directory '
        f"-Force -Path '{meta['reference_dir']}' | Out-Null\"",
        check=False,
    )
    await session.write_file(ref_path, _EXPECTED)

    try:
        actual = await session.read_file(out_path)
    except Exception as exc:
        logger.info("output unreadable: %s", exc)
        return [0.0]

    # Normalize ONLY line endings — never the metacharacters under test.
    actual_norm = actual.replace("\r\n", "\n")
    expected_norm = _EXPECTED.replace("\r\n", "\n")
    if actual_norm == expected_norm:
        return [1.0]

    # Partial credit by exact-line hits, so a corrupted run scores >0 but <1
    # and the corruption is visible in the diff.
    expected_lines = [l for l in expected_norm.split("\n") if l.strip()]
    if not expected_lines:
        return [0.0]
    hits = sum(1 for l in expected_lines if l in actual_norm)
    logger.info(
        "apply_patch_win: %d/%d exact-line hits (corruption if <%d)",
        hits, len(expected_lines), len(expected_lines),
    )
    return [round(hits / len(expected_lines) * 0.5, 3)]
