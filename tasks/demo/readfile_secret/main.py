"""Demo task: ``demo/readfile_secret`` — proves native read_file forwards content.

A fresh, unguessable random token is generated per run and written into
``input/secret.txt`` during setup. The agent must read that file via the native
``read_file`` tool (the shell is disabled in the smoke config) and write the
EXACT token to ``output/answer.txt``.

Because the token is random and never appears in the prompt, a non-zero score
can ONLY happen if the read_file tool-result content actually reached the model.
If the converter drops tool results (the agenthle OpenRouter bug), the model has
nothing to copy and must hallucinate — which will not match the token, scoring 0.

Agent's solving path:
  - read input/secret.txt  (read_file)
  - write the token verbatim to output/answer.txt  (write_file)
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)


def _make_token() -> str:
    # Short, unguessable, but trivial to copy verbatim (8 hex chars). Long
    # random hex (32 chars) is reproduced in *format* but reliably garbled by
    # gemini-2.5-flash even when it receives the content, which conflates a
    # copy-fidelity failure with a content-forwarding failure. A short token
    # cleanly separates the two: if the converter forwards the read_file
    # result, the model copies these 8 chars exactly.
    return f"ALE-SECRET-{secrets.token_hex(4)}-END"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "readfile_secret"
    VARIANT_NAME: str = "base"
    token: str = field(default_factory=_make_token)

    @property
    def answer_path(self) -> str:
        return f"{self.remote_output_dir}/answer.txt"

    @property
    def secret_path(self) -> str:
        return f"{self.input_dir}/secret.txt"

    @property
    def reference_path(self) -> str:
        return f"{self.reference_dir}/expected.txt"

    @property
    def task_description(self) -> str:
        return (
            f"There is a file at {self.secret_path} containing a single secret "
            f"token on one line.\n\n"
            f"Steps:\n"
            f"1. Read {self.secret_path} to obtain the exact secret token.\n"
            f"2. Write that exact token (and nothing else) to {self.answer_path}.\n\n"
            f"You only have file tools (read_file / write_file); the shell is not "
            f"available. The token is random — you MUST read it from the file. "
            f"Verification compares your output to the token exactly."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update(
            {
                "answer_path": self.answer_path,
                "secret_path": self.secret_path,
                "reference_path": self.reference_path,
                "token": self.token,
            }
        )
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": cfg.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    token = meta["token"]

    for d in (meta["input_dir"], meta["remote_output_dir"]):
        await session.run_command(f"mkdir -p {d!r}", check=False)
    await session.run_command(f"rm -f {meta['answer_path']!r}", check=False)
    await session.run_command(f"rm -rf {meta['reference_dir']!r}", check=False)

    # Stage the unguessable secret the agent must read back.
    await session.write_file(meta["secret_path"], token + "\n")

    logger.info("[readfile_secret] staged secret token (len=%d)", len(token))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    out_path = meta["answer_path"]
    secret_path = meta["secret_path"]

    # Ground truth is the secret that was actually staged on disk, NOT a
    # regenerated metadata token. ``default_factory`` re-runs if the task is
    # re-instantiated between the setup and evaluate phases, so ``meta["token"]``
    # in this phase may not match what setup wrote. Reading the on-disk file is
    # the authoritative comparison and isolates the agent's behavior from any
    # token-regeneration in the task harness.
    try:
        truth = (await session.read_file(secret_path)).strip()
    except Exception as exc:
        logger.info("[readfile_secret] secret file unreadable at eval: %s", exc)
        return [0.0]

    try:
        actual = await session.read_file(out_path)
    except Exception as exc:
        logger.info("[readfile_secret] output unreadable: %s", exc)
        return [0.0]

    if truth and truth in actual:
        logger.info("[readfile_secret] PASS — staged secret present in output")
        return [1.0]
    logger.info(
        "[readfile_secret] FAIL — secret absent. output=%r staged_secret=%r",
        actual[:200],
        truth,
    )
    return [0.0]
