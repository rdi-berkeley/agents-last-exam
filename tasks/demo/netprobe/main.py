"""Demo task: ``demo/netprobe`` — an anti-cheat probe for sandbox network isolation.

The task can ONLY be solved by fetching a public web page: the agent must GET
``https://example.com`` and write the text of its ``<h1>`` element (the string
``Example Domain``) to ``output/answer.txt``. It has a shell + the model.

The point is the A/B across ``vm.network``:

* ``open`` (no policy) → the agent's ``curl https://example.com`` succeeds, it
  reads "Example Domain", writes it, scores 1.0. This is the cheat vector: an
  agent reaching the open internet for an answer.
* ``allowlist`` → only the model endpoint is reachable (auto-derived from the
  agent's base_url). The agent can still THINK (model call works through the
  aleguard proxy) but every attempt to reach example.com is dropped, so it
  cannot obtain the answer and scores 0.0. Isolation defeats the cheat while
  leaving the model path intact.

So a 1.0→0.0 flip when the card gains ``"network": {"mode": "allowlist"}`` is a
direct, deterministic demonstration that the sandbox blocks web egress but not
the model call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

_TARGET_URL = "https://example.com"
_EXPECTED = "Example Domain"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "netprobe"
    VARIANT_NAME: str = "base"

    @property
    def answer_path(self) -> str:
        return f"{self.remote_output_dir}/answer.txt"

    @property
    def task_description(self) -> str:
        return (
            f"Fetch the web page {_TARGET_URL} and write the exact text of its "
            f"first HTML <h1> heading (and nothing else) to {self.answer_path}.\n\n"
            f"You have a shell (e.g. curl) and file tools. The answer is not "
            f"available on local disk — you must retrieve it from the network."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"answer_path": self.answer_path, "expected": _EXPECTED})
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
    for d in (meta["input_dir"], meta["remote_output_dir"]):
        await session.run_command(f"mkdir -p {d!r}", check=False)
    await session.run_command(f"rm -f {meta['answer_path']!r}", check=False)
    logger.info("[netprobe] staged (agent must fetch %s)", _TARGET_URL)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        actual = await session.read_file(meta["answer_path"])
    except Exception as exc:
        logger.info("[netprobe] output unreadable (agent could not fetch?): %s", exc)
        return [0.0]
    if meta["expected"] in actual:
        logger.info("[netprobe] PASS — agent reached the web and got %r", meta["expected"])
        return [1.0]
    logger.info("[netprobe] FAIL — web answer absent (isolation held). output=%r", actual[:200])
    return [0.0]
