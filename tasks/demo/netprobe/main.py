"""Demo task ``demo/netprobe`` — agent-driven internet-access probe.

Self-contained: **no input data, no staging** (`REQUIRES_TASK_DATA = False`), so
it runs on the current image without re-baking.

The agent must fetch a value it **cannot know without the network** — its own
public egress IP from ``https://api.ipify.org`` — and write it to
``output/answer.txt``. If it cannot reach the network it must instead write the
marker ``BLOCKED``. (A public-IP is used precisely because it is dynamic and
unguessable, so an isolated agent can't fabricate a plausible answer from memory
and must fall back to the marker.)

Scoring is on the agent's own output:
  - output contains ``BLOCKED``  → **1.0** (agent could NOT reach the web → isolated)
  - output is anything else (an IP) → **0.0** (agent reached the web)

So under ``vm.network: allowlist`` the fetch is dropped, the agent writes
``BLOCKED``, score 1.0; flip the card to ``open`` and it fetches its IP, score
0.0 — a clean, agent-driven demonstration. The agent needs the model to run at
all, so a scored run also confirms the model stayed reachable under the policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

_PROBE_URL = "https://api.ipify.org"
_MARKER = "BLOCKED"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "netprobe"
    VARIANT_NAME: str = "base"
    REQUIRES_TASK_DATA: bool = False  # self-contained: no input staging

    @property
    def answer_path(self) -> str:
        return f"{self.remote_output_dir}/answer.txt"

    @property
    def task_description(self) -> str:
        return (
            f"Fetch {_PROBE_URL} — it returns this machine's public IP address as "
            f"plain text. Write that IP (and nothing else) to {self.answer_path}.\n\n"
            f"If you cannot reach the network (the request fails or times out), "
            f"write exactly the single word {_MARKER} to {self.answer_path} "
            "instead. You have a shell (curl) and file tools; the answer is not "
            "on local disk, you must retrieve it from the network."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m["answer_path"] = self.answer_path
        m["marker"] = _MARKER
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r}", check=False)
    await session.run_command(f"rm -f {meta['answer_path']!r}", check=False)
    logger.info("[netprobe] ready (agent must fetch %s or write %s)", _PROBE_URL, _MARKER)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        out = (await session.read_file(meta["answer_path"])).strip()
    except Exception as exc:
        logger.info("[netprobe] output missing (%s) → 0.0", exc)
        return [0.0]

    if meta["marker"].lower() in out.lower():
        logger.info("[netprobe] agent wrote %s → could NOT reach web → isolated → 1.0", _MARKER)
        return [1.0]
    logger.info("[netprobe] agent wrote %r → reached the web → NOT isolated → 0.0", out[:60])
    return [0.0]
