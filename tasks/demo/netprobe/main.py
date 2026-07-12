"""Demo task ``demo/netprobe`` — cleanly shows whether the agent has internet.

Self-contained: **no input data, no staging** (`REQUIRES_TASK_DATA = False`), so
it runs on the current image without re-baking. The agent just probes the
network and reports; the score does NOT trust that report — `evaluate()` runs
its OWN deterministic probe on the VM (hallucination-proof: the agent knows
`example.com`'s content from training, so a self-report alone would be
unreliable).

Semantics of the score = "is the agent network-isolated":
  - `open`      → the VM reaches `example.com` → **0.0** (agent CAN reach the web)
  - `allowlist` → `example.com` is dropped     → **1.0** (agent is isolated)
So flipping the card's `vm.network` between `open` and `allowlist` flips the
score 0.0 ↔ 1.0 — a direct, deterministic demonstration.

The agent still needs the model to do anything, so a run that reaches
`evaluate()` at all also confirms the model endpoint stayed reachable under the
policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

_PROBE_URL = "https://example.com"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "netprobe"
    VARIANT_NAME: str = "base"
    REQUIRES_TASK_DATA: bool = False  # self-contained: no input staging

    @property
    def report_path(self) -> str:
        return f"{self.remote_output_dir}/net_report.txt"

    @property
    def task_description(self) -> str:
        # Deliberately trivial + always-completable: running at all needs a model
        # round-trip, so a completed run confirms the model stayed reachable under
        # the policy. Whether the *network* is reachable is judged by evaluate()'s
        # own deterministic probe, not by anything the agent does.
        return (
            f"Write exactly the word DONE to the file {self.report_path} using "
            "your file tools, then stop. Do not run any other commands."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m["report_path"] = self.report_path
        m["probe_url"] = _PROBE_URL
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
    await session.run_command(f"rm -f {meta['report_path']!r}", check=False)
    logger.info("[netprobe] ready (agent will probe %s)", meta["probe_url"])


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    url = meta["probe_url"]

    # Deterministic, agent-independent probe: can the VM actually reach the web
    # under the applied policy? Write the HTTP code to a file and read it back
    # (avoids depending on run_command's return shape across cua versions).
    probe = (
        f"curl -sS -m 12 -o /dev/null -w '%{{http_code}}' {url} "
        "> /tmp/netprobe_code 2>/dev/null; "
        "test -s /tmp/netprobe_code || printf 000 > /tmp/netprobe_code"
    )
    await session.run_command(probe, check=False)
    try:
        code = (await session.read_file("/tmp/netprobe_code")).strip()
    except Exception:
        code = "000"
    reachable = bool(code) and not code.startswith("0")

    try:
        marker = (await session.read_file(meta["report_path"])).strip()
    except Exception:
        marker = "(missing)"

    logger.info("[netprobe] deterministic probe %s → HTTP %r (reachable=%s)", url, code, reachable)
    logger.info("[netprobe] agent marker (model reachable if present): %r", marker[:40])
    if reachable:
        logger.info("[netprobe] VERDICT: agent CAN reach the internet → NOT isolated → 0.0")
        return [0.0]
    logger.info("[netprobe] VERDICT: external egress BLOCKED → agent isolated → 1.0")
    return [1.0]
