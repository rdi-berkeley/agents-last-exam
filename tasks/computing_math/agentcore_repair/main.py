"""ALE task: computing_math/agentcore_repair.

A SWE-bench-style repo-edit task, patterned on tasks/demo/hello/main.py. start() makes a writable
copy of the package the agent edits in place (the framework auto-stages input/); evaluate() reads
the edited modules via the session, runs the hidden conformance suite host-side, and gate-and-scores
it (scripts/score_agentcore_repair.py). No LLM judge.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_agentcore_repair import score_submission  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "agentcore_repair"
VARIANT_NAME = "base"
AGENTCORE_MODULES = ["__init__.py", "agent.py", "tools.py", "model.py", "types.py", "errors.py", "adapters.py"]


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def agentcore_dir(self) -> str:
        # Writable working copy of the package the agent edits in place (made by start()).
        return f"{self.task_dir}/agentcore"

    @property
    def tests_visible_dir(self) -> str:
        return f"{self.task_dir}/tests_visible"

    @property
    def readme_file(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def conformance_reference(self) -> str:
        return f"{self.reference_dir}/conformance_runner.py"

    @property
    def task_description(self) -> str:
        return (
            "You are a senior engineer working on an Ubuntu machine.\n\n"
            f"`{self.agentcore_dir}` is a partially-working Python tool-calling agent runtime. The "
            f"happy path works, but several behaviors in `{self.readme_file}` are broken or "
            "unimplemented. Repair the bugs and implement the missing loop-detection feature so the "
            "runtime conforms to the spec. Edit the source in place (e.g. agentcore/agent.py and "
            "agentcore/tools.py); do NOT change the public API or the exported names.\n\n"
            f"Check progress with `cd {self.task_dir} && python -m pytest tests_visible/ -q`. Your "
            f"repaired `{self.agentcore_dir}` package is the deliverable."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update(
            {
                "agentcore_dir": self.agentcore_dir,
                "tests_visible_dir": self.tests_visible_dir,
                "readme_file": self.readme_file,
                "conformance_reference": self.conformance_reference,
                "modules": AGENTCORE_MODULES,
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
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    # Make a writable copy of the package + visible tests the agent works on (framework staged input/).
    await session.run_command(f"rm -rf {meta['agentcore_dir']!r}", check=False)
    await session.run_command(f"cp -r {meta['input_dir']!r}/agentcore {meta['agentcore_dir']!r}", check=False)
    await session.run_command(f"cp -r {meta['input_dir']!r}/tests_visible {meta['tests_visible_dir']!r}", check=False)
    try:
        await session.read_file(f"{meta['agentcore_dir']}/__init__.py")
    except Exception as exc:
        raise RuntimeError(f"staged agentcore missing ({exc})")
    logger.info("[agentcore_repair] writable package staged at %s", meta["agentcore_dir"])


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        base = meta["agentcore_dir"]
        # Enumerate the package tree at eval time so a correct fix that ADDS a module or
        # subpackage is graded in full -- not just the originally-seeded module list.
        try:
            res = await session.run_command(f"cd {base!r} && find . -name '*.py' -type f", check=False)
            listing = res if isinstance(res, str) else (getattr(res, "stdout", "") or "")
            rels = [
                (ln.strip()[2:] if ln.strip().startswith("./") else ln.strip())
                for ln in listing.splitlines()
                if ln.strip().endswith(".py")
            ]
        except Exception:
            rels = []
        if not rels:  # fallback: the originally-seeded top-level modules
            rels = list(meta["modules"])
        modules: dict[str, bytes] = {}
        for rel in rels:
            try:
                modules[rel] = await session.read_bytes(f"{base}/{rel}")
            except Exception:
                continue
        if "__init__.py" not in modules:
            logger.info("[agentcore_repair] package missing from workspace")
            return [0.0]
        runner_bytes = await session.read_bytes(meta["conformance_reference"])
        score, report = score_submission(modules, runner_bytes)
    except Exception as exc:
        logger.exception("[agentcore_repair] evaluation failed: %s", exc)
        return [0.0]
    logger.info("[agentcore_repair] eval_report=%s", json.dumps(report, sort_keys=True))
    return [score]
