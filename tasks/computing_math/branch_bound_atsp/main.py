"""AgentHLE task for computing_math/branch_bound_atsp."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import DATA_ROOT, LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import score_outputs  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "branch_bound_atsp"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIR_NAMES = {"output"}
ADMIN_OUTPUT_PREFIX = "output_admin_"


def _normalize_output_dir_name(raw: str) -> str:
    normalized = raw.replace("\\", "/").strip("/")
    if not normalized or "/" in normalized:
        raise ValueError(f"REMOTE_OUTPUT_DIR must be a single directory name, got {raw!r}")
    if normalized in ALLOWED_OUTPUT_DIR_NAMES or normalized.startswith(ADMIN_OUTPUT_PREFIX):
        return normalized
    raise ValueError(
        "REMOTE_OUTPUT_DIR must be one of "
        f"{sorted(ALLOWED_OUTPUT_DIR_NAMES)} or start with {ADMIN_OUTPUT_PREFIX!r}"
    )


class BranchBoundATSPConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self, remote_output_dir: str | None = None) -> None:
        self.REMOTE_OUTPUT_DIR = remote_output_dir or os.environ.get("REMOTE_OUTPUT_DIR", "output")
        self.DOMAIN_NAME = DOMAIN_NAME
        self.TASK_NAME = TASK_NAME
        self.VARIANT_NAME = VARIANT_NAME
        self.OS_TYPE = "linux"
        self.REQUIRES_TASK_DATA = True

    @property
    def output_dir_name(self) -> str:
        return _normalize_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def task_dir(self) -> str:
        return f"{DATA_ROOT}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return f"{self.task_dir}/input"

    @property
    def reference_dir(self) -> str:
        return f"{self.task_dir}/reference"

    @property
    def software_dir(self) -> str:
        return f"{self.task_dir}/software"

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def problem_spec_file(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/TASK_PROMPT.md"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def python_entry(self) -> str:
        return f"{self.software_dir}/python"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/results.json"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/results.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Task prompt: `{self.task_prompt_file}`
- ATSP problem specification: `{self.problem_spec_file}`
- Runtime manifest: `{self.runtime_manifest}`
- Python entry point with task dependencies: `{self.python_entry}`

## Your Task
Implement a branch-and-bound solver for the asymmetric travelling salesman
problem described in the problem specification. Solve the 12-city and 20-city
instances to proven optimality, solve the 35-city instance to at most 0.5%
optimality gap, and write the required JSON report to:

`{self.output_file}`

Use only Python, NumPy, and SciPy. Do not use external optimization solvers or
TSP packages. Do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "output_dir_name": self.output_dir_name,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
                "problem_spec_file": self.problem_spec_file,
                "task_prompt_file": self.task_prompt_file,
                "runtime_manifest": self.runtime_manifest,
                "python_entry": self.python_entry,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = BranchBoundATSPConfig()


@cb.tasks_config(split="train")
def load():
    cfg = BranchBoundATSPConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not await session.exists(meta["output_file"]):
        logger.error("missing results.json at %s", meta["output_file"])
        return [0.0]
    if not await session.exists(meta["reference_file"]):
        raise RuntimeError(f"evaluator-controlled reference missing: {meta['reference_file']}")

    submission = json.loads((await session.read_bytes(meta["output_file"])).decode("utf-8-sig"))
    reference = json.loads((await session.read_bytes(meta["reference_file"])).decode("utf-8-sig"))
    report = score_outputs.score_payload(submission, reference)

    logger.info("score report: %s", json.dumps(report, ensure_ascii=True))
    return [float(report.get("score", 0.0))]
