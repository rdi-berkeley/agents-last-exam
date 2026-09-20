"""AgentHLE task: abm_hangzhou_metro."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

from tasks.transport_safety.abm_hangzhou_metro.scripts.score_outputs import (
    PUBLIC_CONTRACT,
    ScoreResult,
    score_output_bundle,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if await session.file_exists(path) or await session.directory_exists(path):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


async def _executable(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    result = await session.run_command(f'test -x "{path}" && printf "__ok__"', check=False)
    if result.get("stdout", "").strip() == "__ok__":
        return True
    logger.error("Non-executable %s: %s", label, path)
    return False


@dataclass
class HangzhouMetroConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "transport_safety"
    TASK_NAME: str = "abm_hangzhou_metro"
    VARIANT_NAME: str = "base"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def simulation_contract_file(self) -> str:
        return f"{self.input_dir}/simulation_contract.md"

    @property
    def candidate_manifest(self) -> str:
        return f"{self.remote_output_dir}/simulation_manifest.json"

    @property
    def afc_csv(self) -> str:
        return f"{self.input_dir}/data/afc_hangzhou.csv"

    @property
    def lines_geojson(self) -> str:
        return f"{self.input_dir}/gis/hangzhou_lines.json"

    @property
    def stations_geojson(self) -> str:
        return f"{self.input_dir}/gis/hangzhou_stations.json"

    @property
    def station_sequence_csv(self) -> str:
        return f"{self.input_dir}/network_config/station_sequence.csv"

    @property
    def operation_parameters_json(self) -> str:
        return f"{self.input_dir}/network_config/operation_parameters.json"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def bootstrap_wrapper(self) -> str:
        return f"{self.software_dir}/bootstrap_uv_env.sh"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def candidate_csv(self) -> str:
        return f"{self.remote_output_dir}/passenger_records.csv"

    @property
    def candidate_report(self) -> str:
        return f"{self.remote_output_dir}/validation_report.txt"

    @property
    def candidate_scatter(self) -> str:
        return f"{self.remote_output_dir}/scatter_plot.png"

    @property
    def runtime_state_dir(self) -> str:
        return f"{self.remote_output_dir}/.runtime_state"

    @property
    def reference_csv(self) -> str:
        return f"{self.reference_dir}/passenger_records.csv"

    @property
    def reference_report(self) -> str:
        return f"{self.reference_dir}/validation_report.txt"

    @property
    def evaluation_contract_file(self) -> str:
        return f"{self.reference_dir}/evaluation_contract.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to produce a one-day Hangzhou metro passenger simulation output bundle.

Visible task files:
- `{self.task_prompt_file}`
- `{self.output_contract_file}`
- `{self.simulation_contract_file}`
- `{self.afc_csv}`
- `{self.lines_geojson}`
- `{self.stations_geojson}`
- `{self.station_sequence_csv}`
- `{self.operation_parameters_json}`
- `{self.runtime_env_dir}/`
- `{self.bootstrap_wrapper}`
- `{self.python_wrapper}`

What you must do:
1. Read `{self.task_prompt_file}`, `{self.output_contract_file}`, and `{self.simulation_contract_file}`.
2. Implement the public v2 stochastic model, including the published GIS distances, MNL route choice, train schedule, entry limits, FIFO queues, and train capacities. Choose and declare an unsigned 64-bit seed. Do not fit to AFC end times.
3. Write the final required deliverables under `{self.remote_output_dir}`:
   - `{self.candidate_csv}`
   - `{self.candidate_report}`
   - `{self.candidate_manifest}`
4. You may also write `{self.candidate_scatter}` if you want an optional plot artifact.

Rules:
- Keep trip identity aligned to the visible AFC records.
- Every trip completed by the public model under your declared seed is required; do not invent output for unserved demand. Scoring recomputes from public inputs, not a hidden historical answer table.
- Do not modify files under `input/`.
- Keep final deliverables at the output-directory root.
- If you use the staged Python runtime, the helper wrappers may create task-local state under `{self.runtime_state_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "task_prompt_file": self.task_prompt_file,
                "output_contract_file": self.output_contract_file,
                "simulation_contract_file": self.simulation_contract_file,
                "candidate_manifest": self.candidate_manifest,
                "afc_csv": self.afc_csv,
                "lines_geojson": self.lines_geojson,
                "stations_geojson": self.stations_geojson,
                "station_sequence_csv": self.station_sequence_csv,
                "operation_parameters_json": self.operation_parameters_json,
                "runtime_env_dir": self.runtime_env_dir,
                "bootstrap_wrapper": self.bootstrap_wrapper,
                "python_wrapper": self.python_wrapper,
                "candidate_csv": self.candidate_csv,
                "candidate_report": self.candidate_report,
                "candidate_scatter": self.candidate_scatter,
                "runtime_state_dir": self.runtime_state_dir,
                "reference_csv": self.reference_csv,
                "reference_report": self.reference_report,
                "evaluation_contract_file": self.evaluation_contract_file,
            }
        )
        return metadata


config = HangzhouMetroConfig()


@cb.tasks_config(split="train")
def load():
    cfg = HangzhouMetroConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_score(result: ScoreResult) -> None:
    logger.info("score=%s passed=%s reason=%s", result.score, result.passed, result.reason)
    for key, value in result.details.items():
        logger.info("detail %s=%s", key, value)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    evaluator_keys = (
        ("afc_csv", "data/afc_hangzhou.csv"),
        ("lines_geojson", "gis/hangzhou_lines.json"),
        ("stations_geojson", "gis/hangzhou_stations.json"),
        ("station_sequence_csv", "network_config/station_sequence.csv"),
        ("operation_parameters_json", "network_config/operation_parameters.json"),
        ("simulation_contract_file", "simulation_contract.md"),
    )
    for key, label in evaluator_keys:
        if not await session.file_exists(meta[key]):
            raise RuntimeError(f"evaluator-controlled {label} missing: {meta[key]}")

    contract_bytes = await session.read_bytes(meta["simulation_contract_file"])
    if contract_bytes != PUBLIC_CONTRACT.read_bytes():
        raise RuntimeError("evaluator/public contract mismatch: stage simulation_contract.md v2")

    agent_keys = (
        ("candidate_csv", "passenger_records.csv"),
        ("candidate_report", "validation_report.txt"),
        ("candidate_manifest", "simulation_manifest.json"),
    )
    for key, label in agent_keys:
        if not await session.file_exists(meta[key]):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    scratch = Path(
        os.environ.get("ALE_EVALUATOR_SCRATCH_DIR", Path.home() / ".cache/ale-evaluations")
    )
    scratch.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(scratch.resolve())],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if filesystem in {"tmpfs", "ramfs"} or not filesystem:
        raise RuntimeError("Hangzhou evaluator scratch must be disk-backed")
    with tempfile.TemporaryDirectory(prefix="abm_hangzhou_metro_eval_", dir=scratch) as tmp_dir:
        tmp = Path(tmp_dir)
        output_dir = tmp / "candidate"
        input_dir = tmp / "input"
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, relative in evaluator_keys:
            destination = input_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                contract_bytes
                if key == "simulation_contract_file"
                else await session.read_bytes(meta[key])
            )
        for key, relative in agent_keys:
            (output_dir / relative).write_bytes(await session.read_bytes(meta[key]))

        result = score_output_bundle(
            output_dir=output_dir,
            input_dir=input_dir,
        )
        _log_score(result)
        return [result.score]
