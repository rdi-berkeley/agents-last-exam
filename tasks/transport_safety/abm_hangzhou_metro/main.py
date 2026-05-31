"""AgentHLE task: abm_hangzhou_metro."""

from __future__ import annotations

import logging
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

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import ScoreResult, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
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

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

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
        return f"{self.output_dir}/passenger_records.csv"

    @property
    def candidate_report(self) -> str:
        return f"{self.output_dir}/validation_report.txt"

    @property
    def candidate_scatter(self) -> str:
        return f"{self.output_dir}/scatter_plot.png"

    @property
    def runtime_state_dir(self) -> str:
        return f"{self.output_dir}/.runtime_state"

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
- `{self.afc_csv}`
- `{self.lines_geojson}`
- `{self.stations_geojson}`
- `{self.station_sequence_csv}`
- `{self.operation_parameters_json}`
- `{self.runtime_env_dir}/`
- `{self.bootstrap_wrapper}`
- `{self.python_wrapper}`

What you must do:
1. Read `{self.task_prompt_file}` and `{self.output_contract_file}`.
2. Build a metro simulation workflow from the staged AFC, GIS, and network-configuration files.
3. Write the final required deliverables under `{self.output_dir}`:
   - `{self.candidate_csv}`
   - `{self.candidate_report}`
4. You may also write `{self.candidate_scatter}` if you want an optional plot artifact.

Rules:
- Keep trip identity aligned to the visible AFC records.
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
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
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
    agent_keys = (("candidate_csv", "candidate passenger_records.csv"),
                  ("candidate_report", "candidate validation_report.txt"))
    for key, label in agent_keys:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    evaluator_keys = (("reference_csv", "reference passenger_records.csv"),
                      ("reference_report", "reference validation_report.txt"),
                      ("afc_csv", "visible afc csv"))
    for key, label in evaluator_keys:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            raise RuntimeError(
                f"evaluator-controlled {label} missing: {meta[key]}"
            )

    with tempfile.TemporaryDirectory(prefix="abm_hangzhou_metro_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        output_dir = tmp / "candidate"
        reference_dir = tmp / "reference"
        input_dir = tmp / "input"
        (input_dir / "data").mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "passenger_records.csv").write_bytes(
            await session.read_bytes(meta["candidate_csv"])
        )
        (output_dir / "validation_report.txt").write_bytes(
            await session.read_bytes(meta["candidate_report"])
        )
        (reference_dir / "passenger_records.csv").write_bytes(
            await session.read_bytes(meta["reference_csv"])
        )
        (reference_dir / "validation_report.txt").write_bytes(
            await session.read_bytes(meta["reference_report"])
        )
        (input_dir / "data" / "afc_hangzhou.csv").write_bytes(
            await session.read_bytes(meta["afc_csv"])
        )

        result = score_output_bundle(
            output_dir=output_dir,
            reference_dir=reference_dir,
            input_dir=input_dir,
        )
        _log_score(result)
        return [result.score]
