"""AgentHLE task: sumo_urban_am_peak_calibration."""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
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
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "sumo_urban_am_peak_calibration"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


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
class SumoUrbanCalibrationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def schema_file(self) -> str:
        return f"{self.input_dir}/calibration_report.schema.json"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def example_report_file(self) -> str:
        return f"{self.input_dir}/examples/report_template.example.json"

    @property
    def starter_project_dir(self) -> str:
        return f"{self.input_dir}/starter_project"

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
    def task_runner_wrapper(self) -> str:
        return f"{self.software_dir}/run_in_task_env.sh"

    @property
    def runtime_state_dir(self) -> str:
        return f"{self.remote_output_dir}/.runtime_state"

    @property
    def network_file(self) -> str:
        return f"{self.remote_output_dir}/network.net.xml"

    @property
    def additionals_dir(self) -> str:
        return f"{self.remote_output_dir}/additionals"

    @property
    def bus_stops_file(self) -> str:
        return f"{self.additionals_dir}/bus_stops.add.xml"

    @property
    def detectors_public_file(self) -> str:
        return f"{self.additionals_dir}/detectors_public.add.xml"

    @property
    def turn_restrictions_file(self) -> str:
        return f"{self.additionals_dir}/turn_restrictions.add.xml"

    @property
    def tls_logic_file(self) -> str:
        return f"{self.additionals_dir}/tlsLogic.add.xml"

    @property
    def vtypes_file(self) -> str:
        return f"{self.additionals_dir}/vtypes.add.xml"

    @property
    def demand_dir(self) -> str:
        return f"{self.remote_output_dir}/demand"

    @property
    def calibrated_od_file(self) -> str:
        return f"{self.demand_dir}/calibrated_od.xml"

    @property
    def routes_file(self) -> str:
        return f"{self.demand_dir}/routes.rou.xml"

    @property
    def simulation_dir(self) -> str:
        return f"{self.remote_output_dir}/simulation"

    @property
    def sumocfg_file(self) -> str:
        return f"{self.simulation_dir}/sumocfg.sumocfg"

    @property
    def calibration_report_file(self) -> str:
        return f"{self.remote_output_dir}/calibration_report.json"

    @property
    def decisions_file(self) -> str:
        return f"{self.remote_output_dir}/DECISIONS.md"

    @property
    def evaluator_python(self) -> str:
        return f"{self.reference_dir}/evaluator_env/.venv/bin/python"

    @property
    def evaluator_script(self) -> str:
        return f"{self.reference_dir}/evaluator_only/evaluate.py"

    @property
    def ground_truth_dir(self) -> str:
        return f"{self.reference_dir}/evaluator_only/ground_truth"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to build a calibrated SUMO AM-peak submission for an urban grid sub-network.

Visible task files:
- `{self.task_prompt_file}`
- `{self.schema_file}`
- `{self.output_contract_file}`
- `{self.example_report_file}`
- `{self.starter_project_dir}/`
- `{self.runtime_env_dir}/`
- `{self.bootstrap_wrapper}`
- `{self.python_wrapper}`
- `{self.task_runner_wrapper}`

What you must do:
1. Read `{self.task_prompt_file}` and `{self.output_contract_file}` first.
2. Use the visible starter project under `{self.starter_project_dir}` to repair the broken SUMO configuration, demand, signal timing, and vehicle-type setup.
3. If you want the staged Python environment, materialize it with `{self.bootstrap_wrapper}` and run Python with `{self.python_wrapper}`. Use `{self.task_runner_wrapper}` for SUMO-facing commands inside that task-local environment.
4. Write the final submission tree under `{self.remote_output_dir}` with the required files at the documented locations.

Required output files:
- `{self.network_file}`
- `{self.bus_stops_file}`
- `{self.tls_logic_file}`
- `{self.vtypes_file}`
- `{self.calibrated_od_file}`
- `{self.routes_file}`
- `{self.sumocfg_file}`
- `{self.calibration_report_file}`
- `{self.decisions_file}`

Rules:
- Do not modify files under `{self.input_dir}`.
- Keep any runtime state you create outside `input/`; the staged wrappers already use `{self.runtime_state_dir}` by default.
- Do not rely on evaluator-only data or fixture directories.
- You do not need to submit simulation result folders; the evaluator re-runs your SUMO configuration itself.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_prompt_file": self.task_prompt_file,
                "schema_file": self.schema_file,
                "output_contract_file": self.output_contract_file,
                "example_report_file": self.example_report_file,
                "starter_project_dir": self.starter_project_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "bootstrap_wrapper": self.bootstrap_wrapper,
                "python_wrapper": self.python_wrapper,
                "task_runner_wrapper": self.task_runner_wrapper,
                "runtime_state_dir": self.runtime_state_dir,
                "network_file": self.network_file,
                "additionals_dir": self.additionals_dir,
                "bus_stops_file": self.bus_stops_file,
                "detectors_public_file": self.detectors_public_file,
                "turn_restrictions_file": self.turn_restrictions_file,
                "tls_logic_file": self.tls_logic_file,
                "vtypes_file": self.vtypes_file,
                "demand_dir": self.demand_dir,
                "calibrated_od_file": self.calibrated_od_file,
                "routes_file": self.routes_file,
                "simulation_dir": self.simulation_dir,
                "sumocfg_file": self.sumocfg_file,
                "calibration_report_file": self.calibration_report_file,
                "decisions_file": self.decisions_file,
                "evaluator_python": self.evaluator_python,
                "evaluator_script": self.evaluator_script,
                "ground_truth_dir": self.ground_truth_dir,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = SumoUrbanCalibrationConfig()


@cb.tasks_config(split="train")
def load():
    cfg = SumoUrbanCalibrationConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _cleanup_default_output(task_cfg, session: cb.DesktopSession) -> None:
    meta = task_cfg.metadata
    cleanup_paths = [
        meta["network_file"],
        meta["additionals_dir"],
        meta["demand_dir"],
        meta["simulation_dir"],
        meta["calibration_report_file"],
        meta["decisions_file"],
        meta["runtime_state_dir"],
    ]
    quoted = " ".join(shlex.quote(path) for path in cleanup_paths)
    await session.run_command(f"bash -lc {shlex.quote(f'rm -rf {quoted}')}", check=False)


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    required_eval_paths = [
        ("remote_output_dir", "submission output directory"),
        ("evaluator_script", "hidden evaluator script"),
        ("ground_truth_dir", "hidden evaluator ground truth"),
    ]
    for key, label in required_eval_paths:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            raise RuntimeError(f"missing {label} at {meta[key]}")
    # GCS staging strips Unix executable permissions; restore them on evaluator venv binaries
    venv_root = f'{meta["reference_dir"]}/evaluator_env/.venv'
    await session.run_command(
        f'chmod +x "{venv_root}/bin/python" && '
        f'find "{venv_root}/lib" -path "*/sumo/bin/*" -type f -exec chmod +x {{}} +'
    )
    if not await _executable(session, meta["evaluator_python"], label="hidden evaluator python"):
        raise RuntimeError(f"hidden evaluator python is not executable at {meta['evaluator_python']}")

    await session.interface.create_dir(EVAL_TMP_DIR)
    await session.interface.create_dir(f"{EVAL_TMP_DIR}/tmp")
    verifier_path = f"{EVAL_TMP_DIR}/verify_submission.py"
    await session.write_file(verifier_path, _read_script("verify_submission.py"))

    verify_cmd = " ".join(
        [
            "python",
            shlex.quote(verifier_path),
            "--submission-dir",
            shlex.quote(meta["remote_output_dir"]),
            "--evaluator-python",
            shlex.quote(meta["evaluator_python"]),
            "--evaluator-script",
            shlex.quote(meta["evaluator_script"]),
            "--ground-truth-dir",
            shlex.quote(meta["ground_truth_dir"]),
            "--tmp-dir",
            shlex.quote(f"{EVAL_TMP_DIR}/tmp"),
        ]
    )
    result = await session.run_command(
        "bash -lc " + json.dumps(verify_cmd),
        check=False,
    )

    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if not stdout:
        raise RuntimeError(f"hidden verifier produced empty stdout; stderr={stderr}")

    try:
        report = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"hidden verifier returned non-JSON stdout={stdout} stderr={stderr}")

    raw_wrapper_return_code = report.get("return_code", result.get("return_code"))
    try:
        wrapper_return_code = int(raw_wrapper_return_code) if raw_wrapper_return_code is not None else -1
    except (TypeError, ValueError):
        wrapper_return_code = -1
    if report.get("error") or wrapper_return_code not in {0, 1}:
        raise RuntimeError(
            "hidden verifier infrastructure failure: "
            + json.dumps(
                {
                    "wrapper_return_code": wrapper_return_code,
                    "error": report.get("error"),
                    "stderr": report.get("stderr"),
                    "session_stderr": stderr,
                },
                ensure_ascii=True,
            )
        )

    score = float(report.get("score", 0.0))
    logger.info("[%s] score=%.4f report=%s", TASK_ID, score, json.dumps(report, ensure_ascii=True))
    return [max(0.0, min(1.0, score))]


if __name__ == "__main__":  # pragma: no cover - manual sanity helper
    for task in load():
        print(task.description)
