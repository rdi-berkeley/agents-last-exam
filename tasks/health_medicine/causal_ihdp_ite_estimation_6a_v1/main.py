"""AgentHLE task: causal_ihdp_ite_estimation_6a_v1."""

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path, PurePosixPath
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

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import HARD_FAIL_SCORE, PASS_THRESHOLD, ScoreResult, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "causal_ihdp_ite_estimation_6a_v1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_ROOT = f"/tmp/agenthle_eval/{TASK_NAME}"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _is_fixture_or_admin_output(output_dir: str, task_dir: str) -> bool:
    default_output_dir = _remote_join(task_dir, "output")
    return output_dir != default_output_dir


class CausalIHDPITEConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
        )

    @property
    def runtime_env_dir(self) -> str:
        return _remote_join(self.input_dir, "runtime_env")

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg")

    @property
    def output_dir(self) -> str:
        if self.OUTPUT_SUBDIR == "output_test_pos":
            return self.output_test_pos_dir
        if self.OUTPUT_SUBDIR == "output_test_neg":
            return self.output_test_neg_dir
        return _remote_join(self.task_dir, self.OUTPUT_SUBDIR)

    @property
    def task_brief_file(self) -> str:
        return _remote_join(self.input_dir, "task_brief.md")

    @property
    def task_readme_file(self) -> str:
        return _remote_join(self.input_dir, "task_readme.md")

    @property
    def feature_schema_file(self) -> str:
        return _remote_join(self.input_dir, "feature_schema.json")

    @property
    def train_file(self) -> str:
        return _remote_join(self.input_dir, "train.csv")

    @property
    def dummy_test_dir(self) -> str:
        return _remote_join(self.input_dir, "dummy_test")

    @property
    def dummy_test_file(self) -> str:
        return _remote_join(self.dummy_test_dir, "test.csv")

    @property
    def runtime_pyproject(self) -> str:
        return _remote_join(self.runtime_env_dir, "pyproject.toml")

    @property
    def runtime_lock(self) -> str:
        return _remote_join(self.runtime_env_dir, "uv.lock")

    @property
    def runtime_requirements(self) -> str:
        return _remote_join(self.runtime_env_dir, "requirements.txt")

    @property
    def runtime_wheelhouse(self) -> str:
        return _remote_join(self.runtime_env_dir, "wheelhouse")

    @property
    def python_wrapper(self) -> str:
        return _remote_join(self.software_dir, "python_with_task_deps.sh")

    @property
    def predict_sh(self) -> str:
        return _remote_join(self.output_dir, "predict.sh")

    @property
    def output_csv(self) -> str:
        return _remote_join(self.output_dir, "output.csv")

    @property
    def artifacts_dir(self) -> str:
        return _remote_join(self.output_dir, "artifacts")

    @property
    def model_selection_csv(self) -> str:
        return _remote_join(self.artifacts_dir, "model_selection.csv")

    @property
    def overlap_csv(self) -> str:
        return _remote_join(self.artifacts_dir, "overlap_diagnostics.csv")

    @property
    def subgroup_csv(self) -> str:
        return _remote_join(self.artifacts_dir, "subgroup_ite.csv")

    @property
    def run_notes_txt(self) -> str:
        return _remote_join(self.artifacts_dir, "run_notes.txt")

    @property
    def hidden_test_dir(self) -> str:
        return _remote_join(self.reference_dir, "hidden_test")

    @property
    def hidden_test_file(self) -> str:
        return _remote_join(self.hidden_test_dir, "test.csv")

    @property
    def gold_output_file(self) -> str:
        return _remote_join(self.reference_dir, "gold_output.csv")

    @property
    def eval_tmp_dir(self) -> str:
        return _remote_join(EVAL_TMP_ROOT, self.OUTPUT_SUBDIR.replace("/", "_"))

    @property
    def eval_tmp_output_csv(self) -> str:
        return _remote_join(self.eval_tmp_dir, "output.csv")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux causal-inference benchmark.

Read these staged solve-time inputs first:
- `{self.task_brief_file}`
- `{self.task_readme_file}`
- `{self.feature_schema_file}`
- `{self.train_file}`
- `{self.dummy_test_file}`

If you want the pinned scientific Python stack, use:
- `{self.python_wrapper}`

Your job is to author `{_remote_join(self.task_dir, 'output', 'predict.sh')}` so that the evaluator can later run:

`bash output/predict.sh <hidden_test_dir> output/output.csv`

from the task root.

When your script runs, it must create these required files:
- `{_remote_join(self.task_dir, 'output', 'output.csv')}`
- `{_remote_join(self.task_dir, 'output', 'artifacts', 'model_selection.csv')}`
- `{_remote_join(self.task_dir, 'output', 'artifacts', 'overlap_diagnostics.csv')}`
- `{_remote_join(self.task_dir, 'output', 'artifacts', 'subgroup_ite.csv')}`
- `{_remote_join(self.task_dir, 'output', 'artifacts', 'run_notes.txt')}`

`output/output.csv` must have exactly these columns:
`replication,unit_id,mu0_hat,mu1_hat,ite_hat`

Rules:
- `ite_hat` must equal `mu1_hat - mu0_hat` row by row.
- You may keep helper code under `output/` and may emit extra artifact files if you want.
- Do not modify files under `input/`.
- During solve time, treat `input/` plus `software/` as your visible task surface; `reference/`, `output_test_pos/`, and `output_test_neg/` are evaluator-side paths, not solve-time inputs.
- Write solver-created files only under `{_remote_join(self.task_dir, 'output')}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_dir": self.output_dir,
                "task_brief_file": self.task_brief_file,
                "task_readme_file": self.task_readme_file,
                "feature_schema_file": self.feature_schema_file,
                "train_file": self.train_file,
                "dummy_test_dir": self.dummy_test_dir,
                "dummy_test_file": self.dummy_test_file,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "runtime_requirements": self.runtime_requirements,
                "runtime_wheelhouse": self.runtime_wheelhouse,
                "python_wrapper": self.python_wrapper,
                "predict_sh": self.predict_sh,
                "output_csv": self.output_csv,
                "artifacts_dir": self.artifacts_dir,
                "model_selection_csv": self.model_selection_csv,
                "overlap_csv": self.overlap_csv,
                "subgroup_csv": self.subgroup_csv,
                "run_notes_txt": self.run_notes_txt,
                "hidden_test_dir": self.hidden_test_dir,
                "hidden_test_file": self.hidden_test_file,
                "gold_output_file": self.gold_output_file,
                "eval_tmp_dir": self.eval_tmp_dir,
                "eval_tmp_output_csv": self.eval_tmp_output_csv,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = CausalIHDPITEConfig()


@cb.tasks_config(split="train")
def load():
    cfg = CausalIHDPITEConfig()
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
    logger.info(
        "[%s] score=%.6f passed=%s reason=%s hard_gate=%s",
        TASK_NAME,
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("[%s] details=%s", TASK_NAME, json.dumps(result.to_dict(), ensure_ascii=True))


def _final_score_from_raw(raw_score: float) -> float:
    if raw_score >= HARD_FAIL_SCORE:
        return 0.0
    return max(0.0, min(1.0, 1.0 - raw_score / PASS_THRESHOLD))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_eval_paths = [
        meta["task_dir"],
        meta["predict_sh"],
        meta["reference_dir"],
        meta["hidden_test_dir"],
        meta["hidden_test_file"],
        meta["gold_output_file"],
    ]
    missing_eval_paths = [path for path in required_eval_paths if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing_eval_paths:
        logger.error("[%s] missing evaluation paths: %s", TASK_NAME, "; ".join(missing_eval_paths))
        return [0.0]

    fixture_mode = _is_fixture_or_admin_output(meta["output_dir"], meta["task_dir"])
    if fixture_mode:
        output_csv_path = meta["eval_tmp_output_csv"]
        artifacts_dir = _remote_join(meta["eval_tmp_dir"], "artifacts")
        predict_sh_invocation = shlex.quote(meta["predict_sh"])
        hidden_test_invocation = shlex.quote(meta["hidden_test_dir"])
        output_csv_invocation = shlex.quote(output_csv_path)
        prep_command = _shell_join(
            [
                "bash",
                "-lc",
                (
                    f"rm -rf {shlex.quote(meta['eval_tmp_dir'])} && "
                    f"mkdir -p {shlex.quote(meta['eval_tmp_dir'])}"
                ),
            ]
        )
    else:
        output_csv_path = meta["output_csv"]
        artifacts_dir = meta["artifacts_dir"]
        predict_sh_invocation = "output/predict.sh"
        hidden_test_invocation = "reference/hidden_test"
        output_csv_invocation = "output/output.csv"
        prep_command = _shell_join(
            [
                "bash",
                "-lc",
                (
                    f"cd {shlex.quote(meta['task_dir'])} && "
                    "mkdir -p output/artifacts && "
                    "rm -f output/output.csv "
                    "output/artifacts/model_selection.csv "
                    "output/artifacts/overlap_diagnostics.csv "
                    "output/artifacts/subgroup_ite.csv "
                    "output/artifacts/run_notes.txt"
                ),
            ]
        )

    prep_result = await _run_command(session, prep_command, check=False)
    if prep_result.get("return_code", 0) != 0:
        logger.error("[%s] failed to prepare evaluation output dir: %s", TASK_NAME, prep_result)
        return [0.0]

    invoke_command = _shell_join(
        [
            "bash",
            "-lc",
            (
                f"cd {shlex.quote(meta['task_dir'])} && "
                f"bash {predict_sh_invocation} "
                f"{hidden_test_invocation} "
                f"{output_csv_invocation}"
            ),
        ]
    )
    invoke_result = await _run_command(session, invoke_command, check=False)
    if invoke_result.get("return_code", 0) != 0:
        logger.warning(
            "[%s] predict.sh exited non-zero but evaluation will continue if outputs are present: stdout=%s stderr=%s",
            TASK_NAME,
            _as_text(invoke_result.get("stdout", "")),
            _as_text(invoke_result.get("stderr", "")),
        )

    if not (await session.file_exists(output_csv_path) or await session.directory_exists(output_csv_path)):
        logger.error("[%s] missing scored output after predict.sh: %s", TASK_NAME, output_csv_path)
        return [0.0]

    artifact_paths = {
        "model_selection_csv": _remote_join(artifacts_dir, "model_selection.csv"),
        "overlap_csv": _remote_join(artifacts_dir, "overlap_diagnostics.csv"),
        "subgroup_csv": _remote_join(artifacts_dir, "subgroup_ite.csv"),
        "run_notes_txt": _remote_join(artifacts_dir, "run_notes.txt"),
    }

    try:
        result = score_output_bundle(
            candidate_output_csv=_as_text(await session.read_file(output_csv_path)),
            reference_gold_csv=_as_text(await session.read_file(meta["gold_output_file"])),
            model_selection_csv=(
                _as_text(await session.read_file(artifact_paths["model_selection_csv"]))
                if (await session.file_exists(artifact_paths["model_selection_csv"]) or await session.directory_exists(artifact_paths["model_selection_csv"]))
                else None
            ),
            overlap_csv=(
                _as_text(await session.read_file(artifact_paths["overlap_csv"]))
                if (await session.file_exists(artifact_paths["overlap_csv"]) or await session.directory_exists(artifact_paths["overlap_csv"]))
                else None
            ),
            subgroup_csv=(
                _as_text(await session.read_file(artifact_paths["subgroup_csv"]))
                if (await session.file_exists(artifact_paths["subgroup_csv"]) or await session.directory_exists(artifact_paths["subgroup_csv"]))
                else None
            ),
            run_notes_txt=(
                _as_text(await session.read_file(artifact_paths["run_notes_txt"]))
                if (await session.file_exists(artifact_paths["run_notes_txt"]) or await session.directory_exists(artifact_paths["run_notes_txt"]))
                else None
            ),
        )
    except Exception as exc:
        logger.error("[%s] evaluation failure: %s", TASK_NAME, exc)
        return [0.0]

    _log_score(result)
    return [_final_score_from_raw(float(result.score))]
