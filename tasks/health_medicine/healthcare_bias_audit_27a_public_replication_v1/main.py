"""AgentHLE task: healthcare_bias_audit_27a_public_replication_v1."""

from __future__ import annotations

import json
import logging
import os
import shlex
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

try:
    from tasks.linux_runtime import LinuxTaskConfig
except ModuleNotFoundError:  # pragma: no cover - local import fallback only
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tasks.linux_runtime import LinuxTaskConfig

from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import REQUIRED_OUTPUT_FILES, ScoreResult, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "healthcare_bias_audit_27a_public_replication_v1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"


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


async def _list_relative_files(session: cb.DesktopSession, directory: str) -> list[str]:
    command = (
        "bash -lc "
        + shlex.quote(
            f'find {shlex.quote(directory)} -type f -printf "%P\\n" | sort'
        )
    )
    result = await _run_command(session, command, check=False)
    if result.get("return_code") not in (0, None):
        return []
    stdout = _as_text(result.get("stdout", ""))
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f passed=%s reason=%s hard_gate=%s",
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


class HealthcareBiasAuditConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    OS_TYPE: str = "linux"

    def __init__(self, variant_name: str = VARIANT_NAME) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=variant_name,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def env_spec_file(self) -> str:
        return f"{self.input_dir}/bias.yml"

    @property
    def task_note_file(self) -> str:
        return f"{self.input_dir}/task_note.txt"

    @property
    def starter_answers_file(self) -> str:
        return f"{self.input_dir}/audit_answers.json"

    @property
    def starter_memo_file(self) -> str:
        return f"{self.input_dir}/audit_memo.md"

    @property
    def data_file(self) -> str:
        return f"{self.input_dir}/data/data_new.csv"

    @property
    def data_dictionary_file(self) -> str:
        return f"{self.input_dir}/data/data_dictionary.md"

    @property
    def code_root(self) -> str:
        return f"{self.input_dir}/code"

    @property
    def model_script(self) -> str:
        return f"{self.code_root}/model/main.py"

    @property
    def table2_script(self) -> str:
        return f"{self.code_root}/table2.py"

    @property
    def figure1b_script(self) -> str:
        return f"{self.code_root}/figure1/figure1b.R"

    @property
    def table3_script(self) -> str:
        return f"{self.code_root}/table3.R"

    @property
    def task_python_wrapper(self) -> str:
        return f"{self.software_dir}/task_python"

    @property
    def task_rscript_wrapper(self) -> str:
        return f"{self.software_dir}/task_rscript"

    @property
    def results_dir(self) -> str:
        return f"{self.remote_output_dir}/results"

    @property
    def answers_output_file(self) -> str:
        return f"{self.remote_output_dir}/audit_answers.json"

    @property
    def memo_output_file(self) -> str:
        return f"{self.remote_output_dir}/audit_memo.md"

    @property
    def reference_answers_file(self) -> str:
        return f"{self.reference_dir}/audit_answers.json"

    @property
    def reference_memo_file(self) -> str:
        return f"{self.reference_dir}/audit_memo.md"

    @property
    def reference_results_dir(self) -> str:
        return f"{self.reference_dir}/results"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to reproduce a public synthetic healthcare bias audit.

Read these staged task materials first:
- `{self.task_note_file}`
- `{self.env_spec_file}`
- `{self.starter_answers_file}`
- `{self.starter_memo_file}`

`{self.env_spec_file}` lists the package versions used in the original public bundle.
Set up a working Python and R environment that satisfies those version constraints (or functionally equivalent versions) before running the scripts.

Inspect the provided workflow scripts before you run anything:
- `{self.model_script}`
- `{self.table2_script}`
- `{self.figure1b_script}`
- `{self.table3_script}`

Visible data files:
- `{self.data_file}`
- `{self.data_dictionary_file}`

Your job is to:
1. Create a writable working copy of the staged bundle under `{self.remote_output_dir}` (or another writable subdirectory inside it) so the supplied scripts can write `results/` without mutating the staged `input/` tree.
2. Run the provided Python and R scripts in the dependency order needed to generate the five required CSV outputs.
3. Fill in `audit_answers.json` using only the public synthetic replication outputs from this run.
4. Write `audit_memo.md` for a clinical analytics lead, grounding it in this run's outputs and clearly distinguishing the public synthetic replication from the paper's private-data values.
5. Save exactly these deliverables under `{self.remote_output_dir}`:
   - `results/model_lasso_predictors.csv`
   - `results/model_r2.csv`
   - `results/table2_concentration_metric.csv`
   - `results/figure1b.csv`
   - `results/table3.csv`
   - `audit_answers.json`
   - `audit_memo.md`

Rules:
- Do not modify the staged dataset.
- Do not add race as a model feature.
- Do not replace the supplied workflow with a different analytical pipeline.
- Do not use the original paper's `17.7% -> 46.5%` value or the later `59%` correction as the graded answer.
- Treat the staged solve-time files as read-only and do not rely on any hidden evaluator-only data.
- If you create scratch files under `{self.remote_output_dir}`, clean them up before finishing so only the required deliverables remain.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "env_spec_file": self.env_spec_file,
                "task_note_file": self.task_note_file,
                "starter_answers_file": self.starter_answers_file,
                "starter_memo_file": self.starter_memo_file,
                "data_file": self.data_file,
                "data_dictionary_file": self.data_dictionary_file,
                "code_root": self.code_root,
                "model_script": self.model_script,
                "table2_script": self.table2_script,
                "figure1b_script": self.figure1b_script,
                "table3_script": self.table3_script,
                "task_python_wrapper": self.task_python_wrapper,
                "task_rscript_wrapper": self.task_rscript_wrapper,
                "results_dir": self.results_dir,
                "answers_output_file": self.answers_output_file,
                "memo_output_file": self.memo_output_file,
                "reference_answers_file": self.reference_answers_file,
                "reference_memo_file": self.reference_memo_file,
                "reference_results_dir": self.reference_results_dir,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = HealthcareBiasAuditConfig()


@cb.tasks_config(split="train")
def load():
    cfg = HealthcareBiasAuditConfig()
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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    expected_files = set(REQUIRED_OUTPUT_FILES)
    observed_files = set(await _list_relative_files(session, meta["remote_output_dir"]))
    if observed_files != expected_files:
        logger.error("Unexpected candidate output file set: %s", sorted(observed_files))
        return [0.0]

    reference_paths = [
        meta["reference_answers_file"],
        meta["reference_memo_file"],
        *[f'{meta["reference_dir"]}/{name}' for name in REQUIRED_OUTPUT_FILES if name.startswith("results/")],
    ]
    missing_reference = [path for path in reference_paths if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing_reference:
        logger.error("Missing reference evaluation paths: %s", missing_reference)
        return [0.0]

    try:
        candidate_files = {}
        reference_files = {}
        for rel_path in REQUIRED_OUTPUT_FILES:
            candidate_path = f'{meta["remote_output_dir"]}/{rel_path}'
            reference_path = (
                meta["reference_dir"] + "/" + rel_path
            )
            candidate_files[rel_path] = _as_text(await session.read_file(candidate_path))
            reference_files[rel_path] = _as_text(await session.read_file(reference_path))
    except Exception as exc:
        logger.error("Failed to read task outputs/reference files: %s", exc)
        return [0.0]

    result = score_output_bundle(candidate_files=candidate_files, reference_files=reference_files)
    _log_score(result)
    return [float(result.score)]
