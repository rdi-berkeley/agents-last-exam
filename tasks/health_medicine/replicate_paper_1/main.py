"""AgentHLE task: replicate_paper_1."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback

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

DOMAIN_NAME = "health_medicine"
TASK_NAME = "replicate_paper_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_DIR = "/tmp/agenthle_eval/replicate_paper_1"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(session: cb.DesktopSession, command: str, *, timeout: float | None = None) -> dict:
    try:
        if timeout is None:
            return await session.run_command(command, check=False)
        return await session.run_command(command, timeout=timeout, check=False)
    except TypeError:
        return await session.run_command(command, check=False)


class ReplicatePaperConfig(LinuxTaskConfig):
    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
        )

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def output_dir(self) -> str:
        if self.OUTPUT_SUBDIR == "output_test_pos":
            return self.output_test_pos_dir
        if self.OUTPUT_SUBDIR == "output_test_neg":
            return self.output_test_neg_dir
        return f"{self.task_dir}/{self.OUTPUT_SUBDIR}"

    @property
    def simulation_spec_pdf(self) -> str:
        return f"{self.input_dir}/simulation_spec.pdf"

    @property
    def simulation_spec_txt(self) -> str:
        return f"{self.input_dir}/simulation_spec.txt"

    @property
    def task_instructions(self) -> str:
        return f"{self.input_dir}/task_instructions.md"

    @property
    def output_schema(self) -> str:
        return f"{self.input_dir}/output_schema.json"

    @property
    def summary_results(self) -> str:
        return f"{self.output_dir}/summary_results.csv"

    @property
    def coverage_results(self) -> str:
        return f"{self.output_dir}/coverage_results.csv"

    @property
    def findings_file(self) -> str:
        return f"{self.output_dir}/summary_findings.txt"

    @property
    def metadata_file(self) -> str:
        return f"{self.output_dir}/run_metadata.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux R biostatistics simulation benchmark.

Task root:
- `{self.task_dir}`

Read the staged task specification and schema:
- `{self.simulation_spec_pdf}`
- `{self.simulation_spec_txt}`
- `{self.task_instructions}`
- `{self.output_schema}`

Your job is to reproduce the continuous-treatment shift-intervention simulation study from the specification.

Required work:
1. Estimate the large-sample Monte Carlo truth for `psi0 = E[Q(A + 2, W)]` using at least 1,000,000 draws.
2. Implement IPTW, A-IPTW, and TMLE estimators for the shift intervention.
3. Run at least 1000 repetitions for each sample size: 50, 100, 200, and 500.
4. Summarize mean estimate, bias, variance, Monte Carlo SE, 95% interval, and coverage for every estimator/sample-size pair.
5. Write all deliverables only under `{self.output_dir}`.

Required outputs:
- `{self.summary_results}`
- `{self.coverage_results}`
- `{self.output_dir}/comparison_plots/` with at least one non-empty `.png` or `.pdf`
- `{self.findings_file}`
- `{self.metadata_file}`

Software guidance:
- Use the provisioned R entry point `{self.software_dir}/Rscript`.
- You may write temporary simulation files inside `{self.output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_dir": self.output_dir,
                "remote_output_label": self.OUTPUT_SUBDIR,
                "simulation_spec_pdf": self.simulation_spec_pdf,
                "simulation_spec_txt": self.simulation_spec_txt,
                "task_instructions": self.task_instructions,
                "output_schema": self.output_schema,
                "summary_results": self.summary_results,
                "coverage_results": self.coverage_results,
                "findings_file": self.findings_file,
                "metadata_file": self.metadata_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = ReplicatePaperConfig()


@cb.tasks_config(split="train")
def load():
    cfg = ReplicatePaperConfig()
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
    if not (await session.file_exists(meta["reference_dir"]) or await session.directory_exists(meta["reference_dir"])):
        logger.error("missing evaluator reference directory: %s", meta["reference_dir"])
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    verifier_path = f"{EVAL_TMP_DIR}/score_outputs.py"
    await session.write_file(verifier_path, _read_script("score_outputs.py"))

    allow_fixture = " --allow-fixture-copy" if meta.get("remote_output_label") == "output_test_pos" else ""
    result = await _run_command(
        session,
        (
            f'python "{verifier_path}" '
            f'--output-dir "{meta["output_dir"]}" '
            f'--reference-dir "{meta["reference_dir"]}"'
            f"{allow_fixture}"
        ),
        timeout=300.0,
    )
    stdout = result.get("stdout", "")
    if result.get("return_code") != 0 and not stdout.strip():
        logger.error("verifier failed before JSON output: %s", result.get("stderr", "")[:800])
        return [0.0]
    try:
        payload = json.loads(stdout)
    except Exception:
        logger.error("could not parse verifier JSON: stdout=%r stderr=%r", stdout[:800], result.get("stderr", "")[:800])
        return [0.0]
    logger.info("replicate_paper_1 score=%s reasons=%s", payload.get("score"), payload.get("reasons"))
    return [float(payload.get("score", 0.0))]
