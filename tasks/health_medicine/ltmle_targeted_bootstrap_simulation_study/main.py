"""AgentHLE task: ltmle_targeted_bootstrap_simulation_study."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import shlex
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Optional

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

from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study.scripts.score_outputs import (  # noqa: E402
    ScoreResult,
    compare_summary_csv,
    validate_public_raw,
)
from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study.scripts.verify_hidden_smoke import (  # noqa: E402
    CONTRACT_VERSION,
)

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "ltmle_targeted_bootstrap_simulation_study"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
PREFERRED_EVAL_TMP_ROOT = f"/tmp/agenthle_eval/{TASK_NAME}"
RSCRIPT_BINARY = "/usr/bin/Rscript"
PINNED_RSCRIPT_BINARY = "/opt/R/4.3.2/bin/Rscript"
REQUIRED_SCRIPT_NAMES = [
    "02_variance_methods_longitudinal.R",
    "03_simulation_runner_longitudinal.R",
    "04_analysis_functions_longitudinal.R",
    "05_run_full_simulation_longitudinal.R",
    "06_analyze_part2_results_longitudinal.R",
    "06b_analyze_by_sample_size.R",
]


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _output_dir_name(remote_output_dir: str) -> str:
    return PurePosixPath(remote_output_dir).name


def _is_fixture_dir_name(output_dir_name: str) -> bool:
    return output_dir_name in {"output_test_pos", "output_test_neg"}


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


async def _ensure_remote_dir_with_fallback(
    session: cb.DesktopSession,
    *,
    preferred_dir: str,
    fallback_dir: str,
    label: str,
) -> str:
    preferred_result = await _run_command(
        session,
        _shell_join(["mkdir", "-p", preferred_dir]),
        check=False,
    )
    if preferred_result.get("return_code") == 0:
        return preferred_dir

    fallback_result = await _run_command(
        session,
        _shell_join(["mkdir", "-p", fallback_dir]),
        check=False,
    )
    if fallback_result.get("return_code") != 0:
        raise RuntimeError(
            f"could_not_create_{label}: preferred={preferred_dir} fallback={fallback_dir} "
            f"preferred_stderr={preferred_result.get('stderr', '')[:400]} "
            f"fallback_stderr={fallback_result.get('stderr', '')[:400]}"
        )
    logger.warning(
        "using fallback %s dir %s after preferred dir %s failed: %s",
        label,
        fallback_dir,
        preferred_dir,
        preferred_result.get("stderr", "")[:400],
    )
    return fallback_dir


async def _select_rscript(
    session: cb.DesktopSession, *, software_rscript: str, required_version: str
) -> str:
    probes = []
    for binary in dict.fromkeys([PINNED_RSCRIPT_BINARY, software_rscript, RSCRIPT_BINARY]):
        result = await _run_command(
            session,
            _shell_join(["timeout", "30", binary, "-e", "cat(as.character(getRversion()))"]),
            timeout=35.0,
            check=False,
        )
        version = _as_text(result.get("stdout", "")).strip()
        if result.get("return_code") == 0 and version == required_version:
            logger.info("LTMLE evaluator runtime: %s (R %s)", binary, version)
            return binary
        probes.append(
            {"binary": binary, "return_code": result.get("return_code"), "version": version[:120]}
        )
    raise RuntimeError(f"LTMLE evaluator pinned R {required_version} unavailable: {probes}")


def _log_score(label: str, result: ScoreResult) -> None:
    logger.info(
        "%s score=%.3f passed=%s reason=%s details=%s",
        label,
        result.score,
        result.passed,
        result.reason,
        json.dumps(result.details, ensure_ascii=True, sort_keys=True),
    )


class LtmleTargetedBootstrapConfig(LinuxTaskConfig):
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
    def task_input_root(self) -> str:
        return f"{self.input_dir}/LTMLE_Targeted_Bootstrap_Task_INPUT"

    @property
    def dgp_source_file(self) -> str:
        return f"{self.task_input_root}/01_data_generation_longitudinal.R"

    @property
    def lmtp_source_dir(self) -> str:
        return f"{self.task_input_root}/lmtp-bootstrap"

    @property
    def theory_paper_file(self) -> str:
        return f"{self.task_input_root}/tmleVariance.pdf"

    @property
    def public_benchmark_dir(self) -> str:
        return f"{self.input_dir}/public_benchmark"

    @property
    def public_note_file(self) -> str:
        return f"{self.public_benchmark_dir}/README_output.md"

    @property
    def public_summary_file(self) -> str:
        return f"{self.public_benchmark_dir}/reference_summary.csv"

    @property
    def public_study_plan_file(self) -> str:
        return f"{self.public_benchmark_dir}/study_plan_public.json"

    @property
    def reference_expected_summary_file(self) -> str:
        return f"{self.reference_dir}/expected_summary.csv"

    @property
    def evaluation_contract_file(self) -> str:
        return f"{self.reference_dir}/evaluation_contract.json"

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def output_summary_file(self) -> str:
        return f"{self.remote_output_dir}/summary.csv"

    @property
    def output_report_file(self) -> str:
        return f"{self.remote_output_dir}/report.pdf"

    @property
    def output_dir_name(self) -> str:
        return _output_dir_name(self.remote_output_dir)

    @property
    def software_rscript(self) -> str:
        return f"{self.software_dir}/Rscript"

    @property
    def eval_dir(self) -> str:
        # Task-local evaluator scratch root. Not part of the canonical staged
        # tree; used here as the fallback location for eval_tmp + hidden_smoke
        # workspaces when PREFERRED_EVAL_TMP_ROOT is unavailable.
        return f"{self.task_dir}/eval_data"

    @property
    def preferred_eval_tmp_dir(self) -> str:
        suffix = self.output_dir_name.replace("/", "_")
        return f"{PREFERRED_EVAL_TMP_ROOT}/{suffix}"

    @property
    def fallback_eval_tmp_dir(self) -> str:
        suffix = self.output_dir_name.replace("/", "_")
        return f"{self.eval_dir}/_eval_tmp/{suffix}"

    @property
    def preferred_hidden_smoke_eval_dir(self) -> str:
        suffix = self.output_dir_name.replace("/", "_")
        return f"{PREFERRED_EVAL_TMP_ROOT}/hidden_smoke/{suffix}"

    @property
    def fallback_hidden_smoke_eval_dir(self) -> str:
        suffix = self.output_dir_name.replace("/", "_")
        return f"{self.eval_dir}/hidden_smoke/{suffix}"

    @property
    def task_description(self) -> str:
        required_script_lines = "\n".join(
            f"- `{self.task_dir}/output/{script_name}`" for script_name in REQUIRED_SCRIPT_NAMES
        )
        return f"""\
You are working on a Linux R biostatistics simulation benchmark.

Read these staged inputs first:
- `{self.dgp_source_file}`
- `{self.lmtp_source_dir}`
- `{self.theory_paper_file}`
- `{self.public_note_file}`
- `{self.public_summary_file}`
- `{self.public_study_plan_file}`
- `{self.public_benchmark_dir}/study_plan_public.csv`
- `{self.public_benchmark_dir}/SCIENTIFIC_CONTRACT.md`
- `{self.public_benchmark_dir}/runtime_manifest.json`

Your job is to reconstruct the missing longitudinal simulation pipeline and write the required deliverables under `{self.task_dir}/output`.

Runtime:
Use `{PINNED_RSCRIPT_BINARY}` (R 4.3.2) for every R session, including analysis. Do not use unqualified `Rscript` or `/usr/bin/Rscript`.
Use the installed task-controlled package closure in `input/public_benchmark/runtime_manifest.json`; do not install replacement CRAN/GitHub packages.
From `output/`, run `{PINNED_RSCRIPT_BINARY} 05_run_full_simulation_longitudinal.R`.

Required pipeline files:
{required_script_lines}

Required generated outputs:
- `{self.task_dir}/output/summary.csv`
- `{self.task_dir}/output/report.pdf`
- `{self.task_dir}/output/raw_results.csv`

Requirements:
1. Use the bundled longitudinal DGP, the local `lmtp-bootstrap` source tree, the variance paper, and the public benchmark materials under `input/public_benchmark/`.
2. Cover the two public benchmark scenarios from `study_plan_public.json` and generate your own `summary.csv` plus `report.pdf` from a working pipeline.
3. The public benchmark summary is visible for orientation, but do not treat it as a file to relocate or copy into `output/`.
4. Follow scientific contract v2 (2026-09-10), not the superseded April numbers. Standard uses paired EIF variance. Full-data targeting uses cumulative density ratios; its point is shared by both targeted-bootstrap methods. Do not use the bundled raw-ratio `boot=TRUE` shortcut.
5. `05` must honor a local `fixture_smoke_plan.csv` in place of the public plan. The evaluator copies only your six R scripts, stages sibling input/, then executes `05` and the two documented analysis functions in separate R sessions. Exact APIs, seed schedule, schemas, numerical boundary rules, and tolerances are in `SCIENTIFIC_CONTRACT.md`.
6. Generate raw results, derive summary from them, and discuss the study and numerical diagnostics in your report. No dropped replications or bootstrap draws. Write solver-created files only under `{self.task_dir}/output`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "task_input_root": self.task_input_root,
                "dgp_source_file": self.dgp_source_file,
                "lmtp_source_dir": self.lmtp_source_dir,
                "theory_paper_file": self.theory_paper_file,
                "public_benchmark_dir": self.public_benchmark_dir,
                "public_note_file": self.public_note_file,
                "public_summary_file": self.public_summary_file,
                "public_study_plan_file": self.public_study_plan_file,
                "reference_expected_summary_file": self.reference_expected_summary_file,
                "evaluation_contract_file": self.evaluation_contract_file,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_summary_file": self.output_summary_file,
                "output_report_file": self.output_report_file,
                "output_dir_name": self.output_dir_name,
                "software_rscript": self.software_rscript,
                "preferred_eval_tmp_dir": self.preferred_eval_tmp_dir,
                "fallback_eval_tmp_dir": self.fallback_eval_tmp_dir,
                "preferred_hidden_smoke_eval_dir": self.preferred_hidden_smoke_eval_dir,
                "fallback_hidden_smoke_eval_dir": self.fallback_hidden_smoke_eval_dir,
                "required_scripts": REQUIRED_SCRIPT_NAMES,
            }
        )
        return metadata


config = LtmleTargetedBootstrapConfig()


@cb.tasks_config(split="train")
def load():
    cfg = LtmleTargetedBootstrapConfig()
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
    output_dir_name = meta["output_dir_name"]
    fixture_mode = _is_fixture_dir_name(output_dir_name)

    required_eval_paths = [
        meta["reference_dir"],
        meta["reference_expected_summary_file"],
        meta["evaluation_contract_file"],
        *(
            f"{meta['reference_dir']}/{name}"
            for name in (
                "raw_results.csv",
                "public_raw_results.csv",
                "summary.csv",
                "fixture_smoke_plan.csv",
            )
        ),
    ]
    missing_eval = [
        path
        for path in required_eval_paths
        if not (await session.file_exists(path) or await session.directory_exists(path))
    ]
    if missing_eval:
        raise RuntimeError(f"LTMLE evaluator files are missing: {missing_eval}")

    required_candidate_paths = [
        meta["output_summary_file"],
        meta["output_report_file"],
        f"{meta['remote_output_dir']}/raw_results.csv",
        *(f"{meta['remote_output_dir']}/{name}" for name in meta["required_scripts"]),
    ]
    missing_candidate = [
        path
        for path in required_candidate_paths
        if not (await session.file_exists(path) or await session.directory_exists(path))
    ]
    if missing_candidate:
        logger.error("missing candidate output paths: %s", missing_candidate)
        return [0.0]

    contract = json.loads(_as_text(await session.read_file(meta["evaluation_contract_file"])))
    if contract.get("scientific_contract_version") != CONTRACT_VERSION:
        raise RuntimeError("LTMLE scientific contract v2 data is not staged")

    script_bodies = {
        script_name: _as_text(await session.read_file(f"{meta['remote_output_dir']}/{script_name}"))
        for script_name in meta["required_scripts"]
    }
    empty_scripts = [name for name, text in script_bodies.items() if not text.strip()]
    if empty_scripts:
        logger.error("empty candidate scripts: %s", empty_scripts)
        return [0.0]

    summary_text = _as_text(await session.read_file(meta["output_summary_file"]))
    if not summary_text.strip():
        logger.error("candidate summary.csv is empty")
        return [0.0]

    report_check = await _run_command(
        session,
        _shell_join(["bash", "-lc", f"test -s {shlex.quote(meta['output_report_file'])}"]),
        check=False,
    )
    if report_check.get("return_code") != 0:
        logger.error("candidate report.pdf is missing or empty: %s", meta["output_report_file"])
        return [0.0]

    if not fixture_mode:
        reference_summary_text = _as_text(
            await session.read_file(meta["reference_expected_summary_file"])
        )
        if (
            hashlib.sha256(reference_summary_text.encode()).hexdigest()
            != contract["public_benchmark"]["reference_sha256"]
        ):
            raise RuntimeError("LTMLE public reference hash mismatch")
        raw_result = validate_public_raw(
            _as_text(await session.read_file(f"{meta['remote_output_dir']}/raw_results.csv")),
            summary_text,
            contract,
        )
        _log_score("public_raw", raw_result)
        if not raw_result.passed:
            return [0.0]
        public_result = compare_summary_csv(
            candidate_summary_csv=summary_text,
            expected_summary_csv=reference_summary_text,
            contract_section=contract["public_benchmark"],
            label="public_summary",
        )
        _log_score("public_summary", public_result)
        if not public_result.passed:
            return [0.0]

    eval_tmp_dir = await _ensure_remote_dir_with_fallback(
        session,
        preferred_dir=meta["preferred_eval_tmp_dir"],
        fallback_dir=meta["fallback_eval_tmp_dir"],
        label="eval_tmp",
    )
    hidden_smoke_eval_dir = await _ensure_remote_dir_with_fallback(
        session,
        preferred_dir=meta["preferred_hidden_smoke_eval_dir"],
        fallback_dir=meta["fallback_hidden_smoke_eval_dir"],
        label="hidden_smoke_eval",
    )
    verify_script_path = f"{eval_tmp_dir}/verify_hidden_smoke.py"
    await session.write_file(verify_script_path, _read_script("verify_hidden_smoke.py"))

    rscript_binary = await _select_rscript(
        session,
        software_rscript=meta["software_rscript"],
        required_version=contract["hidden_smoke"]["r_version"],
    )

    verify_command = _shell_join(
        [
            "python",
            verify_script_path,
            "--candidate-dir",
            meta["remote_output_dir"],
            "--input-dir",
            meta["input_dir"],
            "--reference-dir",
            meta["reference_dir"],
            # The hidden-smoke verifier only reads fixture_smoke_plan.csv from
            # this dir. That plan is evaluator config and lives in reference/
            # (which IS staged to the eval VM); output_test_pos/ is a self-test
            # fixture and is never staged, so do not depend on it at eval time.
            "--positive-fixture-dir",
            meta["reference_dir"],
            "--eval-data-dir",
            hidden_smoke_eval_dir,
            "--rscript-binary",
            rscript_binary,
        ]
    )
    hidden_result = await _run_command(session, verify_command, timeout=2400.0, check=False)
    stdout = hidden_result.get("stdout", "").strip()
    if not stdout:
        raise RuntimeError(
            f"LTMLE verifier did not emit JSON: rc={hidden_result.get('return_code')} "
            f"stderr={hidden_result.get('stderr', '')[:1200]}"
        )

    try:
        hidden_payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LTMLE verifier emitted invalid JSON: {stdout[:1200]!r}") from exc

    if hidden_result.get("return_code") == 2 or hidden_payload.get("error_type") == "evaluator":
        raise RuntimeError(f"LTMLE evaluator failed: {hidden_payload}")

    hidden_passed = bool(hidden_payload.get("passed"))
    logger.info("hidden_smoke=%s", json.dumps(hidden_payload, ensure_ascii=True, sort_keys=True))
    return [1.0 if hidden_passed else 0.0]
