"""AgentHLE task: prostate IMRT matRad reproduction (base)."""

import asyncio
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "prostate_imrt_matrad_reproduction"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"
MICROMAMBA_ENV_NAME = "rtplan-matrad"
PASS_THRESHOLD = 70


def _canonical_output_dir_name(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "OUTPUT_SUBDIR must be one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


class ProstateIMRTConfig(LinuxTaskConfig):
    def __init__(self, output_subdir: str = "output") -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
        )
        self.OUTPUT_SUBDIR = output_subdir

    @property
    def dicom_dir(self) -> str:
        return f"{self.input_dir}/dicom"

    @property
    def corrupted_rtstruct_file(self) -> str:
        return f"{self.dicom_dir}/RTSTRUCT.dcm"

    @property
    def prescription_file(self) -> str:
        return f"{self.input_dir}/prescription.json"

    @property
    def beam_geometry_file(self) -> str:
        return f"{self.input_dir}/beam_geometry.json"

    @property
    def institutional_constraints_file(self) -> str:
        return f"{self.input_dir}/institutional_constraints.md"

    @property
    def beam_model_file(self) -> str:
        return f"{self.input_dir}/machine/benchmark_6MV.mat"

    @property
    def pln_template_file(self) -> str:
        return f"{self.input_dir}/pln_template.mat"

    @property
    def manifest_file(self) -> str:
        return f"{self.input_dir}/manifest.json"

    @property
    def readme_file(self) -> str:
        return f"{self.input_dir}/README_AGENT.md"

    @property
    def submission_template_dir(self) -> str:
        return f"{self.input_dir}/submission_template"

    @property
    def matrad_wrapper(self) -> str:
        return f"{self.software_dir}/run_matrad.sh"

    @property
    def output_dir(self) -> str:
        output_dir_name = _canonical_output_dir_name(self.OUTPUT_SUBDIR)
        return f"{self.task_dir}/{output_dir_name}"

    @property
    def task_description(self) -> str:
        return f"""You are a senior medical physicist performing an independent research replanning benchmark on a fixed-field prostate IMRT reference plan using matRad (open-source treatment planning) and GNU Octave on Linux.

Task directory:
- `{self.task_dir}`

Visible inputs (do not modify):
- CT DICOM series: `{self.dicom_dir}/CT`
- Corrupted RTSTRUCT (contains 3 planted defects): `{self.corrupted_rtstruct_file}`
- Prescription: `{self.prescription_file}`
- Institutional DVH constraints and margin recipe: `{self.institutional_constraints_file}`
- Beam geometry (7 gantry angles): `{self.beam_geometry_file}`
- matRad beam model: `{self.beam_model_file}`
- matRad plan template: `{self.pln_template_file}`
- Agent-facing README: `{self.readme_file}`
- Repair recipe notes: `{self.input_dir}/repair_recipes.md`
- Submission output template: `{self.submission_template_dir}`
- File hashes: `{self.manifest_file}`
- matRad/Octave launcher (preinstalled): `{self.matrad_wrapper}`

Your task (six phases):
1. Structure QA: identify and fix three defects planted in the supplied RTSTRUCT (clinical-rule corrections described in `{self.input_dir}/repair_recipes.md`).
2. Import into matRad using the shipped `{self.pln_template_file}` and `{self.beam_model_file}` (7 coplanar 6 MV beams, Engel sequencing 7 levels, RNG seed 42).
3. Fluence optimization with Engel leaf sequencing. Naive weights violate the rectum V70Gy constraint; expect 3-6 weight-tuning iterations.
4. Deterministic pencil-beam (SVDPB) dose calculation at 3 mm grid.
5. DICOM-RT export of RTPLAN.dcm, RTDOSE.dcm, and corrected RTSTRUCT, all sharing a consistent FrameOfReferenceUID.
6. Independent QA in Python: recompute DVH from the submitted RTDOSE + RTSTRUCT, render axial/sagittal/coronal isodose PNGs, and produce `report.md`, `plan_metrics.json`, `beam_summary.csv`, `decisions.md`.

Required outputs, written only under `{self.output_dir}`:
- `RTPLAN.dcm`, `RTDOSE.dcm`, `RTSTRUCT_corrected.dcm`
- `replay_state.mat` (numeric-only matRad state for independent replay)
- `dvh_metrics.csv`, `plan_metrics.json`, `beam_summary.csv`
- `figures/axial.png`, `figures/sagittal.png`, `figures/coronal.png` (each >= 800x800)
- `report.md`, `decisions.md`

Runtime guidance:
- Invoke matRad / Octave via `{self.matrad_wrapper}`; the wrapper activates the pinned `rtplan-matrad` environment (GNU Octave 6.4.0, matRad commit `c014dc82`, Python 3.10 with pydicom / pymedphys / numba / numpy / scipy / matplotlib / scikit-image).
- No network access is required after the environment is installed.
- Do not modify files under `input/`. Write the final bundle only under `{self.output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "dicom_dir": self.dicom_dir,
                "corrupted_rtstruct_file": self.corrupted_rtstruct_file,
                "prescription_file": self.prescription_file,
                "beam_geometry_file": self.beam_geometry_file,
                "institutional_constraints_file": self.institutional_constraints_file,
                "beam_model_file": self.beam_model_file,
                "pln_template_file": self.pln_template_file,
                "manifest_file": self.manifest_file,
                "readme_file": self.readme_file,
                "submission_template_dir": self.submission_template_dir,
                "matrad_wrapper": self.matrad_wrapper,
                "output_dir_name": _canonical_output_dir_name(self.OUTPUT_SUBDIR),
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
                "eval_tmp_dir": EVAL_TMP_DIR,
                "micromamba_env": MICROMAMBA_ENV_NAME,
                "pass_threshold": PASS_THRESHOLD,
            }
        )
        return metadata


config = ProstateIMRTConfig(output_subdir=os.environ.get("OUTPUT_SUBDIR", "output"))


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    sub_dir = meta["output_dir"]
    ref_dir = meta["reference_dir"]
    env_name = meta["micromamba_env"]
    tmp_dir = meta["eval_tmp_dir"]

    required_after = [
        f"{sub_dir}/RTPLAN.dcm",
        f"{sub_dir}/RTDOSE.dcm",
        f"{sub_dir}/RTSTRUCT_corrected.dcm",
    ]
    missing = [p for p in required_after if not (await session.file_exists(p) or await session.directory_exists(p))]
    if missing:
        logger.error("[%s] missing submission outputs: %s", TASK_NAME, missing)
        return [0.0]

    await session.interface.create_dir(tmp_dir)
    await session.write_file(f"{tmp_dir}/evaluate.py", _read_script("evaluate.py"))
    await session.write_file(f"{tmp_dir}/replay.m", _read_script("replay.m"))

    await _run_command(
        session,
        f"pkill -f {shlex.quote(tmp_dir + '/evaluate.py')} || true",
        check=False,
    )

    score_path = f"{tmp_dir}/score.json"
    done_flag = f"{tmp_dir}/.score_done"
    log_path = f"{tmp_dir}/evaluate.log"
    rc_path = f"{tmp_dir}/.score_rc"
    await _run_command(
        session,
        f'rm -f {shlex.quote(score_path)} {shlex.quote(done_flag)} '
        f'{shlex.quote(log_path)} {shlex.quote(rc_path)}',
        check=False,
    )

    inner = (
        "export MAMBA_ROOT_PREFIX=$HOME/.local/share/micromamba && "
        f"$HOME/.local/bin/micromamba run -n {shlex.quote(env_name)} "
        f"python {shlex.quote(tmp_dir + '/evaluate.py')} "
        f"--submission {shlex.quote(sub_dir)} "
        f"--reference {shlex.quote(ref_dir)} "
        f"--out {shlex.quote(score_path)}"
    )
    runner_script = f"{tmp_dir}/run_evaluate.sh"
    script_body = (
        "#!/usr/bin/env bash\n"
        f"{inner}\n"
        f"echo $? > {shlex.quote(rc_path)}\n"
        f"touch {shlex.quote(done_flag)}\n"
    )
    await session.write_file(runner_script, script_body)
    await _run_command(
        session,
        f"bash -lc {shlex.quote('chmod +x ' + shlex.quote(runner_script))}",
        check=False,
    )
    # Double-fork + explicit fd closure via python to fully detach from the cua
    # server's socket fds (setsid alone isn't enough because inherited non-0/1/2
    # fds keep the HTTP response connection open).
    daemonize_py = (
        "import os,resource\n"
        "if os.fork():os._exit(0)\n"
        "os.setsid()\n"
        "if os.fork():os._exit(0)\n"
        "maxfd=resource.getrlimit(resource.RLIMIT_NOFILE)[1]\n"
        "if maxfd==resource.RLIM_INFINITY:maxfd=1024\n"
        "os.closerange(3,maxfd)\n"
        f"fd=os.open({log_path!r},os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o644)\n"
        "os.dup2(fd,1);os.dup2(fd,2)\n"
        "if fd>2:os.close(fd)\n"
        "fd=os.open('/dev/null',os.O_RDONLY)\n"
        "os.dup2(fd,0)\n"
        "if fd>0:os.close(fd)\n"
        f"os.execvp('bash',['bash',{runner_script!r}])\n"
    )
    launcher = (
        "python3 - <<'DAEMONIZE_PY'\n"
        + daemonize_py
        + "DAEMONIZE_PY\n"
    )
    await _run_command(
        session,
        f"bash -lc {shlex.quote(launcher)}",
        check=False,
    )

    poll_attempts = int(meta.get("eval_poll_attempts", 240))
    poll_interval = float(meta.get("eval_poll_interval_sec", 15.0))
    for _ in range(poll_attempts):
        if (await session.file_exists(done_flag) or await session.directory_exists(done_flag)):
            break
        await asyncio.sleep(poll_interval)
    else:
        logger.error("[%s] evaluate.py did not finish within poll window", TASK_NAME)
        return [0.0]

    rc_text = ""
    if (await session.file_exists(rc_path) or await session.directory_exists(rc_path)):
        rc_text = _as_text(await session.read_file(rc_path)).strip()

    # Always try to read score.json — even on non-zero exit code the
    # evaluator may have written partial results before crashing.
    payload = None
    if (await session.file_exists(score_path) or await session.directory_exists(score_path)):
        try:
            payload = json.loads(_as_text(await session.read_file(score_path)))
        except Exception as exc:
            logger.error("[%s] could not parse score.json: %s", TASK_NAME, exc)

    if rc_text != "0":
        log_tail = ""
        if (await session.file_exists(log_path) or await session.directory_exists(log_path)):
            log_tail = _as_text(await session.read_file(log_path))[-2000:]
        logger.error(
            "[%s] evaluate.py failed rc=%s log_tail=%s partial_score=%s",
            TASK_NAME,
            rc_text or "<missing>",
            log_tail,
            json.dumps(payload) if payload else "<none>",
        )
        return [0.0]

    if payload is None:
        logger.error("[%s] score.json not produced", TASK_NAME)
        return [0.0]

    total = int(payload.get("total_score", 0))
    rubric_pass = bool(payload.get("pass", False))
    threshold = int(meta.get("pass_threshold", PASS_THRESHOLD))
    passed = rubric_pass and total >= threshold

    gates = payload.get("gates", {})
    breakdown = ", ".join(
        f"{gate}={info.get('score', 'OK' if info.get('ok') else 'FAIL')}/{info.get('max', '-')}"
        for gate, info in gates.items()
    )
    logger.info(
        "[%s] total=%s pass=%s threshold=%s breakdown=%s",
        TASK_NAME,
        total,
        passed,
        threshold,
        breakdown,
    )
    logger.info("[%s] score_detail=%s", TASK_NAME, json.dumps(payload))

    return [1.0 if passed else 0.0]


if __name__ == "__main__":
    for task in load():
        print(task.description)
