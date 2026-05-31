"""AgentHLE task: psychology_neuro/celegans_neuron_tracking."""

from __future__ import annotations

import json
import logging
import posixpath
import shlex
import textwrap
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "psychology_neuro"
TASK_NAME = "celegans_neuron_tracking"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAMES = ["137", "153", "184", "185"]
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}
# `agenthle-ubuntu` currently has a full root disk, so evaluator scratch must
# live on the data disk instead of `/tmp`.
EVAL_SCRATCH_ROOT = f"/media/user/data/ale-data/.tmp_eval/{TASK_NAME}"


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "OUTPUT_SUBDIR must normalize to one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


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


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"stdout does not contain a JSON object: {stdout!r}")
    return json.loads(text[start : end + 1])


def _uv_runtime_prefix(eval_root: str) -> str:
    return (
        f"export UV_CACHE_DIR={shlex.quote(f'{eval_root}/.uv-cache')} "
        f"UV_PROJECT_ENVIRONMENT={shlex.quote(f'{eval_root}/.venv')} && "
    )


def _build_contract_check_command(meta: dict[str, Any], eval_root: str) -> str:
    script = textwrap.dedent("""
        import json
        import sys

        import h5py
        import numpy as np

        pred_path = sys.argv[1]
        src_path = sys.argv[2]

        def normalize_attr(value):
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, list):
                return [normalize_attr(item) for item in value]
            return value

        def attrs_dict(obj):
            return {key: normalize_attr(obj.attrs[key]) for key in sorted(obj.attrs.keys())}

        def collect_structure(handle):
            groups = []
            datasets = []

            def visitor(name, obj):
                if not name:
                    return
                if isinstance(obj, h5py.Group):
                    groups.append(name)
                elif isinstance(obj, h5py.Dataset):
                    datasets.append(name)

            handle.visititems(visitor)
            return sorted(groups), sorted(datasets)

        def limited_append(errors, message):
            if len(errors) < 12:
                errors.append(message)

        errors = []
        with h5py.File(src_path, "r") as src_handle, h5py.File(pred_path, "r") as pred_handle:
            src_groups, src_datasets = collect_structure(src_handle)
            pred_groups, pred_datasets = collect_structure(pred_handle)

            if src_groups != pred_groups:
                limited_append(errors, "group_structure_mismatch")
            if src_datasets != pred_datasets:
                limited_append(errors, "dataset_structure_mismatch")

            if attrs_dict(src_handle) != attrs_dict(pred_handle):
                limited_append(errors, "root_attrs_mismatch")

            if "points" not in pred_handle or "points" not in src_handle:
                limited_append(errors, "missing_points_dataset")
            else:
                src_points = src_handle["points"]
                pred_points = pred_handle["points"]
                if tuple(pred_points.shape) != tuple(src_points.shape):
                    limited_append(errors, "points_shape_mismatch")
                if str(pred_points.dtype) != str(src_points.dtype):
                    limited_append(errors, "points_dtype_mismatch")
                if not np.isnan(pred_points[:, 0, :]).all():
                    limited_append(errors, "sentinel_not_all_nan")
                if attrs_dict(src_points) != attrs_dict(pred_points):
                    limited_append(errors, "points_attrs_mismatch")

            for group_path in src_groups:
                if group_path not in pred_handle:
                    continue
                if attrs_dict(src_handle[group_path]) != attrs_dict(pred_handle[group_path]):
                    limited_append(errors, f"group_attrs_mismatch:{group_path}")

            for dataset_path in src_datasets:
                if dataset_path == "points" or dataset_path not in pred_handle:
                    continue
                src_dataset = src_handle[dataset_path]
                pred_dataset = pred_handle[dataset_path]
                if tuple(src_dataset.shape) != tuple(pred_dataset.shape):
                    limited_append(errors, f"dataset_shape_mismatch:{dataset_path}")
                    continue
                if str(src_dataset.dtype) != str(pred_dataset.dtype):
                    limited_append(errors, f"dataset_dtype_mismatch:{dataset_path}")
                    continue
                if attrs_dict(src_dataset) != attrs_dict(pred_dataset):
                    limited_append(errors, f"dataset_attrs_mismatch:{dataset_path}")
                if not np.array_equal(src_dataset[...], pred_dataset[...]):
                    limited_append(errors, f"dataset_data_mismatch:{dataset_path}")

        print(json.dumps({"ok": not errors, "errors": errors}))
        raise SystemExit(0 if not errors else 1)
        """).strip()
    shell_command = (
        f"mkdir -p {shlex.quote(eval_root)} && "
        + _uv_runtime_prefix(eval_root)
        + f"uv run --frozen --project {shlex.quote(meta['runtime_env_dir'])} -- "
        + f"python -c {shlex.quote(script)} "
        + f"{shlex.quote(meta['output_file'])} "
        + f"{shlex.quote(meta['input_task_file'])}"
    )
    return "bash -lc " + shlex.quote(shell_command)


def _build_score_command(meta: dict[str, Any], eval_root: str) -> str:
    shell_command = (
        f"mkdir -p {shlex.quote(eval_root)} && "
        + _uv_runtime_prefix(eval_root)
        + f"uv run --frozen --project {shlex.quote(meta['runtime_env_dir'])} -- "
        + f"python {shlex.quote(meta['reference_scorer'])} "
        + f"--pred {shlex.quote(meta['output_file'])} "
        + f"--gt {shlex.quote(meta['reference_gt'])}"
    )
    return "bash -lc " + shlex.quote(shell_command)


class CelegansNeuronTrackingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "137"

    def __init__(self, *, variant_name: str) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=variant_name,
            OS_TYPE="linux",
        )

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def output_dir(self) -> str:
        return f"{self.task_dir}/{self.output_dir_name}"

    @property
    def input_task_file(self) -> str:
        return f"{self.input_dir}/{self.VARIANT_NAME}.h5"

    @property
    def agent_readme(self) -> str:
        return f"{self.input_dir}/AGENT_README.md"

    @property
    def variant_manifest_file(self) -> str:
        return f"{self.input_dir}/variant_manifest.json"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def points_launcher(self) -> str:
        return f"{self.software_dir}/open_points_on_output.sh"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def bootstrap_wrapper(self) -> str:
        return f"{self.software_dir}/bootstrap_runtime.sh"

    @property
    def points_gui_entry(self) -> str:
        return f"{self.software_dir}/POINTS/launch_gui.py"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/{self.VARIANT_NAME}.h5"

    @property
    def reference_scorer(self) -> str:
        return f"{self.reference_dir}/evaluate.py"

    @property
    def reference_gt(self) -> str:
        return f"{self.reference_dir}/gt/{self.VARIANT_NAME}_gt.h5"

    @property
    def task_description(self) -> str:
        return f"""You are completing a C. elegans neuron-tracking task on Linux.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Task file: `{self.input_task_file}`
- Solve guide: `{self.agent_readme}`
- Variant metadata: `{self.variant_manifest_file}`
- Staged runtime manifest: `{self.runtime_pyproject}`
- Staged lockfile: `{self.runtime_lock}`
- GUI launcher: `{self.points_launcher}`
- Programmatic Python wrapper: `{self.python_wrapper}`

## Your Task
1. Read `{self.agent_readme}`.
2. Work from `{self.input_task_file}` for variant `{self.VARIANT_NAME}`.
3. Either launch the GUI with `{self.points_launcher}` or edit the file programmatically with `{self.python_wrapper}`.
4. Fill the `points[t, 1..30, :]` trajectories across frames while preserving neuron identity.
5. Save the completed result only to `{self.output_file}`.

## Output Requirements
- Preserve the existing image data and root attrs.
- Keep `points[:, 0, :]` as `NaN`.
- Do not rename datasets or change array shapes.
- Do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "output_dir_name": self.output_dir_name,
                "input_task_file": self.input_task_file,
                "agent_readme": self.agent_readme,
                "variant_manifest_file": self.variant_manifest_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "points_launcher": self.points_launcher,
                "python_wrapper": self.python_wrapper,
                "bootstrap_wrapper": self.bootstrap_wrapper,
                "points_gui_entry": self.points_gui_entry,
                "output_file": self.output_file,
                "reference_scorer": self.reference_scorer,
                "reference_gt": self.reference_gt,
                "eval_scratch_root": EVAL_SCRATCH_ROOT,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=CelegansNeuronTrackingConfig(
                variant_name=variant_name,
            ).task_description,
            metadata=CelegansNeuronTrackingConfig(
                variant_name=variant_name,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name in VARIANT_NAMES
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_paths = [
        meta["input_task_file"],
        meta["output_file"],
        meta["reference_scorer"],
        meta["reference_gt"],
        meta["runtime_pyproject"],
        meta["runtime_lock"],
    ]
    missing = [path for path in required_paths if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing:
        logger.error("[%s] missing evaluation paths: %s", meta["variant_name"], missing)
        return [0.0]

    eval_root = meta["eval_scratch_root"]
    contract_result = await _run_command(
        session,
        _build_contract_check_command(meta, eval_root),
        check=False,
        timeout=1800,
    )
    try:
        contract_payload = _extract_json_payload(contract_result.get("stdout", ""))
    except Exception:
        logger.exception(
            "[%s] output contract validation produced unparsable stdout=%s stderr=%s",
            meta["variant_name"],
            contract_result.get("stdout", ""),
            contract_result.get("stderr", ""),
        )
        return [0.0]
    if contract_result.get("return_code") != 0 or not contract_payload.get("ok", False):
        logger.error(
            "[%s] output contract validation failed rc=%s stdout=%s stderr=%s",
            meta["variant_name"],
            contract_result.get("return_code"),
            contract_result.get("stdout", ""),
            contract_result.get("stderr", ""),
        )
        return [0.0]

    result = await _run_command(
        session,
        _build_score_command(meta, eval_root),
        check=False,
        timeout=1800,
    )
    if result.get("return_code") != 0:
        logger.error(
            "[%s] remote scoring failed rc=%s stdout=%s stderr=%s",
            meta["variant_name"],
            result.get("return_code"),
            result.get("stdout", ""),
            result.get("stderr", ""),
        )
        return [0.0]

    try:
        payload = _extract_json_payload(result.get("stdout", ""))
        score = float(payload["final"])
    except Exception as exc:
        logger.exception("[%s] failed to parse evaluation JSON: %s", meta["variant_name"], exc)
        logger.error("[%s] raw stdout=%s", meta["variant_name"], result.get("stdout", ""))
        return [0.0]

    logger.info(
        "[%s] output_dir=%s score=%.6f",
        meta["variant_name"],
        meta["output_dir_name"],
        score,
    )
    return [score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
