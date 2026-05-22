"""AgentHLE task: compress_3dgs_scene_ply."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover

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

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import score_output_bundle  # noqa: E402

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "visual_media"
TASK_NAME = "compress_3dgs_scene_ply"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
WINDOWS_REMOTE_ROOT = r"E:\agenthle"


def _remote_join(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        if part:
            path = path / part
    return str(path)


async def _run_command(
    session: cb.DesktopSession, command: str, *, check: bool = False
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


async def _path_exists(session: cb.DesktopSession, path: str) -> bool:
    result = await _run_command(
        session,
        f"powershell -NoProfile -Command \"if (Test-Path -LiteralPath '{path}') {{ exit 0 }} else {{ exit 1 }}\"",
        check=False,
    )
    return result.get("return_code", 1) == 0


async def _read_text(session: cb.DesktopSession, path: str) -> str:
    try:
        return await session.read_file(path)
    except Exception:
        data = await session.read_bytes(path)
        return data.decode("utf-8")


async def _read_bytes(session: cb.DesktopSession, path: str) -> bytes:
    return await session.read_bytes(path)


async def _count_remote_ply_vertices(session: cb.DesktopSession, path: str) -> int | None:
    command = (
        "powershell -NoProfile -Command "
        '"'
        f"$path = '{path}'; "
        "$fs = [System.IO.File]::OpenRead($path); "
        "$sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::ASCII, $false, 1024, $true); "
        "try { "
        "  while (($line = $sr.ReadLine()) -ne $null) { "
        "    if ($line.StartsWith('element vertex ')) { "
        "      Write-Output $line.Substring(15); "
        "      exit 0 "
        "    } "
        "    if ($line -eq 'end_header') { break } "
        "  } "
        "  exit 2 "
        "} finally { "
        "  $sr.Dispose(); "
        "  $fs.Dispose(); "
        "}"
        '"'
    )
    result = await _run_command(session, command, check=False)
    if result.get("return_code", 1) != 0:
        return None
    stdout = (result.get("stdout") or "").strip()
    try:
        return int(stdout)
    except ValueError:
        return None


@dataclass
class Compress3dgsScenePlyConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", WINDOWS_REMOTE_ROOT)
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")

    def __init__(self, *, remote_output_dir: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="windows",
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", WINDOWS_REMOTE_ROOT),
            REMOTE_OUTPUT_DIR=remote_output_dir or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def task_dir(self) -> str:
        return _remote_join(self.REMOTE_ROOT_DIR, DOMAIN_NAME, TASK_NAME, VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return _remote_join(self.task_dir, "input")

    @property
    def scene_dir(self) -> str:
        return _remote_join(self.input_dir, "scene")

    @property
    def task_prompt_file(self) -> str:
        return _remote_join(self.input_dir, "task_prompt.md")

    @property
    def baseline_results_file(self) -> str:
        return _remote_join(self.input_dir, "baseline_results.json")

    @property
    def scene_manifest_file(self) -> str:
        return _remote_join(self.input_dir, "scene_manifest.json")

    @property
    def scene_ply_file(self) -> str:
        return _remote_join(self.scene_dir, "point_cloud_30000.ply")

    @property
    def sparse_dir(self) -> str:
        return _remote_join(self.scene_dir, "sparse", "0")

    @property
    def images_dir(self) -> str:
        return _remote_join(self.scene_dir, "images")

    @property
    def eval_contract_file(self) -> str:
        return _remote_join(self.reference_dir, "eval_contract.json")

    @property
    def reference_test_images_dir(self) -> str:
        return _remote_join(self.reference_dir, "test_images")

    @property
    def output_ply_file(self) -> str:
        return _remote_join(self.remote_output_dir, "point_cloud_30000_compressed.ply")

    @property
    def output_render_dir(self) -> str:
        return _remote_join(self.remote_output_dir, "rendered_test_views")

    @property
    def output_results_file(self) -> str:
        return _remote_join(self.remote_output_dir, "results.json")

    @property
    def output_compression_report_file(self) -> str:
        return _remote_join(self.remote_output_dir, "compression_report.json")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Windows GPU neural-rendering benchmark.

Read these staged inputs first:
- `{self.task_prompt_file}`
- `{self.baseline_results_file}`
- `{self.scene_manifest_file}`

Core visible asset roots:
- `{self.scene_ply_file}`
- `{self.sparse_dir}`
- `{self.images_dir}`

Pre-installed 3DGS rendering runtime (software entry points):
- `{self.software_dir}\\python_3dgs.bat <script.py>` — Python with the official gaussian-splatting repo on PYTHONPATH and CUDA configured
- `{self.software_dir}\\render_3dgs.bat --model_path <path> [...]` — official render.py wrapper
- `{self.software_dir}\\README.md` — details on the installed runtime

Write your outputs only under `{self.remote_output_dir}`.
Required output files:
- `{self.output_ply_file}`
- `{self.output_render_dir}\\<holdout filename>.jpg`
- `{self.output_results_file}`
- `{self.output_compression_report_file}`

Important:
- the holdout filenames are public
- the visible JPGs for those holdout names are placeholders only
- hidden evaluator images are used for scoring after you finish
- do not modify files under `{self.input_dir}`
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
                "task_prompt_file": self.task_prompt_file,
                "baseline_results_file": self.baseline_results_file,
                "scene_manifest_file": self.scene_manifest_file,
                "scene_ply_file": self.scene_ply_file,
                "sparse_dir": self.sparse_dir,
                "images_dir": self.images_dir,
                "eval_contract_file": self.eval_contract_file,
                "reference_test_images_dir": self.reference_test_images_dir,
                "output_ply_file": self.output_ply_file,
                "output_render_dir": self.output_render_dir,
                "output_results_file": self.output_results_file,
                "output_compression_report_file": self.output_compression_report_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


@cb.tasks_config
def load() -> list[cb.Task]:
    config = Compress3dgsScenePlyConfig()
    return [
        cb.Task(description=config.task_description, metadata=config.to_metadata(), computer=None)
    ]


@cb.setup_task
async def start(task: cb.Task, session: cb.DesktopSession) -> None:
    await _setup(task, session)


@cb.evaluate_task
async def evaluate(task: cb.Task, session: cb.DesktopSession) -> list[float]:
    meta = task.metadata
    contract = json.loads(await _read_text(session, meta["eval_contract_file"]))
    holdout_names = contract["holdout_filenames"]
    original_count = int(contract["original_gaussian_count"])
    min_reduction = float(contract["min_reduction_fraction"])

    output_required = [
        meta["output_ply_file"],
        meta["output_results_file"],
        meta["output_compression_report_file"],
        meta["output_render_dir"],
    ]
    missing = [path for path in output_required if not await _path_exists(session, path)]
    if missing:
        logger.warning("missing output paths: %s", missing)
        return [0.0]

    output_count = await _count_remote_ply_vertices(session, meta["output_ply_file"])
    if output_count is None:
        logger.warning("could not read PLY vertex count from %s", meta["output_ply_file"])
        return [0.0]
    reduction_fraction = 1.0 - (output_count / original_count)
    if reduction_fraction < min_reduction:
        logger.info(
            "short-circuit fail: insufficient_gaussian_reduction (original=%s output=%s reduction=%s required=%s)",
            original_count,
            output_count,
            reduction_fraction,
            min_reduction,
        )
        return [0.0]

    rendered = {}
    reference = {}
    for name in holdout_names:
        rendered_path = _remote_join(meta["output_render_dir"], name)
        reference_path = _remote_join(meta["reference_test_images_dir"], name)
        if await _path_exists(session, rendered_path):
            rendered[name] = await _read_bytes(session, rendered_path)
        if await _path_exists(session, reference_path):
            reference[name] = await _read_bytes(session, reference_path)

    result = score_output_bundle(
        output_vertex_count=output_count,
        results_json=await _read_text(session, meta["output_results_file"]),
        compression_report_json=await _read_text(session, meta["output_compression_report_file"]),
        eval_contract_json=json.dumps(contract),
        rendered_images=rendered,
        reference_images=reference,
    )
    logger.info("score result: %s", json.dumps(result.to_dict(), ensure_ascii=False))
    return [float(result.score)]
