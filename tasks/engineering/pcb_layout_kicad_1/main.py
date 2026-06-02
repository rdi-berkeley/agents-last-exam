"""Mini Encabulator KiCad PCB layout task."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.engineering.pcb_layout_kicad_1.scripts.score_outputs import score_from_text


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "pcb_layout_kicad_1"
VARIANT_NAME = "base"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
EXPECTED_OUTPUT_FILE = "mini_encabulator.kicad_pcb"
EVAL_TMP_DIR = rf"C:\Users\User\AppData\Local\Temp\agenthle_eval\{TASK_NAME}"
ADMIN_FIXTURE_OUTPUT_DIRS = {"output_test_pos", "output_test_neg"}


def _cmd_stdout(result) -> str:
    if isinstance(result, dict):
        return result.get("stdout", "") or ""
    return getattr(result, "stdout", "") or ""


def _cmd_stderr(result) -> str:
    if isinstance(result, dict):
        return result.get("stderr", "") or ""
    return getattr(result, "stderr", "") or ""


def _cmd_returncode(result) -> int | None:
    if isinstance(result, dict):
        return result.get("return_code", result.get("returncode"))
    return getattr(result, "returncode", None)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"
    KICAD_CLI_PATH: str = os.environ.get(
        "KICAD_CLI_PATH",
        r"C:\Users\User\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe",
    )

    @property
    def output_dir_name(self) -> str:
        return self.REMOTE_OUTPUT_DIR.replace("/", "\\").strip("\\")

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_pcb_path(self) -> str:
        return rf"{self.remote_output_dir}\{EXPECTED_OUTPUT_FILE}"

    @property
    def schematic_path(self) -> str:
        return rf"{self.input_dir}\mini_encabulator.kicad_sch"

    @property
    def open_kicad_launcher(self) -> str:
        return rf"{self.software_dir}\OpenKiCad.bat"

    @property
    def task_description(self) -> str:
        return f"""\
You are designing a KiCad PCB layout for a Mini Encabulator.

## Input
- Schematic: `{self.schematic_path}`
- KiCad launcher: `{self.open_kicad_launcher}`

## Your Task
1. Open the schematic in KiCad and create a PCB layout from it.
2. Import all assigned footprints from the schematic.
3. Place all components for a compact board sized for an SN25 Polycase enclosure.
4. Add exactly four M3 mounting holes on a 1.575 inch by 1.26 inch rectangular spacing pattern.
5. Draw the board outline on `Edge.Cuts`.
6. Route all electrical connections.
7. Add GND copper zones on both `F.Cu` and `B.Cu`, fill them, and add ground stitching vias.
8. Run KiCad DRC and resolve all errors and unconnected items.

## Output
Save the final board layout as:
`{self.output_pcb_path}`
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_tag": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "schematic_path": self.schematic_path,
                "output_pcb_path": self.output_pcb_path,
                "open_kicad_launcher": self.open_kicad_launcher,
                "kicad_cli_path": self.KICAD_CLI_PATH,
                "output_dir_name": self.output_dir_name,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _find_output_pcb(meta: dict, session: cb.DesktopSession) -> str | None:
    expected = meta["output_pcb_path"]
    if (await session.file_exists(expected) or await session.directory_exists(expected)):
        return expected

    result = await session.run_command(
        'powershell -NoProfile -Command '
        + _ps_quote(
            rf"Get-ChildItem -Path {meta['remote_output_dir']} -Filter *.kicad_pcb -File "
            r"| Select-Object -First 1 -ExpandProperty FullName"
        )
    )
    candidate = _cmd_stdout(result).strip()
    return candidate or None


async def _run_kicad_drc(meta: dict, pcb_path: str, session: cb.DesktopSession) -> tuple[str | None, str | None]:
    await session.interface.create_dir(EVAL_TMP_DIR)
    drc_json_path = rf"{EVAL_TMP_DIR}\drc_report.json"
    candidates = [
        meta.get("kicad_cli_path") or "",
        r"C:\Users\User\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe",
        r"C:\Softwares\KiCad\10.0\bin\kicad-cli.exe",
        r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
        r"C:\Users\User\AppData\Local\Programs\KiCad\9.0\bin\kicad-cli.exe",
        r"C:\Softwares\KiCad\9.0\bin\kicad-cli.exe",
        r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",
        r"C:\Softwares\KiCad\8.0\bin\kicad-cli.exe",
        r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",
    ]
    ps_candidates = "@(" + ",".join(_ps_quote(path) for path in candidates if path) + ")"
    command = (
        "powershell -NoProfile -Command "
        + _ps_quote(
            "$ErrorActionPreference='Continue'; "
            f"$candidates={ps_candidates}; "
            "$cli=$null; foreach($p in $candidates){ if($p -and (Test-Path $p)){ $cli=$p; break } }; "
            "if(-not $cli){ $cmd=Get-Command kicad-cli -ErrorAction SilentlyContinue; if($cmd){ $cli=$cmd.Source } }; "
            "if(-not $cli){ Write-Error 'kicad-cli not found'; exit 70 }; "
            f'& $cli pcb drc --format json --output "{drc_json_path}" "{pcb_path}"; '
            "$code=$LASTEXITCODE; "
            f'if(Test-Path "{drc_json_path}"){{ Get-Content "{drc_json_path}" -Raw }}; '
            "exit $code"
        )
    )
    result = await session.run_command(command)
    stdout = _cmd_stdout(result).strip()
    stderr = _cmd_stderr(result).strip()
    returncode = _cmd_returncode(result)

    if stdout.startswith("{"):
        return stdout, None

    unavailable_markers = [
        "Failed to load board",
        "kicad-cli not found",
        "unrecognized file format",
    ]
    if returncode not in (0, None) and any(marker in stderr for marker in unavailable_markers):
        return None, stderr or f"kicad-cli exited {returncode}"
    if returncode not in (0, None) and not stdout:
        return None, stderr or f"kicad-cli exited {returncode} without JSON"
    return stdout or None, None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    pcb_path = await _find_output_pcb(meta, session)
    if not pcb_path:
        logger.warning("missing output PCB in %s", meta["remote_output_dir"])
        return [0.0]

    try:
        pcb_bytes = await session.read_bytes(pcb_path)
    except Exception as exc:
        logger.warning("failed to read output PCB %s: %s", pcb_path, exc)
        return [0.0]

    pcb_text = pcb_bytes.decode("utf-8", errors="replace")
    drc_json, drc_unavailable_reason = await _run_kicad_drc(meta, pcb_path, session)
    result = score_from_text(
        pcb_text,
        drc_json_text=drc_json,
        drc_unavailable_reason=drc_unavailable_reason,
        allow_structural_fallback=meta.get("output_dir_name") in ADMIN_FIXTURE_OUTPUT_DIRS,
    )
    logger.info("PCB layout score result: %s", json.dumps(result, sort_keys=True))

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            prefix=f"{TASK_NAME}_score_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2)
    except Exception:
        pass

    return [float(result["score"])]
