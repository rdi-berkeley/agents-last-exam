"""Cailian Road alignment task using native FreeCAD Road on Linux."""

import asyncio
import base64
import json
import logging
import math
import shlex
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.cailian_road_highway_alignment_2.setup_native import stage_native_task
from tasks.common_setup import BaseTaskSetup
from tasks.engineering.cailian_road_highway_alignment_2.native_commands import (
    NativeRoadEnvironmentError,
    run_native_command,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "cailian_road_highway_alignment_2"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"

CIVIL3D_EXE = r"C:\Program Files\Autodesk\AutoCAD 2024\acad.exe"

# Control points from the submission
START_X, START_Y, START_Z = -52093.6660, -5836.2683, 5.5
END_X, END_Y, END_Z = -50855.6202, -4142.4687, 5.3
CONTROL_POINT_TOLERANCE = 0.5  # metres

# Design constraints
MIN_CURVE_RADIUS = 85.0
MIN_SPIRAL_LENGTH = 25.0
MIN_TOTAL_LENGTH = 1800.0
MAX_TOTAL_LENGTH = 2400.0

# Admin hard gates
MIN_CURVE_COUNT = 2
MIN_PATH_OVER_CHORD = 1.05

# Scoring constants
PASS_THRESHOLD = 70
VERTICAL_TOLERANCE = 0.2  # metres
STATION_INTERVAL_TOLERANCE = 0.5  # metres

CHORD_LENGTH = math.sqrt(
    (END_X - START_X) ** 2 + (END_Y - START_Y) ** 2
)

SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\cailian_road_highway_alignment_2"


def _win(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


@dataclass
class CailianRoadConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REQUIRES_TASK_DATA: bool = False

    @property
    def topo_surface_file(self):
        return f"{self.input_dir}/declared-survey-terrain.obj"

    @property
    def alignment_fcstd(self):
        return f"{self.remote_output_dir}/alignment.FCStd"

    @property
    def alignment_tsv(self):
        return f"{self.remote_output_dir}/alignment_metrics.tsv"

    @property
    def task_description(self):
        return (Path(__file__).parent / "assets/instruction.md").read_text().replace("{input_dir}", self.input_dir).replace("{output_dir}", self.remote_output_dir).replace("{software_dir}", self.software_dir)

    def to_metadata(self):
        return {
            **super().to_metadata(),
            "task_id": TASK_ID,
            "cad_backend": "freecad-road",
            "release_status": "native_controls_passed_awaiting_agent_review",
            "topo_surface_file": self.topo_surface_file,
            "alignment_fcstd": self.alignment_fcstd,
            "alignment_tsv": self.alignment_tsv,
            "native_road_adapter_dir": self.software_dir,
            "native_road_runtime": f"{self.software_dir}/runtime",
            "native_road_work_dir": f"{self.task_dir}/.evaluation",
            "native_road_cpu": 3,
        }


config = CailianRoadConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    await stage_native_task(task_cfg.metadata, session)


def _score_from_verifier(vr: dict) -> dict:
    """Compute the 0-100 score from the verifier JSON result.

    Returns a dict with subscores and the final total.
    """
    result = {
        "hard_gate_failures": [],
        "curve_subscore": 0.0,
        "spiral_subscore": 0.0,
        "vertical_subscore": 0.0,
        "formatting_subscore": 0.0,
        "submitter_total": 0.0,
        "admin_gates_passed": True,
        "final_score": 0.0,
    }

    ainfo = vr.get("alignment_info")
    pinfo = vr.get("profile_info")
    tsv_headers = vr.get("tsv_headers", [])
    tsv_row_count = vr.get("tsv_row_count", 0)
    surface_elevations = vr.get("surface_elevations", [])

    # --- Submitter hard gates 1-6 ---

    # Gate 1: alignment.dwg must contain an Alignment object
    if not ainfo:
        result["hard_gate_failures"].append("gate_1_no_alignment")
        result["final_score"] = 0.0
        return result

    # Gate 2: alignment_metrics.tsv must exist with required columns
    required_cols = ["Station", "X", "Y", "Z"]
    if not vr.get("tsv_exists") or tsv_headers != required_cols:
        result["hard_gate_failures"].append("gate_2_tsv_missing_or_bad_columns")
        result["final_score"] = 0.0
        return result

    # Gate 3: start point within tolerance
    sx, sy = ainfo.get("start_x"), ainfo.get("start_y")
    if sx is None or sy is None:
        result["hard_gate_failures"].append("gate_3_start_point_missing")
        result["final_score"] = 0.0
        return result
    start_dist = math.sqrt((sx - START_X) ** 2 + (sy - START_Y) ** 2)
    if start_dist > CONTROL_POINT_TOLERANCE:
        result["hard_gate_failures"].append(
            f"gate_3_start_point_too_far({start_dist:.3f}m)"
        )
        result["final_score"] = 0.0
        return result

    # Gate 4: end point within tolerance
    ex, ey = ainfo.get("end_x"), ainfo.get("end_y")
    if ex is None or ey is None:
        result["hard_gate_failures"].append("gate_4_end_point_missing")
        result["final_score"] = 0.0
        return result
    end_dist = math.sqrt((ex - END_X) ** 2 + (ey - END_Y) ** 2)
    if end_dist > CONTROL_POINT_TOLERANCE:
        result["hard_gate_failures"].append(
            f"gate_4_end_point_too_far({end_dist:.3f}m)"
        )
        result["final_score"] = 0.0
        return result

    # Gate 5: total length in [1800, 2400]
    total_length = ainfo.get("length", 0.0)
    if total_length < MIN_TOTAL_LENGTH or total_length > MAX_TOTAL_LENGTH:
        result["hard_gate_failures"].append(
            f"gate_5_length_out_of_range({total_length:.1f}m)"
        )
        result["final_score"] = 0.0
        return result

    # Gate 6: at least one Profile associated with the alignment
    pinfo = vr.get("profile_info")
    if not pinfo or pinfo.get("count", 0) < 1:
        result["hard_gate_failures"].append("gate_6_no_profile")
        result["final_score"] = 0.0
        return result

    # --- Admin hard gates 7-8 ---

    n_curves = ainfo.get("n_curves", 0)
    if n_curves < MIN_CURVE_COUNT:
        result["hard_gate_failures"].append(
            f"admin_gate_7_curve_count({n_curves})"
        )
        result["admin_gates_passed"] = False

    path_over_chord = total_length / CHORD_LENGTH if CHORD_LENGTH > 0 else 0
    if path_over_chord < MIN_PATH_OVER_CHORD:
        result["hard_gate_failures"].append(
            f"admin_gate_8_path_over_chord({path_over_chord:.4f})"
        )
        result["admin_gates_passed"] = False

    # --- Subscores ---

    # 1a. Curve radii (20 points)
    curves = ainfo.get("curves", [])
    if n_curves < 2:
        result["curve_subscore"] = 0.0
    else:
        m = sum(1 for c in curves if c.get("radius", 0) >= MIN_CURVE_RADIUS)
        result["curve_subscore"] = 20.0 * m / n_curves

    # 1b. Spiral lengths (20 points)
    spirals = ainfo.get("spirals", [])
    n_spirals = len(spirals)
    if n_spirals == 0:
        result["spiral_subscore"] = 20.0  # spirals optional
    else:
        t = sum(1 for s in spirals if s.get("length", 0) >= MIN_SPIRAL_LENGTH)
        result["spiral_subscore"] = 20.0 * t / n_spirals

    # 2. Vertical profile (40 points) — computed in evaluate() with TSV data

    # 3. Formatting (20 points)
    header_score = 10.0 if tsv_headers == required_cols else 0.0
    result["formatting_subscore"] = header_score
    # Station interval check deferred to evaluate() where we have TSV data

    result["submitter_total"] = (
        result["curve_subscore"]
        + result["spiral_subscore"]
        + result["vertical_subscore"]
        + result["formatting_subscore"]
    )

    if not result["admin_gates_passed"]:
        result["final_score"] = 0.0
    else:
        result["final_score"] = result["submitter_total"]

    return result


async def _run_cmd(session, cmd, retries=3, delay=5, check=False):
    """Run a remote command with retry logic for transient connection failures."""
    last_err = None
    for attempt in range(retries):
        try:
            return await session.run_command(cmd, check=check)
        except RuntimeError as e:
            last_err = e
            if attempt < retries - 1:
                logger.warning(
                    "run_command attempt %d failed: %s — retrying in %ds",
                    attempt + 1, e, delay,
                )
                await asyncio.sleep(delay)
    raise last_err


def _score_native_road(vr: dict) -> dict:
    """Score independently reopened Road geometry with the original weights."""
    result = _score_from_verifier(vr)
    if vr.get("error") or not vr.get("native_road_verified"):
        result["hard_gate_failures"].append("native_road_verification_failed")
    if result["hard_gate_failures"]:
        result["final_score"] = 0.0
        return result
    rows = vr.get("tsv_rows", [])
    elevations = vr.get("surface_elevations", [])
    if len(rows) < 3 or len(rows) != len(elevations):
        result["hard_gate_failures"].append("native_surface_samples_missing")
        result["final_score"] = 0.0
        return result
    try:
        stations = [float(row["Station"]) for row in rows]
        heights = [float(row["Z"]) for row in rows]
        if not all(math.isfinite(value) for value in stations + heights + elevations):
            raise ValueError("nonfinite samples")
    except (KeyError, TypeError, ValueError):
        result["hard_gate_failures"].append("invalid_native_samples")
        result["final_score"] = 0.0
        return result
    matches = sum(
        abs(actual - expected) <= VERTICAL_TOLERANCE
        for actual, expected in zip(heights[1:-1], elevations[1:-1])
    )
    result["vertical_subscore"] = 40.0 * matches / (len(rows) - 2)
    interval_ok = all(
        abs(current - previous - 20.0) <= STATION_INTERVAL_TOLERANCE
        for previous, current in zip(stations[:-2], stations[1:-1])
    )
    result["formatting_subscore"] += 10.0 if interval_ok else 0.0
    result["submitter_total"] = sum(
        result[key]
        for key in (
            "curve_subscore", "spiral_subscore", "vertical_subscore", "formatting_subscore"
        )
    )
    result["final_score"] = result["submitter_total"]
    return result


async def _evaluate_native_road(meta: dict, session) -> list[float]:
    """Run the trusted Road adapter in a separate, resource-limited display."""
    work_dir = meta["native_road_work_dir"]
    adapter_dir = meta["native_road_adapter_dir"]
    request_path = f"{work_dir}/evaluation-request.json"
    receipt_path = f"{work_dir}/native-verification.json"
    request = {
        "mode": "verify",
        "adapter_dir": adapter_dir,
        "runtime": meta["native_road_runtime"],
        "terrain": meta["topo_surface_file"],
        "alignment": meta["alignment_fcstd"],
        "tsv": meta["alignment_tsv"],
        "receipt": receipt_path,
    }
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    prepare = (
        "import base64,pathlib; "
        f"pathlib.Path({work_dir!r}).mkdir(parents=True,exist_ok=True); "
        f"pathlib.Path({receipt_path!r}).unlink(missing_ok=True); "
        f"pathlib.Path({request_path!r}).write_bytes(base64.b64decode({encoded!r}))"
    )
    await run_native_command(session, shlex.join(["python3", "-c", prepare]), seconds=30)
    command = shlex.join([
        "sudo", "-n", "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
        "--uid=user", "-p", "CPUQuota=100%", "-p",
        f"AllowedCPUs={meta.get('native_road_cpu', 0)}",
        "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", "-p", f"RuntimeMaxSec={min(300, int(meta.get('native_road_seconds', 300)))}",
        "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "QTWEBENGINE_DISABLE_SANDBOX=1", f"CAILIAN_NATIVE_REQUEST={request_path}",
        "xvfb-run", "-a", "-e", f"{work_dir}/xvfb.log",
        f"{meta['native_road_runtime']}/FreeCAD.AppImage",
        "--user-cfg", f"{work_dir}/FreeCAD-user.cfg",
        "--system-cfg", f"{work_dir}/FreeCAD-system.cfg",
        f"{adapter_dir}/entry.FCMacro",
    ])
    await run_native_command(session, command)
    receipt = await run_native_command(session, shlex.join(["cat", receipt_path]), seconds=30)
    try:
        verification = json.loads(receipt.get("stdout") or "")
    except (TypeError, json.JSONDecodeError) as error:
        raise NativeRoadEnvironmentError("Native Road verifier did not write valid JSON") from error
    if not isinstance(verification, dict) or not (
        verification.get("native_road_verified") is True or isinstance(verification.get("error"), str)
    ):
        raise NativeRoadEnvironmentError("Native Road verifier receipt is incomplete")
    score = _score_native_road(verification)
    logger.info("cailian native Road scoring payload: %s", json.dumps(score))
    return [score["final_score"] / 100.0]


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if meta.get("cad_backend") == "freecad-road":
        return await _evaluate_native_road(meta, session)
    output_dir = meta["remote_output_dir"]
    alignment_dwg = meta["alignment_dwg"]
    alignment_tsv = meta["alignment_tsv"]
    topo_surface = meta["topo_surface_file"]

    # Check basic file existence via run_command (avoids otel wrapper bug)
    dwg_chk = await _run_cmd(session,
        f'if exist "{alignment_dwg}" (echo EXISTS) else (echo MISSING)',
        check=False,
    )
    if "EXISTS" not in (dwg_chk.get("stdout") or ""):
        logger.error("alignment.dwg missing at %s", alignment_dwg)
        return [0.0]
    tsv_chk = await _run_cmd(session,
        f'if exist "{alignment_tsv}" (echo EXISTS) else (echo MISSING)',
    )
    if "EXISTS" not in (tsv_chk.get("stdout") or ""):
        logger.error("alignment_metrics.tsv missing at %s", alignment_tsv)
        return [0.0]

    # Upload verifier to eval temp dir via chunked base64 powershell writes
    await _run_cmd(session,
        f'mkdir "{EVAL_TMP_DIR}" 2>nul',
    )
    verify_script = _read_script("verify_alignment.py")
    verify_path = _win(EVAL_TMP_DIR, "verify_alignment.py")
    b64 = base64.b64encode(verify_script.encode("utf-8")).decode("ascii")

    CHUNK_SIZE = 4000
    chunks = [b64[i : i + CHUNK_SIZE] for i in range(0, len(b64), CHUNK_SIZE)]
    b64_var = _win(EVAL_TMP_DIR, "_b64.txt")

    # Write first chunk (overwrite)
    await _run_cmd(session,
        f'powershell -Command "Set-Content -Path \'{b64_var}\' '
        f"-Value '{chunks[0]}' -NoNewline\"",
    )
    # Append remaining chunks
    for chunk in chunks[1:]:
        await _run_cmd(session,
            f'powershell -Command "Add-Content -Path \'{b64_var}\' '
            f"-Value '{chunk}' -NoNewline\"",
        )
    # Decode base64 file to the actual script
    await _run_cmd(session,
        f'powershell -Command "[System.IO.File]::WriteAllBytes('
        f"'{verify_path}', "
        f"[System.Convert]::FromBase64String("
        f"[System.IO.File]::ReadAllText('{b64_var}')))\"",
    )

    # Run verifier on VM (TSV-only mode — LISP/COM extraction unreliable)
    work_dir = _win(EVAL_TMP_DIR, "xml_exports")
    cmd = (
        f'python "{verify_path}" '
        f'--alignment "{alignment_dwg}" '
        f'--topo "{topo_surface}" '
        f'--tsv "{alignment_tsv}" '
        f'--work-dir "{work_dir}" '
        f'--tsv-only'
    )
    logger.info("running VM-side verifier: %s", cmd)
    result = await _run_cmd(session, cmd, retries=3, delay=10)

    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if stderr:
        logger.info("verifier stderr:\n%s", stderr)

    if result.get("return_code", 1) != 0 or not stdout:
        logger.error(
            "verifier failed: rc=%s stderr=%s",
            result.get("return_code"),
            stderr,
        )
        return [0.0]

    try:
        vr = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.error("verifier JSON parse error: %s\nstdout: %s", exc, stdout[:500])
        return [0.0]

    if vr.get("error"):
        logger.error("verifier reported error: %s", vr["error"])
        return [0.0]

    # Read TSV from VM via run_command (read_file triggers otel wrapper bug)
    tsv_result = await _run_cmd(session,
        f'type "{alignment_tsv}"',
    )
    tsv_text = (tsv_result.get("stdout") or "").strip()
    tsv_lines = tsv_text.splitlines()

    # Compute scoring result from verifier output
    sr = _score_from_verifier(vr)

    if sr["hard_gate_failures"]:
        # Check if only admin gates failed (submitter gates passed)
        submitter_gates = [g for g in sr["hard_gate_failures"] if not g.startswith("admin_")]
        if submitter_gates:
            logger.info("submitter hard gates failed: %s", submitter_gates)
            return [0.0]

    # --- Vertical subscore (40 points) ---
    surface_elevations = vr.get("surface_elevations", [])
    tsv_rows = []
    if len(tsv_lines) > 1:
        headers = tsv_lines[0].split("\t")
        for line in tsv_lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    tsv_rows.append({
                        "station": float(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "z": float(parts[3]),
                    })
                except ValueError:
                    pass

    lisp_available = vr.get("_lisp_extraction_available", False)

    if len(tsv_rows) >= 3 and len(surface_elevations) == len(tsv_rows):
        interior_match = 0
        interior_total = 0
        for i in range(1, len(tsv_rows) - 1):
            surf_z = surface_elevations[i]
            if surf_z is None:
                continue
            interior_total += 1
            tsv_z = tsv_rows[i]["z"]
            if abs(tsv_z - surf_z) <= VERTICAL_TOLERANCE:
                interior_match += 1
        if interior_total > 0:
            sr["vertical_subscore"] = 40.0 * interior_match / interior_total
    elif not lisp_available and len(tsv_rows) >= 3:
        z_vals = [r["z"] for r in tsv_rows if r["z"] is not None]
        if len(z_vals) >= 3:
            z_range = max(z_vals) - min(z_vals)
            z_nonzero = sum(1 for z in z_vals if abs(z) > 0.01)
            if z_nonzero == len(z_vals) and z_range > 0.1:
                sr["vertical_subscore"] = 40.0
            else:
                sr["vertical_subscore"] = 0.0
        else:
            sr["vertical_subscore"] = 0.0
    else:
        sr["vertical_subscore"] = 0.0

    # --- Station interval subscore (10 points) ---
    # Exclude last interval (alignment may not end on exact 20m boundary)
    interval_ok = True
    if len(tsv_rows) >= 3:
        for i in range(1, len(tsv_rows) - 1):
            interval = tsv_rows[i]["station"] - tsv_rows[i - 1]["station"]
            if abs(interval - 20.0) > STATION_INTERVAL_TOLERANCE:
                interval_ok = False
                break
    else:
        interval_ok = False

    sr["formatting_subscore"] += 10.0 if interval_ok else 0.0

    # Recompute total
    sr["submitter_total"] = (
        sr["curve_subscore"]
        + sr["spiral_subscore"]
        + sr["vertical_subscore"]
        + sr["formatting_subscore"]
    )

    if not sr["admin_gates_passed"]:
        sr["final_score"] = 0.0
    else:
        sr["final_score"] = sr["submitter_total"]

    # Normalize to [0.0, 1.0]
    normalized = sr["final_score"] / 100.0

    payload = {
        "normalized_score": round(normalized, 4),
        "raw_score": round(sr["final_score"], 2),
        "curve_subscore": round(sr["curve_subscore"], 2),
        "spiral_subscore": round(sr["spiral_subscore"], 2),
        "vertical_subscore": round(sr["vertical_subscore"], 2),
        "formatting_subscore": round(sr["formatting_subscore"], 2),
        "hard_gate_failures": sr["hard_gate_failures"],
        "admin_gates_passed": sr["admin_gates_passed"],
    }
    logger.info("cailian_road scoring payload: %s", json.dumps(payload, ensure_ascii=False))
    return [normalized]
