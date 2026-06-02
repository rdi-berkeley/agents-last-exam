"""cailian_road_highway_alignment_2 — Civil 3D highway alignment design task."""

import asyncio
import base64
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


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
class CailianRoadConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_dir(self) -> str:
        return _win(self.REMOTE_ROOT_DIR, DOMAIN_NAME, TASK_NAME, VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return _win(self.task_dir, "input")

    @property
    def topo_surface_file(self) -> str:
        return _win(self.input_dir, "topo_surface.dwg")

    @property
    def alignment_dwg(self) -> str:
        return _win(self.remote_output_dir, "alignment.dwg")

    @property
    def alignment_tsv(self) -> str:
        return _win(self.remote_output_dir, "alignment_metrics.tsv")

    @property
    def civil3d_launcher(self) -> str:
        return _win(self.software_dir, "open_civil3d_2024.bat")

    @property
    def task_description(self) -> str:
        return f"""\
You are a civil engineer using AutoCAD Civil 3D 2024 on Windows.

## Your Task
Design a horizontal alignment for Cailian Road and generate a vertical profile \
that follows the existing ground surface.

## Control Points
- Start: X = {START_X}, Y = {START_Y}, Z = {START_Z}
- End:   X = {END_X}, Y = {END_Y}, Z = {END_Z}
- Both endpoints must be connected within 0.5 m.

## Design Constraints
- Minimum curve radius: {MIN_CURVE_RADIUS} m
- Minimum spiral length (if spirals are used): {MIN_SPIRAL_LENGTH} m
- Design speed: 30 km/h
- Total alignment length: {MIN_TOTAL_LENGTH} m – {MAX_TOTAL_LENGTH} m
- The alignment should include at least 2 horizontal curves to form a \
meaningful highway design.

## Steps
1. Open `{self.topo_surface_file}` — the existing-ground TIN surface.
2. Create a horizontal Alignment between the start and end control points, \
respecting the design constraints above.
3. Use *Create Profile from Surface* to generate the raw existing-ground \
profile along the alignment. Do NOT edit the profile.
4. Save the drawing with the alignment and profile to: \
`{self.alignment_dwg}`
5. Export alignment metrics at 20 m station intervals to: \
`{self.alignment_tsv}`
   - TSV columns must be exactly: Station, X, Y, Z
   - Z values must be the surface elevations at each (X, Y) point.

## Software
- Launch Civil 3D from: `{self.civil3d_launcher}`

## Output
- Save alignment drawing to: `{self.alignment_dwg}`
- Save metrics TSV to: `{self.alignment_tsv}`
- Keep all work inside `{self.remote_output_dir}`
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update({
            "task_id": TASK_ID,
            "task_dir": self.task_dir,
            "input_dir": self.input_dir,
            "topo_surface_file": self.topo_surface_file,
            "alignment_dwg": self.alignment_dwg,
            "alignment_tsv": self.alignment_tsv,
            "civil3d_launcher": self.civil3d_launcher,
            "civil3d_exe": CIVIL3D_EXE,
            "vm_identity": "sunblaze-4/us-west1-c/agenthle-dev-gpu-licensed",
            "canonical_gcs_root": "gs://ale-data-all/engineering/cailian_road_highway_alignment_2/base/",
        })
        return metadata


config = CailianRoadConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
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
