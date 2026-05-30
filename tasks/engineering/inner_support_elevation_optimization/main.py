"""inner_support_elevation_optimization — PLAXIS 3D support-elevation study task."""

import csv
import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import PureWindowsPath
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "inner_support_elevation_optimization"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"

REMOTE_ROOT_DIR = r"E:\agenthle"
PLAXIS_INPUT_EXE = r"C:\Program Files\Seequent\PLAXIS 3D 2023.2\Plaxis3DInput.exe"
PLAXIS_OUTPUT_EXE = r"C:\Program Files\Seequent\PLAXIS 3D 2023.2\Plaxis3DOutput.exe"
PLAXIS_OUTPUT_VIEWER_EXE = r"C:\Program Files\Seequent\PLAXIS 3D Output Viewer 2023.2\Plaxis3DOutputViewer.exe"

ANSWER_KEY_ORDER = [
    "pos_0m_max_settlement_mm",
    "pos_1m_max_settlement_mm",
    "pos_2m_max_settlement_mm",
    "pos_3m_max_settlement_mm",
    "pos_4m_max_settlement_mm",
    "reduction_0m_to_2m_percent",
    "increase_2m_to_4m_percent",
    "pos_0m_max_disp_mm",
    "pos_1m_max_disp_mm",
    "pos_2m_max_disp_mm",
    "pos_3m_max_disp_mm",
    "pos_4m_max_disp_mm",
    "best_support_position_for_minimum_settlement_m",
    "best_support_position_for_minimum_disp_m",
]

CASE_LABELS = ["pos_0m", "pos_1m", "pos_2m", "pos_3m", "pos_4m"]
CASE_SUMMARY_HEADERS = ["case_label", "support_position_m", "max_settlement_mm", "max_disp_mm"]
REQUIRED_SCREENSHOT_FILES = [
    "pos_0m_output_view.png",
    "pos_2m_output_view.png",
    "pos_4m_output_view.png",
    "settlement_vs_support_position.png",
    "lateral_disp_vs_support_position.png",
]

FIELD_WEIGHTS = {
    "pos_0m_max_settlement_mm": 0.0357,
    "pos_1m_max_settlement_mm": 0.0357,
    "pos_2m_max_settlement_mm": 0.0357,
    "pos_3m_max_settlement_mm": 0.0357,
    "pos_4m_max_settlement_mm": 0.0357,
    "reduction_0m_to_2m_percent": 0.0357,
    "increase_2m_to_4m_percent": 0.0357,
    "pos_0m_max_disp_mm": 0.0357,
    "pos_1m_max_disp_mm": 0.0357,
    "pos_2m_max_disp_mm": 0.0357,
    "pos_3m_max_disp_mm": 0.0357,
    "pos_4m_max_disp_mm": 0.0357,
    "best_support_position_for_minimum_settlement_m": 0.0357,
    "best_support_position_for_minimum_disp_m": 0.0359,
}


def _win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


def _parse_json_text(raw: str) -> dict[str, Any]:
    return json.loads((raw or "").strip())


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid numeric engineering value")
    return float(value)


def _mm_tolerance(reference_value: float) -> float:
    return max(0.30, abs(reference_value) * 0.05)


def _percent_tolerance(reference_value: float) -> float:
    return max(0.30, abs(reference_value) * 0.08)


def _score_numeric_field(field: str, candidate: Any, reference: Any) -> float:
    weight = FIELD_WEIGHTS[field]
    if field.startswith("best_support_position"):
        try:
            return weight if int(candidate) == int(reference) else 0.0
        except Exception:
            return 0.0

    try:
        cand = _safe_float(candidate)
        ref = _safe_float(reference)
    except Exception:
        return 0.0

    tolerance = _percent_tolerance(ref) if field.endswith("_percent") else _mm_tolerance(ref)
    return weight if abs(cand - ref) <= tolerance else 0.0


def _parse_case_summary(csv_text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames != CASE_SUMMARY_HEADERS:
        raise ValueError(f"unexpected case_summary headers: {reader.fieldnames!r}")
    rows = list(reader)
    if len(rows) != len(CASE_LABELS):
        raise ValueError(f"expected {len(CASE_LABELS)} case rows, got {len(rows)}")
    seen_labels = [row["case_label"] for row in rows]
    if seen_labels != CASE_LABELS:
        raise ValueError(f"unexpected case labels: {seen_labels!r}")
    for row in rows:
        int(row["support_position_m"])
        float(row["max_settlement_mm"])
        float(row["max_disp_mm"])
    return rows


@dataclass
class InnerSupportElevationConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = REMOTE_ROOT_DIR
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_dir(self) -> str:
        return _win_join(self.REMOTE_ROOT_DIR, DOMAIN_NAME, TASK_NAME, VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return _win_join(self.task_dir, "input")

    @property
    def answer_template_file(self) -> str:
        return _win_join(self.input_dir, "answer_template.json")

    @property
    def task_brief_file(self) -> str:
        return _win_join(self.input_dir, "task_specific_brief_en.md")

    @property
    def benchmark_spec_file(self) -> str:
        return _win_join(self.input_dir, "benchmark_model_spec_en.md")

    @property
    def assumptions_file(self) -> str:
        return _win_join(self.input_dir, "engineering_assumptions_en.md")

    @property
    def structural_materials_file(self) -> str:
        return _win_join(self.input_dir, "A_zone_structural_materials_en.md")

    @property
    def figure_index_file(self) -> str:
        return _win_join(self.input_dir, "figure_index_en.md")

    @property
    def answer_file(self) -> str:
        return _win_join(self.remote_output_dir, "answer.json")

    @property
    def case_summary_file(self) -> str:
        return _win_join(self.remote_output_dir, "case_summary.csv")

    @property
    def memo_file(self) -> str:
        return _win_join(self.remote_output_dir, "support_position_engineering_memo.md")

    @property
    def ground_truth_file(self) -> str:
        return _win_join(self.reference_dir, "ground_truth.json")

    @property
    def verification_contract_file(self) -> str:
        return _win_join(self.reference_dir, "verification_contract.json")

    @property
    def task_description(self) -> str:
        return f"""\
You are a geotechnical engineer using PLAXIS 3D 2023.2 on Windows.

## Your Task
Evaluate how the inner support elevation changes settlement and lateral wall movement for the staged A-zone pit-in-pit excavation benchmark.

## Input Files
- Benchmark model specification: `{self.benchmark_spec_file}`
- Engineering assumptions: `{self.assumptions_file}`
- Structural materials reference: `{self.structural_materials_file}`
- Task-specific brief and formulas: `{self.task_brief_file}`
- Figure index: `{self.figure_index_file}`
- Answer template: `{self.answer_template_file}`
- Reference figures:
  - `{self.input_dir}\\figure_2_1_project_site_plan.png`
  - `{self.input_dir}\\figure_2_3_excavation_zone_plan.png`
  - `{self.input_dir}\\figure_2_4_2_5_support_layout_and_section.png`
  - `{self.input_dir}\\figure_2_6_monitoring_layout.png`
  - `{self.input_dir}\\figure_3_1_plaxis_model_geometry.png`
  - `{self.input_dir}\\figure_3_2_mesh_layout.png`

## Software
Launch the PLAXIS 3D 2023.2 executables yourself — e.g. via PowerShell:
- `Start-Process '{PLAXIS_INPUT_EXE}'` (PLAXIS 3D Input)
- `Start-Process '{PLAXIS_OUTPUT_EXE}'` (PLAXIS 3D Output)
- `Start-Process '{PLAXIS_OUTPUT_VIEWER_EXE}'` (PLAXIS 3D Output Viewer)

## What You Must Do
1. Review the staged benchmark specification, assumptions, figures, and answer template.
2. Build or duplicate the PLAXIS 3D case family for the tested support elevations `0 m`, `1 m`, `2 m`, `3 m`, and `4 m`.
3. Run the staged calculations through the final excavation state for each case.
4. Extract maximum settlement and maximum outer-wall lateral displacement for each case.
5. Save one or more native PLAXIS project files using the filename prefix:
   `{self.remote_output_dir}\\model_family_support_position`
6. Save the structured summary table exactly to:
   `{self.case_summary_file}`
7. Save the required images exactly to:
   `{self.remote_output_dir}\\pos_0m_output_view.png`
   `{self.remote_output_dir}\\pos_2m_output_view.png`
   `{self.remote_output_dir}\\pos_4m_output_view.png`
   `{self.remote_output_dir}\\settlement_vs_support_position.png`
   `{self.remote_output_dir}\\lateral_disp_vs_support_position.png`
8. Save the engineering memo exactly to:
   `{self.memo_file}`
9. Save the final machine-readable answer exactly to:
   `{self.answer_file}`

## Output Requirements
- `answer.json` must use the exact schema from `{self.answer_template_file}`
- Report settlement and displacement as positive magnitudes in millimeters
- Keep all work products inside `{self.remote_output_dir}`
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "answer_template_file": self.answer_template_file,
                "task_brief_file": self.task_brief_file,
                "benchmark_spec_file": self.benchmark_spec_file,
                "assumptions_file": self.assumptions_file,
                "structural_materials_file": self.structural_materials_file,
                "figure_index_file": self.figure_index_file,
                "answer_file": self.answer_file,
                "case_summary_file": self.case_summary_file,
                "memo_file": self.memo_file,
                "ground_truth_file": self.ground_truth_file,
                "verification_contract_file": self.verification_contract_file,
                "plaxis_input_exe": PLAXIS_INPUT_EXE,
                "plaxis_output_exe": PLAXIS_OUTPUT_EXE,
                "plaxis_output_viewer_exe": PLAXIS_OUTPUT_VIEWER_EXE,
                "required_screenshot_files": REQUIRED_SCREENSHOT_FILES,
                "vm_identity": "sunblaze-4/us-east1-b/agenthle-dev-cpu-licensed",
                "canonical_gcs_root": "gs://ale-data-all/engineering/inner_support_elevation_optimization/base/",
            }
        )
        return metadata


config = InnerSupportElevationConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _list_project_files(session: cb.DesktopSession, output_dir: str) -> list[str]:
    pattern = _win_join(output_dir, "model_family_support_position*")
    result = await session.run_command(f'cmd /c dir /b "{pattern}"', check=False)
    stdout = result.get("stdout", "") or ""
    if result.get("return_code", 1) != 0 and "File Not Found" in (result.get("stderr", "") or ""):
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _has_exact_answer_schema(answer_payload: dict[str, Any]) -> bool:
    return list(answer_payload.keys()) == ANSWER_KEY_ORDER and set(answer_payload.keys()) == set(ANSWER_KEY_ORDER)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir = meta["remote_output_dir"]
    answer_file = meta["answer_file"]
    case_summary_file = meta["case_summary_file"]
    memo_file = meta["memo_file"]
    ground_truth_file = meta["ground_truth_file"]
    verification_contract_file = meta["verification_contract_file"]

    if not await session.exists(ground_truth_file):
        logger.error("missing reference ground truth: %s", ground_truth_file)
        return [0.0]
    if not await session.exists(verification_contract_file):
        logger.error("missing verification contract: %s", verification_contract_file)
        return [0.0]
    if not await session.exists(answer_file):
        logger.error("missing answer.json: %s", answer_file)
        return [0.0]

    required_named_outputs = [case_summary_file, memo_file] + [
        _win_join(output_dir, name) for name in REQUIRED_SCREENSHOT_FILES
    ]
    missing_named_outputs = [path for path in required_named_outputs if not await session.exists(path)]
    project_files = await _list_project_files(session, output_dir)
    if missing_named_outputs or not project_files:
        logger.error(
            "hard gate failed: missing_outputs=%s project_files=%s",
            missing_named_outputs,
            project_files,
        )
        return [0.0]

    try:
        answer_payload = _parse_json_text(await session.read_file(answer_file))
    except Exception as exc:
        logger.error("failed to parse answer.json: %s", exc)
        return [0.0]

    if not isinstance(answer_payload, dict) or not _has_exact_answer_schema(answer_payload):
        logger.error("answer.json does not contain the exact required keys")
        return [0.0]

    try:
        ground_truth = _parse_json_text(await session.read_file(ground_truth_file))
    except Exception as exc:
        logger.error("failed to read ground_truth.json: %s", exc)
        return [0.0]

    score = 0.0

    # Base rubric weights from the submission verification text.
    score += 0.08  # valid answer schema
    score += 0.12  # required evidence file presence and naming
    score += 0.06 if project_files else 0.0
    score += 0.10  # required screenshots/plots present

    try:
        case_summary_rows = _parse_case_summary(await session.read_file(case_summary_file))
        score += 0.08
    except Exception as exc:
        logger.warning("case_summary.csv parse failed: %s", exc)
        case_summary_rows = []

    try:
        memo_text = await session.read_file(memo_file)
        if memo_text.strip():
            score += 0.06
    except Exception as exc:
        logger.warning("failed to read engineering memo: %s", exc)

    numeric_score = 0.0
    for field in ANSWER_KEY_ORDER:
        numeric_score += _score_numeric_field(field, answer_payload.get(field), ground_truth.get(field))
    score += numeric_score

    payload = {
        "score": round(score, 4),
        "numeric_subscore": round(numeric_score, 4),
        "project_file_count": len(project_files),
        "parsed_case_summary": bool(case_summary_rows),
        "missing_named_outputs": missing_named_outputs,
    }
    logger.info("inner_support_elevation_optimization payload: %s", json.dumps(payload, ensure_ascii=False))
    return [float(payload["score"])]
