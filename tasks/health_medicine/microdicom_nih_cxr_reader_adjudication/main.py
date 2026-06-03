"""microdicom_nih_cxr_reader_adjudication — cohort chest X-ray reader adjudication."""

import csv
import io
import logging
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PureWindowsPath
from typing import Any

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

REQUIRED_BOX_HEADERS = [
    "case_id",
    "selected_reader",
    "final_x",
    "final_y",
    "final_width",
    "final_height",
]
REQUIRED_LOG_HEADERS = [
    "case_id",
    "selected_reader",
    "disagreement_type",
    "resolution_basis",
]
REQUIRED_IMPRESSION_HEADERS = [
    "case_id",
    "final_finding_label",
    "final_impression_label",
]
BOX_IOU_THRESHOLD = Decimal("0.50")


def win_join(*parts: str) -> str:
    return str(PureWindowsPath(*parts))


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "microdicom_nih_cxr_reader_adjudication"
    VARIANT_NAME: str = "base"

    @property
    def task_dir(self) -> str:
        return win_join(self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.VARIANT_NAME)

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    @property
    def software_launcher(self) -> str:
        return win_join(self.task_dir, "software", "launch_mdicom.cmd")

    @property
    def adjudication_rules_path(self) -> str:
        return win_join(self.input_dir, "adjudication_rules.md")

    @property
    def case_manifest_path(self) -> str:
        return win_join(self.input_dir, "case_manifest.tsv")

    @property
    def reader_a_path(self) -> str:
        return win_join(self.input_dir, "reader_a_annotations.tsv")

    @property
    def reader_b_path(self) -> str:
        return win_join(self.input_dir, "reader_b_annotations.tsv")

    @property
    def clinical_notes_dir(self) -> str:
        return win_join(self.input_dir, "clinical_notes")

    @property
    def dicom_cases_dir(self) -> str:
        return win_join(self.input_dir, "dicom_cases")

    @property
    def adjudicated_boxes_path(self) -> str:
        return win_join(self.remote_output_dir, "adjudicated_boxes.tsv")

    @property
    def adjudication_log_path(self) -> str:
        return win_join(self.remote_output_dir, "adjudication_log.tsv")

    @property
    def final_impressions_path(self) -> str:
        return win_join(self.remote_output_dir, "final_impressions.tsv")

    @property
    def reference_boxes_path(self) -> str:
        return win_join(self.reference_dir, "adjudicated_boxes.tsv")

    @property
    def reference_log_path(self) -> str:
        return win_join(self.reference_dir, "adjudication_log.tsv")

    @property
    def reference_impressions_path(self) -> str:
        return win_join(self.reference_dir, "final_impressions.tsv")

    @property
    def task_description(self) -> str:
        return f"""\
You are a radiology adjudicator working in a Windows desktop environment.

Your task is to adjudicate between two competing reader annotations for a cohort of 9 chest X-ray studies and save the final cohort outputs.

Files and tools:
- DICOM viewer launcher: `{self.software_launcher}`
- Cohort instructions: `{self.adjudication_rules_path}`
- Case manifest: `{self.case_manifest_path}`
- Reader A draft annotations: `{self.reader_a_path}`
- Reader B draft annotations: `{self.reader_b_path}`
- Clinical notes directory: `{self.clinical_notes_dir}`
- DICOM cases directory: `{self.dicom_cases_dir}`
- Output directory: `{self.remote_output_dir}`

What you must do:
1. Read `adjudication_rules.md` and `case_manifest.tsv`.
2. Use the case-specific clinical notes as context.
3. Launch MicroDicom from the provided launcher.
4. For each case in the manifest, open the listed DICOM file in MicroDicom.
5. Compare the candidate reader boxes from `reader_a_annotations.tsv` and `reader_b_annotations.tsv` against the image.
6. Choose the better-supported reader result for each case.
7. Save exactly these UTF-8 tab-delimited files under `{self.remote_output_dir}`:
   - `{self.adjudicated_boxes_path}`
   - `{self.adjudication_log_path}`
   - `{self.final_impressions_path}`

Required schemas:
- `adjudicated_boxes.tsv`: `case_id`, `selected_reader`, `final_x`, `final_y`, `final_width`, `final_height`
- `adjudication_log.tsv`: `case_id`, `selected_reader`, `disagreement_type`, `resolution_basis`
- `final_impressions.tsv`: `case_id`, `final_finding_label`, `final_impression_label`

Task constraints:
- `selected_reader` must be either `reader_a` or `reader_b`
- `disagreement_type` must be `box_disagreement`
- `final_finding_label` must be `Atelectasis`
- `final_impression_label` must be `positive_for_atelectasis`
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "software_launcher": self.software_launcher,
                "adjudication_rules_path": self.adjudication_rules_path,
                "case_manifest_path": self.case_manifest_path,
                "reader_a_path": self.reader_a_path,
                "reader_b_path": self.reader_b_path,
                "clinical_notes_dir": self.clinical_notes_dir,
                "dicom_cases_dir": self.dicom_cases_dir,
                "adjudicated_boxes_path": self.adjudicated_boxes_path,
                "adjudication_log_path": self.adjudication_log_path,
                "final_impressions_path": self.final_impressions_path,
                "reference_boxes_path": self.reference_boxes_path,
                "reference_log_path": self.reference_log_path,
                "reference_impressions_path": self.reference_impressions_path,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_field(value: str) -> str:
    return _normalize_text(value)


def _parse_decimal(value: str) -> Decimal:
    try:
        return Decimal(_normalize_field(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal field: {value!r}") from exc


def _parse_tsv(text: str, expected_headers: list[str]) -> list[dict[str, str]]:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("empty TSV payload")
    reader = csv.DictReader(io.StringIO(normalized), delimiter="\t")
    headers = reader.fieldnames or []
    if headers != expected_headers:
        raise ValueError(f"header mismatch: expected {expected_headers!r}, got {headers!r}")
    return [{key: _normalize_field(value or "") for key, value in row.items()} for row in reader]


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row["case_id"])


def _exact_text_table_match(candidate_text: str, reference_text: str, headers: list[str]) -> bool:
    candidate_rows = _sort_rows(_parse_tsv(candidate_text, headers))
    reference_rows = _sort_rows(_parse_tsv(reference_text, headers))
    return candidate_rows == reference_rows


def _case_box_map(text: str) -> dict[str, dict[str, Any]]:
    rows = _parse_tsv(text, REQUIRED_BOX_HEADERS)
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row["case_id"]
        if not case_id or case_id in parsed:
            raise ValueError(f"invalid or duplicate case_id in boxes TSV: {case_id!r}")
        parsed[case_id] = {
            "selected_reader": row["selected_reader"],
            "x": _parse_decimal(row["final_x"]),
            "y": _parse_decimal(row["final_y"]),
            "width": _parse_decimal(row["final_width"]),
            "height": _parse_decimal(row["final_height"]),
        }
    return parsed


def _iou(candidate: dict[str, Any], reference: dict[str, Any]) -> Decimal:
    cand_x2 = candidate["x"] + candidate["width"]
    cand_y2 = candidate["y"] + candidate["height"]
    ref_x2 = reference["x"] + reference["width"]
    ref_y2 = reference["y"] + reference["height"]
    inter_x1 = max(candidate["x"], reference["x"])
    inter_y1 = max(candidate["y"], reference["y"])
    inter_x2 = min(cand_x2, ref_x2)
    inter_y2 = min(cand_y2, ref_y2)
    inter_w = max(Decimal("0"), inter_x2 - inter_x1)
    inter_h = max(Decimal("0"), inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return Decimal("0")
    cand_area = candidate["width"] * candidate["height"]
    ref_area = reference["width"] * reference["height"]
    union = cand_area + ref_area - inter_area
    if union <= 0:
        return Decimal("0")
    return inter_area / union


def _boxes_pass(candidate_text: str, reference_text: str) -> bool:
    candidate_boxes = _case_box_map(candidate_text)
    reference_boxes = _case_box_map(reference_text)
    if set(candidate_boxes) != set(reference_boxes):
        return False
    for case_id, ref in reference_boxes.items():
        cand = candidate_boxes[case_id]
        if cand["selected_reader"] not in {"reader_a", "reader_b"}:
            return False
        if cand["selected_reader"] != ref["selected_reader"]:
            return False
        if cand["width"] <= 0 or cand["height"] <= 0:
            return False
        if _iou(cand, ref) < BOX_IOU_THRESHOLD:
            return False
    return True


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        reference_boxes = await session.read_file(meta["reference_boxes_path"])
        reference_log = await session.read_file(meta["reference_log_path"])
        reference_impressions = await session.read_file(meta["reference_impressions_path"])
    except Exception as exc:
        logger.error("Reference evaluation inputs missing or unreadable: %s", exc)
        return [0.0]

    boxes_ok = False
    if (await session.file_exists(meta["adjudicated_boxes_path"]) or await session.directory_exists(meta["adjudicated_boxes_path"])):
        try:
            candidate_boxes = await session.read_file(meta["adjudicated_boxes_path"])
            boxes_ok = _boxes_pass(candidate_boxes, reference_boxes)
        except Exception as exc:
            logger.warning("Boxes evaluation failed: %s", exc)
    else:
        logger.warning("Missing required output file: %s", meta["adjudicated_boxes_path"])

    log_ok = False
    if (await session.file_exists(meta["adjudication_log_path"]) or await session.directory_exists(meta["adjudication_log_path"])):
        try:
            candidate_log = await session.read_file(meta["adjudication_log_path"])
            candidate_log_rows = _parse_tsv(candidate_log, REQUIRED_LOG_HEADERS)
            log_ok = True
            for row in candidate_log_rows:
                if row["selected_reader"] not in {"reader_a", "reader_b"}:
                    log_ok = False
                    break
                if row["disagreement_type"] != "box_disagreement":
                    log_ok = False
                    break
            if log_ok:
                ref_rows = _sort_rows(_parse_tsv(reference_log, REQUIRED_LOG_HEADERS))
                cand_rows = _sort_rows(candidate_log_rows)
                check_keys = ["case_id", "selected_reader", "disagreement_type"]
                if len(cand_rows) != len(ref_rows):
                    log_ok = False
                else:
                    for c, r in zip(cand_rows, ref_rows):
                        if any(c[k] != r[k] for k in check_keys):
                            log_ok = False
                            break
        except Exception as exc:
            logger.warning("Adjudication log evaluation failed: %s", exc)
    else:
        logger.warning("Missing required output file: %s", meta["adjudication_log_path"])

    impressions_ok = False
    if (await session.file_exists(meta["final_impressions_path"]) or await session.directory_exists(meta["final_impressions_path"])):
        try:
            candidate_impressions = await session.read_file(meta["final_impressions_path"])
            candidate_impression_rows = _parse_tsv(candidate_impressions, REQUIRED_IMPRESSION_HEADERS)
            impressions_ok = True
            for row in candidate_impression_rows:
                if row["final_finding_label"] != "Atelectasis":
                    impressions_ok = False
                    break
                if row["final_impression_label"] != "positive_for_atelectasis":
                    impressions_ok = False
                    break
            if impressions_ok:
                impressions_ok = _exact_text_table_match(
                    candidate_impressions,
                    reference_impressions,
                    REQUIRED_IMPRESSION_HEADERS,
                )
        except Exception as exc:
            logger.warning("Final impressions evaluation failed: %s", exc)
    else:
        logger.warning("Missing required output file: %s", meta["final_impressions_path"])

    try:
        score = sum(float(value) for value in (boxes_ok, log_ok, impressions_ok)) / 3.0
        logger.info(
            "boxes_ok=%s log_ok=%s impressions_ok=%s score=%.6f",
            boxes_ok,
            log_ok,
            impressions_ok,
            score,
        )
        return [score]
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        return [0.0]
