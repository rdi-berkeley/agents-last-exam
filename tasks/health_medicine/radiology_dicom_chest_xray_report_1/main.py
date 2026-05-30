"""Radiology chest X-ray report task with deterministic bbox + LLM-judged report scoring."""

import json
import logging
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb
from dotenv import load_dotenv
from openai import OpenAI
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)

VARIANT_NAMES = ["base"] + [f"variant_{index:02d}" for index in range(2, 11)]

DICOM_VIEWER_LAUNCHER_REL_PATH = r"software\launch_mdicom.cmd"
INPUT_REL_PATH = r"input"
BOUNDING_BOX_FILENAME = "bounding_box.json"
FINAL_REPORT_FILENAME = "report.txt"
INDICATION_FILENAME = "indication.txt"

SECTION_ORDER = [
    "examination",
    "indication",
    "technique",
    "comparison",
    "findings",
    "impression",
]
SECTION_LABELS = {
    "examination": "EXAMINATION:",
    "indication": "INDICATION:",
    "technique": "TECHNIQUE:",
    "comparison": "COMPARISON:",
    "findings": "FINDINGS:",
    "impression": "IMPRESSION:",
}
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "radiology_dicom_chest_xray_report_1"
    VARIANT_NAME: str = "base"

    @property
    def dicom_viewer_launcher_path(self) -> str:
        return rf"{self.task_dir}\{DICOM_VIEWER_LAUNCHER_REL_PATH}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\{INPUT_REL_PATH}"

    @property
    def indication_file_path(self) -> str:
        return rf"{self.input_dir}\{INDICATION_FILENAME}"

    @property
    def bounding_box_path(self) -> str:
        return rf"{self.remote_output_dir}\{BOUNDING_BOX_FILENAME}"

    @property
    def report_path(self) -> str:
        return rf"{self.remote_output_dir}\{FINAL_REPORT_FILENAME}"

    @property
    def reference_bounding_box_path(self) -> str:
        return rf"{self.reference_dir}\{BOUNDING_BOX_FILENAME}"

    @property
    def reference_report_path(self) -> str:
        return rf"{self.reference_dir}\{FINAL_REPORT_FILENAME}"

    @property
    def task_description(self) -> str:
        return f"""\
You are a board-certified radiologist working in a Windows desktop environment.

Your task is to review the provided chest radiograph study, localize one required focal finding on the PA image, and then write the final chest X-ray report.

Files and tools:
- DICOM viewer launcher: `{self.dicom_viewer_launcher_path}`
- Study folder: `{self.input_dir}`
- Indication file: `{self.indication_file_path}`
- Output directory: `{self.remote_output_dir}`

Study notes:
- The input folder contains the chest study DICOM files, including a PA view and a lateral view.
- Read `indication.txt` first.
- Launch the DICOM viewer yourself from the provided launcher.
- Use the PA image for the bounding-box task.

Bounding box task:
- On the PA image, localize the nipple-shadow-like nodular opacity on the left side of the displayed image.
- Save the bounding box JSON to `{self.bounding_box_path}`.
- The file must be valid UTF-8 JSON with this schema:
  {{
    "x1": <integer>,
    "y1": <integer>,
    "x2": <integer>,
    "y2": <integer>
  }}
- Use DICOM image coordinates, not screenshot coordinates.
- The box should tightly enclose the required focal finding.

Report task:
- Write the final report to `{self.report_path}` as plain UTF-8 text.
- Use this structure and fill every section with study-specific content:

                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 INDICATION:  <copy the indication exactly>

 TECHNIQUE:  Chest PA and lateral.

 COMPARISON:  <prior studies or "None.">

 FINDINGS:

 <one concise paragraph describing lungs, pleura, cardiomediastinal silhouette, visible upper abdomen, osseous findings, and any devices or clips>

 IMPRESSION:

 <1-3 short bullet-style or sentence-style lines summarizing the key result>

Requirements:
- Keep the report clinically coherent and self-contained.
- Mention if there is or is not focal consolidation, pleural effusion, or pneumothorax.
- If you identify nipple shadows, clips, or chronic osseous findings, mention them in the report when appropriate.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_name": self.TASK_NAME,
                "variant_name": self.VARIANT_NAME,
                "dicom_viewer_launcher_path": self.dicom_viewer_launcher_path,
                "input_dir": self.input_dir,
                "indication_file_path": self.indication_file_path,
                "bounding_box_path": self.bounding_box_path,
                "report_path": self.report_path,
                "reference_bounding_box_path": self.reference_bounding_box_path,
                "reference_report_path": self.reference_report_path,
                "bounding_box_filename": BOUNDING_BOX_FILENAME,
                "report_filename": FINAL_REPORT_FILENAME,
            }
        )
        return metadata


def _cfg_for_variant(variant_name: str) -> TaskConfig:
    return TaskConfig(VARIANT_NAME=variant_name)


config = _cfg_for_variant("base")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def _read_bbox_payload(payload: bytes) -> dict[str, int] | None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return None

    required = ("x1", "y1", "x2", "y2")
    if not all(key in data for key in required):
        return None

    try:
        bbox = {key: int(data[key]) for key in required}
    except Exception:
        return None

    if bbox["x1"] >= bbox["x2"] or bbox["y1"] >= bbox["y2"]:
        return None
    return bbox


def _bbox_within_reference(agent_bbox: dict[str, int] | None, reference_bbox: dict[str, int] | None) -> bool:
    if agent_bbox is None or reference_bbox is None:
        return False
    return (
        agent_bbox["x1"] >= reference_bbox["x1"]
        and agent_bbox["y1"] >= reference_bbox["y1"]
        and agent_bbox["x2"] <= reference_bbox["x2"]
        and agent_bbox["y2"] <= reference_bbox["y2"]
    )


def _extract_sections(report_text: str) -> dict[str, str]:
    normalized = report_text.replace("\r\n", "\n").replace("\r", "\n")
    lowered = normalized.lower()
    sections: dict[str, str] = {}

    for index, section_name in enumerate(SECTION_ORDER):
        label = SECTION_LABELS[section_name]
        start = lowered.find(label.lower())
        if start == -1:
            sections[section_name] = ""
            continue

        content_start = start + len(label)
        next_positions = []
        for following_name in SECTION_ORDER[index + 1 :]:
            following_label = SECTION_LABELS[following_name]
            pos = lowered.find(following_label.lower(), content_start)
            if pos != -1:
                next_positions.append(pos)
        end = min(next_positions) if next_positions else len(normalized)
        sections[section_name] = normalized[content_start:end].strip()

    return sections


def _report_header_present(report_text: str) -> bool:
    return bool(re.search(r"(?im)^\s*final report\s*$", report_text))


def _section_has_value(sections: dict[str, str], section_name: str) -> bool:
    return bool(sections.get(section_name, "").strip())


def _text_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


_CLINICAL_JUDGE_PROMPT = """\
You are a radiology report grader. List each key clinical finding from the \
reference report. For each, decide whether the agent's report mentions the \
same finding (different wording is acceptable).

Reference:
FINDINGS: {ref_findings}
IMPRESSION: {ref_impression}

Agent:
FINDINGS: {agent_findings}
IMPRESSION: {agent_impression}

Respond with ONLY a JSON object, no other text:
{{"findings": [{{"finding": "short description", "present": true or false}}]}}"""

STRUCTURAL_WEIGHT = 0.3
CLINICAL_WEIGHT = 0.7


_LLM_JUDGE_MODEL = "gpt-4o-mini"


def _llm_judge_clinical(
    agent_findings: str,
    agent_impression: str,
    ref_findings: str,
    ref_impression: str,
) -> tuple[float, list[dict]]:
    """Score clinical content by checking finding coverage with a small LLM."""
    prompt = _CLINICAL_JUDGE_PROMPT.format(
        ref_findings=ref_findings,
        ref_impression=ref_impression,
        agent_findings=agent_findings,
        agent_impression=agent_impression,
    )
    try:  # load secret/eval_time/*.env so the OpenAI judge key is present
        from tasks.utils.evaluation import load_eval_env

        load_eval_env()
    except Exception:
        pass
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_LLM_JUDGE_MODEL,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        checks = data["findings"]
        if not checks:
            return 0.0, checks
        score = sum(1.0 for c in checks if c.get("present")) / len(checks)
        return score, checks
    except Exception as exc:
        logger.warning("LLM judge failed (%s); falling back to text similarity", exc)
        f_sim = _text_similarity(agent_findings, ref_findings)
        i_sim = _text_similarity(agent_impression, ref_impression)
        return (f_sim + i_sim) / 2, []


def _score_report(report_text: str, indication_text: str, reference_report_text: str) -> tuple[float, dict[str, Any]]:
    if _normalize_text(report_text) == _normalize_text(reference_report_text):
        return 1.0, {"exact_reference_match": 1.0}

    sections = _extract_sections(report_text)
    reference_sections = _extract_sections(reference_report_text)
    breakdown: dict[str, Any] = {}

    # --- structural metrics (averaged, then weighted at STRUCTURAL_WEIGHT) ---
    structural_scores: list[float] = []

    breakdown["header"] = 1.0 if _report_header_present(report_text) else 0.0
    structural_scores.append(breakdown["header"])

    comparable_sections = [s for s in SECTION_ORDER if _section_has_value(reference_sections, s)]
    if comparable_sections:
        breakdown["section_coverage"] = sum(
            1.0 if _section_has_value(sections, s) else 0.0 for s in comparable_sections
        ) / len(comparable_sections)
        structural_scores.append(breakdown["section_coverage"])

    breakdown["indication_exact"] = 1.0 if _normalize_text(sections.get("indication", "")) == _normalize_text(indication_text) else 0.0
    structural_scores.append(breakdown["indication_exact"])

    for name in ("examination", "technique", "comparison"):
        ref_val = reference_sections.get(name, "")
        if ref_val.strip():
            sim = _text_similarity(sections.get(name, ""), ref_val)
            breakdown[f"{name}_similarity"] = sim
            structural_scores.append(sim)

    structural_avg = sum(structural_scores) / len(structural_scores) if structural_scores else 0.0
    breakdown["structural_avg"] = structural_avg

    # --- clinical content via LLM judge (weighted at CLINICAL_WEIGHT) ---
    clinical_score, clinical_checks = _llm_judge_clinical(
        sections.get("findings", ""),
        sections.get("impression", ""),
        reference_sections.get("findings", ""),
        reference_sections.get("impression", ""),
    )
    breakdown["clinical_score"] = clinical_score
    breakdown["clinical_checks"] = clinical_checks

    report_score = structural_avg * STRUCTURAL_WEIGHT + clinical_score * CLINICAL_WEIGHT
    return report_score, breakdown


def score_output_payloads(
    *,
    agent_bbox_payload: bytes | None,
    agent_report_payload: bytes | None,
    reference_bbox_payload: bytes | None,
    reference_report_payload: bytes | None,
    indication_payload: bytes | None,
) -> dict[str, Any]:
    agent_bbox = _read_bbox_payload(agent_bbox_payload) if agent_bbox_payload else None
    reference_bbox = _read_bbox_payload(reference_bbox_payload) if reference_bbox_payload else None

    bbox_score = 1.0 if _bbox_within_reference(agent_bbox, reference_bbox) else 0.0

    indication_text = indication_payload.decode("utf-8", errors="ignore").strip() if indication_payload else ""
    report_text = agent_report_payload.decode("utf-8", errors="ignore") if agent_report_payload else ""
    reference_report_text = reference_report_payload.decode("utf-8", errors="ignore") if reference_report_payload else ""
    report_score, report_breakdown = (
        _score_report(report_text, indication_text, reference_report_text) if report_text and reference_report_text else (0.0, {})
    )

    final_score = (bbox_score * 0.5) + (report_score * 0.5)
    return {
        "bbox_score": bbox_score,
        "report_score": report_score,
        "final_score": final_score,
        "report_breakdown": report_breakdown,
        "agent_bbox_valid": agent_bbox is not None,
        "reference_bbox_valid": reference_bbox is not None,
    }


def score_output_dir(output_dir: Path, reference_dir: Path, indication_path: Path) -> dict[str, Any]:
    agent_bbox_path = output_dir / BOUNDING_BOX_FILENAME
    agent_report_path = output_dir / FINAL_REPORT_FILENAME
    reference_bbox_path = reference_dir / BOUNDING_BOX_FILENAME
    reference_report_path = reference_dir / FINAL_REPORT_FILENAME

    return score_output_payloads(
        agent_bbox_payload=agent_bbox_path.read_bytes() if agent_bbox_path.exists() else None,
        agent_report_payload=agent_report_path.read_bytes() if agent_report_path.exists() else None,
        reference_bbox_payload=reference_bbox_path.read_bytes() if reference_bbox_path.exists() else None,
        reference_report_payload=reference_report_path.read_bytes() if reference_report_path.exists() else None,
        indication_payload=indication_path.read_bytes() if indication_path.exists() else None,
    )


@cb.tasks_config(split="train")
def load():
    tasks = []
    for variant_name in VARIANT_NAMES:
        cfg = _cfg_for_variant(variant_name)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={
                    "provider": "computer",
                    "setup_config": {
                        "os_type": cfg.OS_TYPE,
                    },
                },
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    metadata = task_cfg.metadata

    async def _maybe_read(path: str) -> bytes | None:
        try:
            return await session.read_bytes(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    result = score_output_payloads(
        agent_bbox_payload=await _maybe_read(metadata["bounding_box_path"]),
        agent_report_payload=await _maybe_read(metadata["report_path"]),
        reference_bbox_payload=await _maybe_read(metadata["reference_bounding_box_path"]),
        reference_report_payload=await _maybe_read(metadata["reference_report_path"]),
        indication_payload=await _maybe_read(metadata["indication_file_path"]),
    )

    logger.info("Radiology evaluation result: %s", result)
    return [float(result["final_score"])]
