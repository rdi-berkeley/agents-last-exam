#!/usr/bin/env python
"""VM-side verifier for health_medicine/repair_vxu_immunization_reporting_interface.

Computes the 100-point deterministic rubric:

    artifact_completeness        10
    public_output_exactness      20
    public_replay_consistency    20
    hidden_replay                30
    channel_export_structure     10
    documentation_quality        10

Prints a single JSON object on stdout; debug logs go to stderr. Exits 0 even
when the submission fails, because the eval harness expects to parse the JSON.

Usage:
    python verify_submission.py --submission-dir <path> --reference-dir <path> \\
        --input-dir <path> --work-dir <path>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


REQUIRED_SUBMISSION_FILES = [
    "channel_export/repair_vxu_immunization_reporting_interface.xml",
    "code_templates/transform_vxu.js",
    "code_templates/ack_router.js",
    "code_templates/batch_export.js",
    "specs/mappings.json",
    "specs/facility_map.json",
    "specs/jurisdiction_rules.json",
    "deployment/export_config.json",
    "generated_messages/public_case_01_vxu.hl7",
    "generated_messages/public_case_02_vxu.hl7",
    "generated_messages/public_case_03_vxu.hl7",
    "acks/public_case_01_ack.hl7",
    "acks/public_case_02_ack.hl7",
    "acks/public_case_03_ack.hl7",
    "batch/public_batch_20260417.hl7",
    "batch/public_export_manifest.json",
    "ROOT_CAUSE_ANALYSIS.md",
    "SHIP_DECISION.md",
]

PUBLIC_EXACT_MATCH_FILES = [
    "generated_messages/public_case_01_vxu.hl7",
    "generated_messages/public_case_02_vxu.hl7",
    "generated_messages/public_case_03_vxu.hl7",
    "acks/public_case_01_ack.hl7",
    "acks/public_case_02_ack.hl7",
    "acks/public_case_03_ack.hl7",
    "batch/public_batch_20260417.hl7",
    "batch/public_export_manifest.json",
]

PUBLIC_REPLAY_ARTIFACTS = [
    "generated_messages/public_case_01_vxu.hl7",
    "generated_messages/public_case_02_vxu.hl7",
    "generated_messages/public_case_03_vxu.hl7",
    "acks/public_case_01_ack.hl7",
    "acks/public_case_02_ack.hl7",
    "acks/public_case_03_ack.hl7",
    "batch/public_batch_20260417.hl7",
    "batch/public_export_manifest.json",
]

HIDDEN_REPLAY_ARTIFACTS = [
    "generated_messages/hidden_case_01_vxu.hl7",
    "generated_messages/hidden_case_02_vxu.hl7",
    "generated_messages/hidden_case_03_vxu.hl7",
    "generated_messages/hidden_case_04_vxu.hl7",
    "generated_messages/hidden_case_05_vxu.hl7",
    "generated_messages/hidden_case_06_vxu.hl7",
    "acks/hidden_case_01_ack.hl7",
    "acks/hidden_case_02_ack.hl7",
    "acks/hidden_case_03_ack.hl7",
    "acks/hidden_case_04_ack.hl7",
    "acks/hidden_case_05_ack.hl7",
    "acks/hidden_case_06_ack.hl7",
    "batch/public_batch_20260417.hl7",
    "batch/public_export_manifest.json",
]

FIXED_EXPORTED_AT = "2026-04-17T12:00:00Z"


@dataclass
class SectionScore:
    name: str
    earned: float
    max: float
    notes: list[str] = field(default_factory=list)


def normalize_lines(data: bytes) -> bytes:
    """Line-ending normalization: strip carriage returns so CRLF == LF == CR."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _json_normalize_equal(a: bytes, b: bytes) -> bool:
    try:
        return json.loads(a.decode("utf-8")) == json.loads(b.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False


def compare_artifact(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    if a.suffix.lower() == ".json":
        return _json_normalize_equal(_read_bytes(a), _read_bytes(b))
    return normalize_lines(_read_bytes(a)) == normalize_lines(_read_bytes(b))


def score_artifact_completeness(submission_dir: Path) -> SectionScore:
    total_points = 10.0
    per = total_points / len(REQUIRED_SUBMISSION_FILES)
    earned = 0.0
    missing: list[str] = []
    for rel in REQUIRED_SUBMISSION_FILES:
        if (submission_dir / rel).exists():
            earned += per
        else:
            missing.append(rel)
    notes = []
    if missing:
        notes.append(f"missing {len(missing)}/{len(REQUIRED_SUBMISSION_FILES)}: {missing[:3]}")
    return SectionScore("artifact_completeness", min(earned, total_points), total_points, notes)


def score_public_exactness(submission_dir: Path, reference_dir: Path) -> SectionScore:
    total_points = 20.0
    per = total_points / len(PUBLIC_EXACT_MATCH_FILES)
    earned = 0.0
    mismatches: list[str] = []
    public_gold = reference_dir / "public_gold"
    for rel in PUBLIC_EXACT_MATCH_FILES:
        if compare_artifact(submission_dir / rel, public_gold / rel):
            earned += per
        else:
            mismatches.append(rel)
    return SectionScore(
        "public_output_exactness",
        min(earned, total_points),
        total_points,
        [f"mismatched: {mismatches}"] if mismatches else [],
    )


def run_replay(
    node_path: str,
    runner_path: Path,
    templates_dir: Path,
    specs_dir: Path,
    cases_dir: Path,
    output_dir: Path,
    exported_at: str,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        node_path,
        str(runner_path),
        str(templates_dir),
        str(specs_dir),
        str(cases_dir),
        str(output_dir),
        exported_at,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, f"replay spawn failed: {exc}"
    if result.returncode != 0:
        return False, f"replay rc={result.returncode} stderr={result.stderr[:400]}"
    return True, "ok"


def prepare_replay_stage(submission_dir: Path, stage_dir: Path) -> None:
    # The harness resolves deployment/export_config.json as ../deployment
    # relative to specs_dir, so code_templates/specs/deployment must be sibling
    # directories under a common root. We stage copies of the submitted ones.
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("code_templates", "specs", "deployment"):
        src = submission_dir / sub
        if not src.exists():
            raise FileNotFoundError(f"submission missing required directory: {sub}")
        shutil.copytree(src, stage_dir / sub)


def score_replay_against_gold(
    replay_dir: Path, gold_dir: Path, artifact_rels: list[str], total_points: float
) -> tuple[float, list[str]]:
    per = total_points / len(artifact_rels)
    earned = 0.0
    mismatches: list[str] = []
    for rel in artifact_rels:
        if compare_artifact(replay_dir / rel, gold_dir / rel):
            earned += per
        else:
            mismatches.append(rel)
    return min(earned, total_points), mismatches


def score_public_replay_consistency(
    submission_dir: Path,
    input_dir: Path,
    reference_dir: Path,
    work_dir: Path,
    node_path: str,
) -> SectionScore:
    total_points = 20.0
    name = "public_replay_consistency"
    stage_dir = work_dir / "public_replay_stage"
    replay_out = work_dir / "public_replay_out"
    runner = input_dir / "starter_project" / "runtime" / "run_cases.js"
    if not runner.exists():
        return SectionScore(name, 0.0, total_points, [f"runner missing: {runner}"])

    try:
        prepare_replay_stage(submission_dir, stage_dir)
    except FileNotFoundError as exc:
        return SectionScore(name, 0.0, total_points, [str(exc)])

    if replay_out.exists():
        shutil.rmtree(replay_out)
    replay_out.mkdir(parents=True, exist_ok=True)

    ok, detail = run_replay(
        node_path,
        runner,
        stage_dir / "code_templates",
        stage_dir / "specs",
        input_dir / "starter_project" / "public_cases",
        replay_out,
        FIXED_EXPORTED_AT,
    )
    if not ok:
        return SectionScore(name, 0.0, total_points, [detail])

    # Consistency: replayed outputs must match the *submitted* public outputs.
    earned, mismatches = score_replay_against_gold(
        replay_out, submission_dir, PUBLIC_REPLAY_ARTIFACTS, total_points
    )
    notes = []
    if mismatches:
        notes.append(f"replay-vs-submission mismatches: {mismatches}")
    return SectionScore(name, earned, total_points, notes)


def score_hidden_replay(
    submission_dir: Path,
    reference_dir: Path,
    work_dir: Path,
    node_path: str,
) -> SectionScore:
    total_points = 30.0
    name = "hidden_replay"
    hidden_cases_dir = reference_dir / "hidden_cases" / "cases"
    hidden_gold_dir = reference_dir / "hidden_cases" / "gold"
    runner = reference_dir.parent / "input" / "starter_project" / "runtime" / "run_cases.js"
    if not hidden_cases_dir.exists() or not hidden_gold_dir.exists():
        return SectionScore(name, 0.0, total_points, [f"hidden fixtures missing under {reference_dir}"])
    if not runner.exists():
        return SectionScore(name, 0.0, total_points, [f"runner missing: {runner}"])

    stage_dir = work_dir / "hidden_replay_stage"
    replay_out = work_dir / "hidden_replay_out"
    try:
        prepare_replay_stage(submission_dir, stage_dir)
    except FileNotFoundError as exc:
        return SectionScore(name, 0.0, total_points, [str(exc)])

    if replay_out.exists():
        shutil.rmtree(replay_out)
    replay_out.mkdir(parents=True, exist_ok=True)

    ok, detail = run_replay(
        node_path,
        runner,
        stage_dir / "code_templates",
        stage_dir / "specs",
        hidden_cases_dir,
        replay_out,
        FIXED_EXPORTED_AT,
    )
    if not ok:
        return SectionScore(name, 0.0, total_points, [detail])

    earned, mismatches = score_replay_against_gold(
        replay_out, hidden_gold_dir, HIDDEN_REPLAY_ARTIFACTS, total_points
    )
    notes = []
    if mismatches:
        notes.append(f"hidden replay mismatches: {mismatches}")
    return SectionScore(name, earned, total_points, notes)


def score_channel_export_structure(submission_dir: Path) -> SectionScore:
    total_points = 10.0
    name = "channel_export_structure"
    path = submission_dir / "channel_export" / "repair_vxu_immunization_reporting_interface.xml"
    if not path.exists():
        return SectionScore(name, 0.0, total_points, ["channel export missing"])

    try:
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
    except (ET.ParseError, UnicodeDecodeError) as exc:
        return SectionScore(name, 0.0, total_points, [f"XML parse failed: {exc}"])

    checks = {
        "root_is_channel": root.tag == "channel",
        "source_connector_present": root.find("sourceConnector") is not None,
        "destination_connectors_present": root.find("destinationConnectors") is not None,
        "has_batch_export_connector": any(
            (conn.findtext("name") or "").strip() == "Batch Export"
            for conn in root.findall(".//destinationConnectors/connector")
        ),
        "has_retry_queue_connector": any(
            (conn.findtext("name") or "").strip() == "Retry Queue"
            for conn in root.findall(".//destinationConnectors/connector")
        ),
        "has_quarantine_queue_connector": any(
            (conn.findtext("name") or "").strip() == "Quarantine Queue"
            for conn in root.findall(".//destinationConnectors/connector")
        ),
        "description_mentions_templates": bool(
            re.search(
                r"transform_vxu\.js.*ack_router\.js.*batch_export\.js",
                root.findtext("description") or "",
                flags=re.S,
            )
        ),
        "hl7v2_outbound_datatype": any(
            (elem.text or "").strip() == "HL7V2"
            for elem in root.findall(".//outboundDataType")
        ),
        "ack_codes_configured": bool(
            root.find(".//successfulACKCode") is not None
            and root.find(".//errorACKCode") is not None
            and root.find(".//rejectedACKCode") is not None
        ),
    }
    passed = sum(1 for v in checks.values() if v)
    earned = total_points * (passed / len(checks))
    notes = [f"{name}: {passed}/{len(checks)} structural checks"]
    failed = [k for k, v in checks.items() if not v]
    if failed:
        notes.append(f"failed: {failed}")
    return SectionScore(name, earned, total_points, notes)


def score_documentation(submission_dir: Path) -> SectionScore:
    total_points = 10.0
    name = "documentation_quality"
    rca_path = submission_dir / "ROOT_CAUSE_ANALYSIS.md"
    ship_path = submission_dir / "SHIP_DECISION.md"
    earned = 0.0
    notes: list[str] = []
    if not rca_path.exists():
        notes.append("ROOT_CAUSE_ANALYSIS.md missing")
    else:
        rca = rca_path.read_text(encoding="utf-8", errors="replace")
        has_root_causes_header = bool(re.search(r"(?mi)^\s*##\s+Root Causes\s*$", rca))
        has_impl_delta_header = bool(re.search(r"(?mi)^\s*##\s+Implementation Delta\s*$", rca))
        # Count list bullets in Root Causes section
        root_causes_section = ""
        m = re.search(r"(?is)^\s*##\s+Root Causes\s*$(.*?)(?=^\s*##\s+|\Z)", rca, flags=re.M)
        if m:
            root_causes_section = m.group(1)
        # A root cause bullet = numbered list item or "-" item, at least 3 total
        root_cause_bullets = re.findall(r"(?m)^\s*(?:-|\d+\.)\s+\S", root_causes_section)
        root_cause_count_ok = len(root_cause_bullets) >= 3
        # Cited artifacts: count filenames with common task extensions
        cited_artifacts = set(
            re.findall(
                r"[\w./_-]+\.(?:js|xml|json|hl7|md)",
                rca,
            )
        )
        cited_artifact_count_ok = len(cited_artifacts) >= 3

        rca_points = 5.0
        rca_checks = [
            has_root_causes_header,
            has_impl_delta_header,
            root_cause_count_ok,
            cited_artifact_count_ok,
        ]
        earned += rca_points * (sum(1 for c in rca_checks if c) / len(rca_checks))
        rca_notes = (
            f"rca: header_root_causes={has_root_causes_header} "
            f"header_impl_delta={has_impl_delta_header} "
            f"root_cause_bullets={len(root_cause_bullets)} "
            f"cited_artifacts={len(cited_artifacts)}"
        )
        notes.append(rca_notes)

    if not ship_path.exists():
        notes.append("SHIP_DECISION.md missing")
    else:
        ship = ship_path.read_text(encoding="utf-8", errors="replace")
        has_decision = bool(re.search(r"(?mi)^\s*##\s+Decision\s*$", ship))
        has_justification = bool(re.search(r"(?mi)^\s*##\s+Justification\s*$", ship))
        has_residual_risks = bool(re.search(r"(?mi)^\s*##\s+Residual Risks\s*$", ship))
        has_recommended_follow_up = bool(
            re.search(r"(?mi)^\s*##\s+Recommended\s+Follow-?up\s*$", ship)
        )
        ship_points = 5.0
        ship_checks = [has_decision, has_justification, has_residual_risks, has_recommended_follow_up]
        earned += ship_points * (sum(1 for c in ship_checks if c) / len(ship_checks))
        notes.append(
            f"ship: decision={has_decision} justification={has_justification} "
            f"residual={has_residual_risks} follow_up={has_recommended_follow_up}"
        )

    return SectionScore(name, min(earned, total_points), total_points, notes)


def hard_gate_fail(submission_dir: Path) -> str | None:
    # hard gate 1: any of the 18 required files missing
    missing = [rel for rel in REQUIRED_SUBMISSION_FILES if not (submission_dir / rel).exists()]
    if missing:
        return f"missing required artifacts: {missing}"
    # hard gate 2: channel export must parse as XML with basic connector refs
    chan = submission_dir / "channel_export" / "repair_vxu_immunization_reporting_interface.xml"
    try:
        root = ET.fromstring(chan.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        return f"channel export not parseable: {exc}"
    names = {(conn.findtext("name") or "").strip() for conn in root.findall(".//destinationConnectors/connector")}
    required_connectors = {"Batch Export", "Retry Queue", "Quarantine Queue"}
    if not required_connectors.issubset(names):
        return f"channel export missing required connectors: {sorted(required_connectors - names)}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--node-path", default="node")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    reference_dir = Path(args.reference_dir)
    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sections: list[SectionScore] = []
    gate = hard_gate_fail(submission_dir)
    if gate is not None:
        sections.append(SectionScore("hard_gate", 0.0, 0.0, [gate]))
        sections.append(score_artifact_completeness(submission_dir))
        payload = {
            "normalized_score": 0.0,
            "total_score": sections[1].earned,
            "max_score": 100.0,
            "passed": False,
            "hard_gate": gate,
            "sections": [section_to_dict(s) for s in sections],
        }
        print(json.dumps(payload))
        return 0

    sections.append(score_artifact_completeness(submission_dir))
    sections.append(score_public_exactness(submission_dir, reference_dir))
    sections.append(
        score_public_replay_consistency(
            submission_dir, input_dir, reference_dir, work_dir, args.node_path
        )
    )
    sections.append(
        score_hidden_replay(submission_dir, reference_dir, work_dir, args.node_path)
    )
    sections.append(score_channel_export_structure(submission_dir))
    sections.append(score_documentation(submission_dir))

    total_earned = sum(s.earned for s in sections)
    total_max = 100.0
    normalized = total_earned / total_max if total_max > 0 else 0.0

    # Debug to stderr
    print(json.dumps([section_to_dict(s) for s in sections], indent=2), file=sys.stderr)

    payload = {
        "normalized_score": round(normalized, 6),
        "total_score": round(total_earned, 4),
        "max_score": total_max,
        "passed": normalized >= 0.6,
        "sections": [section_to_dict(s) for s in sections],
    }
    print(json.dumps(payload))
    return 0


def section_to_dict(s: SectionScore) -> dict:
    return {
        "name": s.name,
        "earned": round(s.earned, 4),
        "max": s.max,
        "notes": s.notes,
    }


if __name__ == "__main__":
    sys.exit(main())
