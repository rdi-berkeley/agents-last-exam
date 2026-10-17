import hashlib
import os
from pathlib import Path

import pytest

from tasks.health_medicine.prostate_imrt_matrad_reproduction.scripts import evaluate


TOPICS = {"beam summary", "dvh", "constraint compliance"}
TABLE_HEADER = "| Structure | Metric | Value | Limit | Status |\n|---|---|---|---|---|"
REPORT = """# Research plan

Case: 7-beam coplanar 6 MV IMRT, 78 Gy in 39 fractions.

## 2. Plan

Gantry angles: 0/51/102/153/204/255/306 degrees.

## 4. Final DVH (independent recompute)

| Structure | Metric | Value | Limit | Status |
|---|---|---|---|---|
| PTV_7800 | V95% | 96.5 % | >=95 % | PASS |
| PTV_7800 | V107% | 1.2 % | <=2 % | PASS |
| PTV_7800 | Dmax | 82 Gy | <=83 Gy | PASS |
| Rectum | V70Gy | 18 % | <=20 % | PASS |
| Rectum | V50Gy | 44 % | <=50 % | PASS |
| Bladder | V70Gy | 24 % | <=25 % | PASS |
| Bladder | V50Gy | 50 % | <=65 % | PASS |
| FemHead L/R | V50Gy | 0.2 % | <=5 % | PASS |
| Penile bulb | Mean | 30 Gy | <=52.5 Gy | PASS |
"""


@pytest.mark.parametrize(
    "heading",
    [
        "## DVH",
        "## Independent DVH",
        "## Final DVH",
        "## 4. Final DVH (RTDOSE / RTSTRUCT)",
        "### 4) Final independent DVH results",
        "## 4.1 Quantitative DVH summary",
        "## Dose-volume histogram",
        "## FINAL DOSE VOLUME HISTOGRAMS",
    ],
)
def test_equivalent_quantitative_headings(heading):
    report = REPORT.replace("## 4. Final DVH (independent recompute)", heading)
    assert evaluate.report_section_coverage(report) == TOPICS


@pytest.mark.parametrize(
    "structure",
    [
        "FemHead L/R",
        "FemHead_L / R",
        "femoral heads (left/right)",
        "Femoral heads L & R",
        "Femoral heads left and right",
    ],
)
def test_bilateral_summary_with_one_explicit_shared_value(structure):
    report = REPORT.replace("FemHead L/R", structure)
    assert evaluate.report_section_coverage(report) == TOPICS


@pytest.mark.parametrize(
    "case",
    [
        "7-beam\ncoplanar\n6 MV IMRT",
        "seven coplanar 6.0 MV beams",
        "7 coplanar 6MV IMRT fields",
    ],
)
def test_beam_facts_across_lines_and_sections(case):
    report = REPORT.replace("7-beam coplanar 6 MV IMRT", case)
    assert evaluate.report_section_coverage(report) == TOPICS


@pytest.mark.parametrize(
    "angles",
    [
        "0, 51, 102, 153, 204, 255,\n306 degrees",
        "0°/51°/102°/153°/204°/255°/306°",
        "0 deg, 51 deg, 102 deg, 153 deg, 204 deg, 255 deg, 306 deg",
    ],
)
def test_equivalent_angular_units(angles):
    report = REPORT.replace("0/51/102/153/204/255/306 degrees", angles)
    assert evaluate.report_section_coverage(report) == TOPICS


@pytest.mark.parametrize(
    "before,after",
    [
        ("7-beam", "6-beam"),
        ("7-beam", ""),
        ("6 MV", ""),
        ("6 MV", "10 MV"),
        ("Gantry angles:", "Values:"),
        ("0/51/102/153/204/255/306", "0/51/102/153/204/255"),
        ("0/51/102/153/204/255/306", "0/51/102/153/204/255/306/350"),
    ],
)
def test_missing_or_wrong_beam_context_does_not_supply_summary(before, after):
    assert "beam summary" not in evaluate.report_section_coverage(REPORT.replace(before, after))


@pytest.mark.parametrize(
    "before,after",
    [
        ("| Penile bulb | Mean | 30 Gy | <=52.5 Gy | PASS |", ""),
        ("FemHead L/R", "FemHead_L"),
        ("FemHead L/R", "FemHead_R"),
        ("FemHead L/R", "FemHead L/unknown"),
        ("0.2 %", "0.2/0.3 %"),
        ("0.2 %", "nan %"),
        ("0.2 %", "1e999 %"),
        ("0.2 %", ""),
        ("0.2 %", "0.2"),
        ("0.2 %", "0.2 Gy"),
        ("<=5 %", "<=5 Gy"),
        ("<=5 %", ">=5 %"),
        ("<=5 %", "<=50 %"),
        ("<=5 %", ""),
        ("| <=5 % | PASS |", "| <=5 % | |"),
        ("## 4. Final DVH (independent recompute)", "## Limitations"),
    ],
)
def test_implicit_compliance_preserves_required_content(before, after):
    assert "constraint compliance" not in evaluate.report_section_coverage(
        REPORT.replace(before, after)
    )


def test_empty_final_dvh_and_unrelated_topic_mentions_do_not_supply_compliance():
    assert evaluate.report_section_coverage("## 4. Final DVH") == {"dvh"}
    assert not evaluate.report_section_coverage("## Missing beam summary DVH constraint compliance")
    assert not evaluate.report_section_coverage("```markdown\n" + REPORT + "```\n")


def test_units_can_be_in_their_own_column_but_cannot_conflict():
    lines = []
    for line in REPORT.splitlines():
        if line.startswith("| Structure"):
            line = line.replace("| Value |", "| Value | Units |")
        elif line.startswith("|---"):
            line += "---|"
        elif line.startswith("|"):
            cells = line.split("|")
            number, units = cells[3].split()
            cells[3:4] = [f" {number} ", f" {units} "]
            line = "|".join(cells)
        lines.append(line)
    report = "\n".join(lines)
    assert evaluate.report_section_coverage(report) == TOPICS
    assert "constraint compliance" not in evaluate.report_section_coverage(
        report.replace("| 0.2 | % |", "| 0.2 | Gy |")
    )
    assert "constraint compliance" not in evaluate.report_section_coverage(
        report.replace("| 0.2 | % |", "| 0.2 % | Gy |")
    )


@pytest.mark.parametrize("outside_dvh", [False, True])
def test_dvh_subsections_combine_only_within_dvh_context(outside_dvh):
    report = REPORT.replace(TABLE_HEADER, "### Target\n\n" + TABLE_HEADER)
    heading = "## Limitations" if outside_dvh else "### Organs at risk"
    report = report.replace(
        "| Rectum | V70Gy", heading + "\n\n" + TABLE_HEADER + "\n| Rectum | V70Gy"
    )
    covered = evaluate.report_section_coverage(report)
    assert ("constraint compliance" in covered) is not outside_dvh


@pytest.fixture(params=["codex", "kimi"])
def actual_report_directory(request):
    configured = os.environ.get(f"PROSTATE_{request.param.upper()}_REPORT_DIR")
    if configured is None:
        pytest.skip(
            "set PROSTATE_CODEX_REPORT_DIR and PROSTATE_KIMI_REPORT_DIR for retained controls"
        )
    directory = Path(configured)
    assert (directory / "report.md").is_file()
    return directory


def test_actual_retained_reports_receive_all_presentation_credit(actual_report_directory):
    report = actual_report_directory / "report.md"
    original_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    assert evaluate.report_section_coverage(report.read_text()) == TOPICS
    assert evaluate.gate_G8_report_structure(actual_report_directory) == (4, [])
    assert hashlib.sha256(report.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("damage", ["missing_table", "missing_metric", "wrong_unit"])
def test_actual_report_negative_controls(actual_report_directory, damage):
    report = (actual_report_directory / "report.md").read_text()
    if damage == "missing_table":
        report = "\n".join(line for line in report.splitlines() if not line.startswith("|"))
    elif damage == "missing_metric":
        report = "\n".join(line for line in report.splitlines() if "| Bladder | V70" not in line)
    else:
        lines = report.splitlines()
        for index, line in enumerate(lines):
            if "| Bladder | V70" in line:
                cells = line.split("|")
                cells[3] = cells[3].replace("%", "Gy")
                lines[index] = "|".join(cells)
        report = "\n".join(lines)
    assert "constraint compliance" not in evaluate.report_section_coverage(report)
