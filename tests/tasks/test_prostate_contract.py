import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pydicom
import pytest


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/health_medicine/prostate_imrt_matrad_reproduction"
DATA = Path(
    os.environ.get(
        "PROSTATE_DATA_ROOT",
        ROOT / "task-data-hf/extracted/health_medicine/prostate_imrt_matrad_reproduction/base",
    )
)
LOGS = Path(os.environ.get("PROSTATE_LOG_ROOT", ROOT.parent / "unified-logs-staging/claude_code"))
GPT6_LOGS = Path(
    os.environ.get(
        "PROSTATE_GPT6_LOG_ROOT",
        ROOT.parent
        / "task-fairness-gpt6-20260907/runs/health_medicine__prostate_imrt_matrad_reproduction"
        / "20260907_102636/logs"
        / "fairness-contract-gpt6-high-health_medicine__prostate_imrt_matrad_reproduction-20260907_102636"
        / "codex/gpt-6-astra/health_medicine__prostate_imrt_matrad_reproduction/v0/20260907_102639",
    )
)
CURRENT_GPT6_LOGS = Path(
    os.environ.get(
        "PROSTATE_CURRENT_LOG_ROOT",
        ROOT.parent
        / "task-fairness-gpt6-20260907/runs/health_medicine__prostate_imrt_matrad_reproduction"
        / "20260911_182633/logs"
        / "fairness-all-fixes-fresh-gpt6-high-health_medicine__prostate_imrt_matrad_reproduction-20260911_182633"
        / "codex/gpt-6-astra/health_medicine__prostate_imrt_matrad_reproduction/v0/20260911_182635",
    )
)
COLUMNS = ["structure", "metric_type", "metric_value", "units", "constraint_pass"]


@pytest.fixture(autouse=True)
def prevent_reference_bytecode(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)


@pytest.fixture(scope="module")
def scorer():
    spec = importlib.util.spec_from_file_location("prostate_evaluate", TASK / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trusted_absent():
    reference = pydicom.Dataset()
    rectum = pydicom.Dataset()
    rectum.ROIName = "Rectum"
    reference.StructureSetROISequence = [rectum]
    return reference


@pytest.fixture
def computed(scorer):
    return {
        key: float(spec["limit"])
        for key, spec in scorer.DVH_CONSTRAINTS.items()
        if key != "Bowel_Dmax"
    }


def rows_for(computed):
    return [
        dict(
            zip(
                COLUMNS,
                [
                    *key.rsplit("_", 1),
                    str(value),
                    "%" if key.rsplit("_", 1)[1].startswith("V") else "Gy",
                    "PASS",
                ],
            )
        )
        for key, value in computed.items()
    ]


def write_csv(path, rows, columns=COLUMNS, **options):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, **options)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "representation",
    [
        "canonical",
        "quoted",
        "scientific",
        "bom",
        "reordered",
        "whitespace",
        "aliases",
        "annotated",
        "extras",
        "na",
        "n/a",
    ],
)
def test_semantic_csv_equivalence(tmp_path, scorer, computed, trusted_absent, representation):
    rows = rows_for(computed)
    columns = COLUMNS
    options = {}
    if representation == "quoted":
        options["quoting"] = csv.QUOTE_ALL
    elif representation == "scientific":
        for row in rows:
            row["metric_value"] = f"{float(row['metric_value']):.10e}"
    elif representation == "reordered":
        rows.reverse()
        columns = list(reversed(COLUMNS))
    elif representation == "whitespace":
        for row in rows:
            for column in COLUMNS:
                row[column] = " " + row[column] + " "
    elif representation == "aliases":
        aliases = {
            "PTV_7800": "ptv",
            "FemHead_L": "Lt femoral head",
            "FemHead_R": "Right femoral head",
            "PenileBulb": "Penile bulb",
        }
        for row in rows:
            row["structure"] = aliases.get(row["structure"], row["structure"].lower())
            if row["metric_type"].startswith("V"):
                row["metric_type"] += "%" if row["structure"] == "ptv" else "Gy"
            elif row["metric_type"] == "mean":
                row["metric_type"] = "MeanDose"
    elif representation == "annotated":
        for row, (key, spec) in zip(rows, ((key, scorer.DVH_CONSTRAINTS[key]) for key in computed)):
            row["metric_type"] += f" ({spec['limit_op']}{spec['limit']}{row['units']})"
    elif representation == "extras":
        rows += [dict(zip(COLUMNS, ["PTV_7800", "D95", "78", "Gy", "PASS"]))]
    elif representation in {"na", "n/a"}:
        rows += [dict(zip(COLUMNS, ["Bowel bag", "Dmax", representation, "Gy", representation]))]
    path = write_csv(tmp_path / "metrics.csv", rows, columns, **options)
    if representation == "bom":
        path.write_text("\ufeff" + path.read_text())
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, trusted_absent) == (4, [])


@pytest.mark.parametrize(
    "structure,metric,expected",
    [
        ("PTV_68", "V95% (>=95%)", "PTV_7800_V95"),
        ("PTV_Prostate_7800", "V107% (<=2%)", "PTV_7800_V107"),
        ("PTV_7800", "MaximumDose", "PTV_7800_Dmax"),
        ("left-femoral-head", "V50Gy", "FemHead_L_V50"),
        ("FemHead R", "V50", "FemHead_R_V50"),
        ("Penile_bulb", "Mean dose (<=52.5Gy)", "PenileBulb_mean"),
        ("PenileBulb", "Dmean", "PenileBulb_mean"),
        ("BowelBag", "Dmax", "Bowel_Dmax"),
    ],
)
def test_aliases(scorer, structure, metric, expected):
    assert scorer.csv_metric_key(structure, metric) == expected


@pytest.mark.parametrize("bound", ["95", "95.0", "95.", "+95", "9.5e1", ".95e2"])
def test_equivalent_constraint_annotation_numbers(scorer, bound):
    assert scorer.csv_metric_key("PTV_7800", f"V95% (>= {bound} %)") == "PTV_7800_V95"


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("change", ["wrong", "missing", "nan", "units", "duplicate", "na"])
def test_every_required_metric_is_checked(
    tmp_path, scorer, computed, trusted_absent, index, change
):
    rows = rows_for(computed)
    row = rows[index]
    if change == "wrong":
        row["metric_value"] = str(float(row["metric_value"]) + 0.501)
    elif change == "missing":
        rows.pop(index)
    elif change == "nan":
        row["metric_value"] = "nan"
    elif change == "units":
        row["units"] = "Gy" if row["units"] == "%" else "%"
    elif change == "duplicate":
        rows += [dict(row)]
    elif change == "na":
        row["metric_value"] = "NA"
    score, notes = scorer.gate_G7_dvh_csv_honesty(
        write_csv(tmp_path / "metrics.csv", rows), computed, trusted_absent
    )
    assert score == (2 if change == "wrong" else 0)
    assert notes


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-inf", "1e9999", "", "n/a", "true", "1/2"])
def test_malformed_numbers(tmp_path, scorer, computed, trusted_absent, value):
    rows = rows_for(computed)
    rows[0]["metric_value"] = value
    assert (
        scorer.gate_G7_dvh_csv_honesty(
            write_csv(tmp_path / "metrics.csv", rows), computed, trusted_absent
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "malformation",
    [
        "empty",
        "header_only",
        "duplicate_header",
        "short",
        "long",
        "quote",
        "unknown_only",
        "wrong_annotation",
        "semantic_duplicate",
        "extra_nan",
    ],
)
def test_malformed_csv(tmp_path, scorer, computed, trusted_absent, malformation):
    rows = rows_for(computed)
    path = write_csv(tmp_path / "metrics.csv", rows)
    text = path.read_text()
    if malformation == "empty":
        text = ""
    elif malformation == "header_only":
        text = text.splitlines()[0] + "\n"
    elif malformation == "duplicate_header":
        text = text.replace("units,constraint_pass", "units,units", 1)
    elif malformation == "short":
        text += "PTV_7800,V95\n"
    elif malformation == "long":
        text += "PTV_7800,V95,95,%,PASS,extra\n"
    elif malformation == "quote":
        text += '"unterminated\n'
    elif malformation == "unknown_only":
        text = text.splitlines()[0] + "\nAnything,Anything,1,Gy,PASS\n"
    elif malformation == "wrong_annotation":
        text = text.replace("V95,", "V95% (<=2%),", 1)
    elif malformation == "semantic_duplicate":
        text += "PTV,V95%,95,%,PASS\n"
    elif malformation == "extra_nan":
        text += "PTV,D95,NaN,Gy,PASS\n"
    path.write_text(text)
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, trusted_absent)[0] == 0


@pytest.mark.parametrize("count,expected", [(0, 4), (1, 2), (2, 2), (3, 0), (10, 0)])
def test_original_numeric_tolerance_and_partial_credit(
    tmp_path, scorer, computed, trusted_absent, count, expected
):
    rows = rows_for(computed)
    for index, row in enumerate(rows):
        row["metric_value"] = str(float(row["metric_value"]) + (0.501 if index < count else 0.5))
    assert (
        scorer.gate_G7_dvh_csv_honesty(
            write_csv(tmp_path / "metrics.csv", rows), computed, trusted_absent
        )[0]
        == expected
    )


@pytest.mark.parametrize("alias", ["Bowel", "BowelBag", "Bowel_bag", "Bowel bag"])
def test_trusted_bowel_presence_requires_measurement(scorer, trusted_absent, alias):
    bowel = pydicom.Dataset()
    bowel.ROIName = alias
    trusted_absent.StructureSetROISequence.append(bowel)
    assert not scorer.unassessed_dvh_metrics(trusted_absent)


def test_no_reference_cannot_authorize_na(tmp_path, scorer, computed, trusted_absent):
    path = write_csv(tmp_path / "metrics.csv", rows_for(computed))
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, None)[0] == 0
    assert not scorer.unassessed_dvh_metrics(pydicom.Dataset())
    computed.pop("Rectum_V70")
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, trusted_absent)[0] == 0


@pytest.mark.parametrize("value", ["0", "50", "NaN", "Infinity"])
def test_absent_bowel_cannot_claim_scientific_success(
    tmp_path, scorer, computed, trusted_absent, value
):
    rows = rows_for(computed) + [dict(zip(COLUMNS, ["Bowel", "Dmax", value, "Gy", "PASS"]))]
    assert (
        scorer.gate_G7_dvh_csv_honesty(
            write_csv(tmp_path / "metrics.csv", rows), computed, trusted_absent
        )[0]
        == 0
    )


def test_trusted_present_bowel_cannot_be_deleted_or_declared_na(
    tmp_path, scorer, computed, trusted_absent
):
    bowel = pydicom.Dataset()
    bowel.ROIName = "Bowel bag"
    trusted_absent.StructureSetROISequence.append(bowel)
    rows = rows_for(computed) + [dict(zip(COLUMNS, ["Bowel", "Dmax", "NA", "Gy", "NA"]))]
    path = write_csv(tmp_path / "metrics.csv", rows)
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, trusted_absent)[0] == 0
    rows[-1]["metric_value"] = "45"
    rows[-1]["constraint_pass"] = "PASS"
    path = write_csv(path, rows)
    assert scorer.gate_G7_dvh_csv_honesty(path, computed, trusted_absent)[0] == 0
    assert (
        scorer.gate_G7_dvh_csv_honesty(path, dict(computed, Bowel_Dmax=45), trusted_absent)[0] == 4
    )


def test_g5_anatomy_applicability_and_unchanged_weights(scorer, trusted_absent):
    dose = np.zeros((2, 2, 2))
    mask = np.ones_like(dose, dtype=bool)
    masks = {name: mask for name in scorer.ROI_ALIASES if name != "PTV_7800"}
    assert scorer.gate_G5_oar_sparing(dose, masks, trusted_absent)[0] == 23
    assert scorer.gate_G5_oar_sparing(dose, masks)[0] == 26
    masks.pop("Bowel")
    assert scorer.gate_G5_oar_sparing(dose, masks, trusted_absent)[0] == 23
    masks.pop("Rectum")
    assert scorer.gate_G5_oar_sparing(dose, masks, trusted_absent)[0] == 15
    assert scorer.gate_G5_oar_sparing(np.full_like(dose, 100), masks, trusted_absent)[0] == 0


@pytest.fixture(scope="module")
def real_reference(scorer):
    path = DATA / "reference/RTSTRUCT_gold.dcm"
    if not path.exists():
        pytest.skip("set PROSTATE_DATA_ROOT for real DICOM evidence")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "c83e34d6aadb93b5523b53bb52d75bae26284c21d7a551e35337a5f46a99f76b"
    )
    reference = scorer.load_rtstruct(path)
    original = scorer.load_rtstruct(DATA / "input/dicom/RTSTRUCT.dcm")
    assert len(reference.StructureSetROISequence) == len(original.StructureSetROISequence) == 10
    assert (
        scorer.unassessed_dvh_metrics(reference)
        == scorer.unassessed_dvh_metrics(original)
        == {"Bowel_Dmax"}
    )
    return reference


@pytest.fixture(scope="module", params=["high", "medium"])
def artifact(request, scorer, real_reference):
    paths = list(
        LOGS.glob(
            f"claude-opus-4-8_reasoning-{request.param}/health_medicine__prostate_imrt_matrad_reproduction/v0/*/output"
        )
    )
    if not paths:
        pytest.skip("set PROSTATE_LOG_ROOT for unchanged historical artifacts")
    assert len(paths) == 1
    path = paths[0]
    dose, geom = scorer.load_rtdose(path / "RTDOSE.dcm")
    masks = scorer.compute_roi_masks(scorer.load_rtstruct(path / "RTSTRUCT_corrected.dcm"), geom)
    ptv = dose[masks["PTV_7800"]]
    computed = {
        "PTV_7800_V95": 100 * np.mean(ptv >= 74.1),
        "PTV_7800_V107": 100 * np.mean(ptv >= 83.46),
        "PTV_7800_Dmax": float(np.max(ptv)),
    }
    for name, canonical in [
        ("Rectum", "Rectum"),
        ("Bladder", "Bladder"),
        ("Lt femoral head", "FemHead_L"),
        ("Rt femoral head", "FemHead_R"),
    ]:
        values = dose[masks[name]]
        for threshold in [50, 70] if canonical in {"Rectum", "Bladder"} else [50]:
            computed[f"{canonical}_V{threshold}"] = (
                100 * np.count_nonzero(values >= threshold) / values.size
            )
    computed["PenileBulb_mean"] = float(np.mean(dose[masks["Penile_bulb"]]))
    return path, dose, masks, computed


def test_unchanged_historical_dicom_and_csv_are_true_positives(scorer, artifact, real_reference):
    path, dose, masks, computed = artifact
    assert scorer.gate_G4_ptv_coverage(dose, masks)[0] == 15
    assert scorer.gate_G5_oar_sparing(dose, masks, real_reference)[0] == 23
    assert scorer.gate_G7_dvh_csv_honesty(path / "dvh_metrics.csv", computed, real_reference) == (
        4,
        [],
    )


@pytest.mark.parametrize(
    "mutation,before,after",
    [
        ("unchanged", 4, 4),
        ("na", 0, 4),
        ("wrong_all", 4, 0),
        ("ptv_wrong", 4, 0),
        ("nan", 4, 0),
        ("unknown_only", 4, 0),
    ],
)
def test_observed_old_verifier_defects(
    tmp_path, scorer, artifact, real_reference, mutation, before, after
):
    path, dose, masks, computed = artifact
    spec = importlib.util.spec_from_file_location(
        "original_prostate", DATA / "reference/evaluator_scripts/evaluate.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    old_ptv = original.gate_G4_ptv_coverage(dose, masks)[2]
    old_computed = {
        **{f"PTV_{key}": value for key, value in old_ptv.items()},
        **original.gate_G5_oar_sparing(dose, masks)[2],
    }
    with (path / "dvh_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if mutation == "na":
        rows += [dict(zip(COLUMNS, ["Bowel", "Dmax", "NA", "Gy", "NA"]))]
    elif mutation in {"wrong_all", "ptv_wrong"}:
        for row in rows:
            if mutation == "wrong_all" or row["structure"] == "PTV_7800":
                row["metric_value"] = str(float(row["metric_value"]) + 1)
    elif mutation == "nan":
        rows[0]["metric_value"] = "NaN"
    elif mutation == "unknown_only":
        rows = [dict(zip(COLUMNS, ["anything", "anything", "1", "Gy", "PASS"]))]
    changed = write_csv(tmp_path / "metrics.csv", rows)
    assert original.gate_G7_dvh_csv_honesty(changed, old_computed)[0] == before
    assert scorer.gate_G7_dvh_csv_honesty(changed, computed, real_reference)[0] == after


def test_actual_gold_structure_repair_preserves_scientific_scores(scorer, real_reference):
    spec = importlib.util.spec_from_file_location(
        "prostate_packaging", DATA / "reference/evaluator_scripts/package_submission.py"
    )
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    repaired = package.rename_external(
        package.derive_ptv_from_ctv(
            package.fix_rectum_type(scorer.load_rtstruct(DATA / "input/dicom/RTSTRUCT.dcm"))
        )
    )
    dose, geom = scorer.load_rtdose(DATA / "reference/RTDOSE_reference.dcm")
    masks = scorer.compute_roi_masks(repaired, geom)
    stored = json.loads((DATA / "reference/score-gold.json").read_text())
    assert stored["total_score"] == 85 and stored["pass"]
    assert (
        scorer.gate_G3_structure_defects(repaired, real_reference)[0]
        == stored["gates"]["G3"]["score"]
        == 13
    )
    assert scorer.gate_G4_ptv_coverage(dose, masks)[0] == stored["gates"]["G4"]["score"] == 12
    assert (
        scorer.gate_G5_oar_sparing(dose, masks, real_reference)[0]
        == stored["gates"]["G5"]["score"]
        == 17
    )


@pytest.mark.parametrize(
    "mutation",
    ["na", "wrong_all", "ptv_wrong", "missing_rectum", "nonfinite", "delete_roi", "invent_bowel"],
)
def test_real_artifact_regressions(tmp_path, scorer, artifact, real_reference, mutation):
    path, dose, masks, computed = artifact
    with (path / "dvh_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = 0
    if mutation == "na":
        rows += [dict(zip(COLUMNS, ["Bowel_bag", "Dmax", "N/A", "Gy", "NA"]))]
        expected = 4
    elif mutation == "wrong_all":
        for row in rows:
            row["metric_value"] = str(float(row["metric_value"]) + 1)
    elif mutation == "ptv_wrong":
        for row in rows:
            if row["structure"] == "PTV_7800":
                row["metric_value"] = str(float(row["metric_value"]) + 1)
    elif mutation == "missing_rectum":
        rows = [row for row in rows if row["structure"] != "Rectum"]
    elif mutation == "nonfinite":
        rows[0]["metric_value"] = "NaN"
    elif mutation == "delete_roi":
        computed = dict(computed)
        computed.pop("Rectum_V70")
    elif mutation == "invent_bowel":
        masks = dict(masks, Bowel=np.zeros_like(dose, dtype=bool))
        masks["Bowel"][0, 0, 0] = True
        assert scorer.gate_G5_oar_sparing(dose, masks, real_reference)[0] == 23
        rows += [dict(zip(COLUMNS, ["Bowel", "Dmax", "0", "Gy", "PASS"]))]
    assert (
        scorer.gate_G7_dvh_csv_honesty(
            write_csv(tmp_path / "metrics.csv", rows), computed, real_reference
        )[0]
        == expected
    )


def test_public_contract_and_diagnostic_maxima(scorer, monkeypatch, tmp_path, real_reference):
    card = json.loads((TASK / "task_card.json").read_text())
    main = (TASK / "main.py").read_text()
    public = main.split("Fixed-phantom applicability and DVH CSV contract:\n", 1)[1].split(
        "Runtime guidance:", 1
    )[0]
    assert public in card["taskPrompt"]
    assert "supersedes" in public and "raw 100 is not available" in public
    assert "PASS_THRESHOLD = 70" in main
    monkeypatch.setattr(scorer, "load_rtstruct", lambda path: copy.deepcopy(real_reference))
    report = scorer.run(tmp_path, DATA / "reference")
    assert report["gates"]["G4"]["max"] == 15
    assert report["gates"]["G5"]["max"] == 23
    assert report["maximum_score"] == 98
    assert not report["pass"]


def test_public_replay_interface_matches_evaluator():
    from tasks.health_medicine.prostate_imrt_matrad_reproduction.main import ProstateIMRTConfig

    prompt = ProstateIMRTConfig().task_description
    card = json.loads((TASK / "task_card.json").read_text())["taskPrompt"]
    replay = (TASK / "scripts/replay.m").read_text()
    for field in ("w_replay", "dij_physicalDose", "dij_dimensions", "nFractions"):
        assert f"data.{field}" in replay
        assert field in prompt and field in card
    assert "dij.physicalDose{1}" in prompt
    assert "MATLAB v7" in prompt
    assert "without applying the fraction count twice" in prompt


@pytest.fixture
def quantitative_report(computed):
    rows = [
        f"| {row['structure']} | {row['metric_type']} | "
        f"{row['metric_value']} {row['units']} | true |"
        for row in rows_for(computed)
    ]
    return (
        "# Research Plan Summary\n\n"
        "Prescription: 78 Gy in 39 fractions, 7 coplanar 6 MV IMRT fields.\n"
        "Gantry angles: 0, 51, 102, 153, 204, 255, 306 degrees.\n\n"
        "## Independent DVH\n"
        "10/10 applicable constraints pass.\n\n"
        "| Structure | Metric | Value | Pass |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


@pytest.mark.parametrize("level", ["#", "##", "###"])
@pytest.mark.parametrize("combined", [False, True])
def test_report_heading_representation(scorer, level, combined):
    headings = ["Beam summary", "DVH", "Constraint compliance"]
    if combined:
        headings = ["Beam summary", "DVH / constraint compliance (independent recompute)"]
    report = "\n".join(f"{level} {heading.upper()}" for heading in headings)
    assert scorer.report_section_coverage(report) == {
        "beam summary",
        "dvh",
        "constraint compliance",
    }


@pytest.mark.parametrize("status", ["true", "false", "PASS", "FAIL"])
def test_quantitative_report_covers_topics_not_scientific_success(
    scorer, quantitative_report, status
):
    report = quantitative_report.replace("| true |", f"| {status} |")
    assert scorer.report_section_coverage(report) == {
        "beam summary",
        "dvh",
        "constraint compliance",
    }


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("mutation", ["missing", "nan", "no_outcome", "units"])
def test_implicit_compliance_requires_every_reported_metric(
    scorer, quantitative_report, index, mutation
):
    lines = quantitative_report.splitlines()
    positions = [position for position, line in enumerate(lines) if line.startswith("| ")][1:]
    position = positions[index]
    if mutation == "missing":
        lines.pop(position)
    else:
        cells = lines[position].split("|")
        if mutation == "nan":
            cells[3] = " nan Gy "
        elif mutation == "no_outcome":
            cells[4] = " "
        elif mutation == "units":
            cells[3] = (
                cells[3].replace("Gy", "%") if "Gy" in cells[3] else cells[3].replace("%", "Gy")
            )
        lines[position] = "|".join(cells)
    assert "constraint compliance" not in scorer.report_section_coverage("\n".join(lines))


@pytest.mark.parametrize("removed", ["7 coplanar", "6 MV", "Gantry angles:"])
def test_implicit_beam_summary_requires_geometry(scorer, quantitative_report, removed):
    assert "beam summary" not in scorer.report_section_coverage(
        quantitative_report.replace(removed, "")
    )


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_report_keywords_and_code_examples_do_not_supply_sections(
    scorer, quantitative_report, fence
):
    assert not scorer.report_section_coverage("Beam summary DVH Constraint compliance")
    assert not scorer.report_section_coverage("## Missing Beam summary DVH Constraint compliance")
    assert not scorer.report_section_coverage(f"{fence}markdown\n{quantitative_report}{fence}\n")
    assert "constraint compliance" not in scorer.report_section_coverage(
        "## Independent DVH\n10/10 applicable constraints pass.\n"
    )
    assert "constraint compliance" not in scorer.report_section_coverage(
        quantitative_report.replace("## Independent DVH", "## Limitations")
    )


@pytest.fixture
def original_report_scorer(real_reference):
    spec = importlib.util.spec_from_file_location(
        "original_prostate_report", DATA / "reference/evaluator_scripts/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unchanged_historical_report_scoring(scorer, artifact, original_report_scorer):
    path = artifact[0]
    report_path = path / "report.md"
    original = report_path.read_bytes()
    expected_before = 4 if "reasoning-high" in str(path) else 2
    assert all(
        heading in original.decode()
        for heading in ("## Beam summary", "## DVH", "## Constraint compliance")
    ) == (expected_before == 4)
    assert original_report_scorer.gate_G8_report_structure(path)[0] == expected_before
    assert scorer.gate_G8_report_structure(path) == (4, [])
    assert report_path.read_bytes() == original


def test_genuine_gpt6_report_scoring_and_unchanged_scientific_gates(
    scorer, real_reference, original_report_scorer
):
    output = GPT6_LOGS / "output"
    if not output.exists():
        pytest.skip("set PROSTATE_GPT6_LOG_ROOT for the genuine GPT6 run")
    report = (output / "report.md").read_bytes()
    assert hashlib.sha256(report).hexdigest() == (
        "5592f342299d69def38859cd308f61040cc8a07c2f7f476291aaf3f17c4d49a9"
    )
    assert not any(
        heading in report.decode()
        for heading in ("## Beam summary", "## DVH", "## Constraint compliance")
    )
    assert original_report_scorer.gate_G8_report_structure(output) == (
        2,
        ["report.md missing sections: ['## Beam summary', '## DVH', '## Constraint compliance']"],
    )
    assert scorer.gate_G8_report_structure(output) == (4, [])
    assert scorer.gate_G0_files_and_validity(output) == (True, [])
    calibration = json.loads((DATA / "reference/reference_calibration.json").read_text())
    plan = scorer.load_rtplan(output / "RTPLAN.dcm")
    assert scorer.gate_G1_beam_geometry(plan, calibration["beam_angles"]) == (6, [])
    assert (
        scorer.gate_G2_deliverability(
            plan, calibration["total_mu_mean"], calibration["per_beam_mu_mean"]
        )[0]
        == 3
    )
    structures = scorer.load_rtstruct(output / "RTSTRUCT_corrected.dcm")
    assert scorer.gate_G3_structure_defects(structures, real_reference) == (16, [])
    assert scorer.gate_G9_decisions(output)[0] == 2
    dose, geom = scorer.load_rtdose(output / "RTDOSE.dcm")
    masks = scorer.compute_roi_masks(structures, geom)
    ptv_score, _, ptv_metrics = scorer.gate_G4_ptv_coverage(dose, masks)
    oar_score, _, oar_metrics = scorer.gate_G5_oar_sparing(dose, masks, real_reference)
    computed = {
        **{
            key.replace("PTV_", "PTV_7800_", 1): value
            for key, value in ptv_metrics.items()
            if key.startswith("PTV_")
        },
        **oar_metrics,
    }
    assert (ptv_score, oar_score) == (15, 23)
    assert scorer.gate_G7_dvh_csv_honesty(output / "dvh_metrics.csv", computed, real_reference) == (
        4,
        [],
    )
    for line in report.decode().splitlines():
        if not line.startswith("| ") or line.startswith("| Structure"):
            continue
        structure, metric, value, status = [cell.strip() for cell in line.strip("|").split("|")]
        assert status == "true"
        assert float(value.split()[0]) == pytest.approx(
            computed[scorer.csv_metric_key(structure, metric)], abs=0.0005
        )
    assert (output / "report.md").read_bytes() == report


def test_genuine_gpt6_rtplan_control_points_remain_valid(scorer):
    output = GPT6_LOGS / "output"
    if not output.exists():
        pytest.skip("set PROSTATE_GPT6_LOG_ROOT for the genuine GPT6 run")
    assert hashlib.sha256((output / "RTPLAN.dcm").read_bytes()).hexdigest() == (
        "857694ffcac1332d356ac051a66974ca1e64a9bcac813c38e43368f5a3377974"
    )
    plan = scorer.load_rtplan(output / "RTPLAN.dcm")
    assert [len(beam.ControlPointSequence) for beam in plan.BeamSequence] == (
        [24, 22, 14, 26, 24, 18, 20]
    )
    assert scorer.gate_G0_files_and_validity(output) == (True, [])
    assert scorer.gate_G1_beam_geometry(plan, [0, 51, 102, 153, 204, 255, 306]) == (6, [])


@pytest.fixture(scope="module")
def malformed_current_output():
    output = CURRENT_GPT6_LOGS / "output"
    if not output.exists():
        pytest.skip("set PROSTATE_CURRENT_LOG_ROOT for the actual malformed GPT6 submission")
    path = output / "RTPLAN.dcm"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "67795c94a4e7c53e853521403fff1163aeff7798d0ed2572ab5dc79ce3c9efae"
    )
    plan = pydicom.dcmread(path)
    assert len(plan.BeamSequence) == 7
    assert all("ControlPointSequence" not in beam for beam in plan.BeamSequence)
    return output


def test_actual_malformed_rtplan_fails_g0(scorer, malformed_current_output):
    reason = "RTPLAN beam 1: missing or empty ControlPointSequence"
    with pytest.raises(ValueError, match=reason):
        scorer.load_rtplan(malformed_current_output / "RTPLAN.dcm")
    assert scorer.gate_G0_files_and_validity(malformed_current_output) == (
        False,
        [f"DICOM parse failed for RTPLAN.dcm: {reason}"],
    )


def test_actual_malformed_rtplan_writes_failed_score(
    scorer, malformed_current_output, real_reference, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        scorer, "gate_G6_gamma", lambda *args: (0, ["not exercised in contract test"], 0.0)
    )
    monkeypatch.setattr(
        scorer, "gate_G10_replay", lambda *args: (0, ["not exercised in contract test"])
    )
    score_path = tmp_path / "score.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--submission",
            str(malformed_current_output),
            "--reference",
            str(DATA / "reference"),
            "--out",
            str(score_path),
        ],
    )
    assert scorer.main() == 0
    report = json.loads(score_path.read_text())
    assert report["gates"]["G0"]["ok"] is False
    assert "ControlPointSequence" in report["gates"]["G0"]["notes"][0]
    assert report["total_score"] == 0
    assert report["pass"] is False
    assert report["maximum_score"] == 98
    assert set(report["gates"]) == {f"G{index}" for index in range(11)}
    for gate in ("G1", "G2"):
        assert report["gates"][gate]["score"] == 0
        assert report["gates"][gate]["notes"] == (
            ["skipped: RTPLAN or reference calibration not loaded"]
        )
    assert report["notes"] == (
        ["RTPLAN.dcm load failed: RTPLAN beam 1: missing or empty ControlPointSequence"]
    )
    assert "TOTAL: 0/98  FAIL" in capsys.readouterr().out


def test_rtplan_loader_does_not_swallow_io_errors(scorer, malformed_current_output, monkeypatch):
    def unavailable_file(path):
        raise PermissionError(str(path))

    monkeypatch.setattr(scorer.pydicom, "dcmread", unavailable_file)
    with pytest.raises(PermissionError):
        scorer.load_rtplan(malformed_current_output / "RTPLAN.dcm")
