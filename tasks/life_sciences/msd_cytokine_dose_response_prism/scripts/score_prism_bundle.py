"""Local scorer for life_sciences/msd_cytokine_dose_response_prism."""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any


@dataclass
class PrismBundleScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    actual_manifest: dict[str, Any] | None = None


def _normalize_title(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        string_value = value.get("string")
        if string_value is None:
            return None
        return str(string_value).strip()
    return str(value).strip()


def _load_json(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    return json.loads(zf.read(member).decode("utf-8"))


def _iter_json_objects(zf: zipfile.ZipFile, prefix: str):
    for name in zf.namelist():
        if name.startswith(prefix) and name.endswith(".json"):
            yield name, _load_json(zf, name)


def _replicate_row_count(dataset: dict[str, Any]) -> int | None:
    counts: list[int] = []
    for replicate in dataset.get("replicates", []):
        first_row = replicate.get("firstRow")
        last_row = replicate.get("lastRow")
        if isinstance(first_row, int) and isinstance(last_row, int) and last_row >= first_row:
            counts.append(last_row - first_row + 1)
    if not counts:
        return None
    return max(counts)


def _choose_raw_sheet(data_sheets: list[dict[str, Any]], expected_title: str) -> dict[str, Any] | None:
    xy_sheets = []
    for sheet in data_sheets:
        table = sheet.get("table", {})
        flags = set(sheet.get("flags", []))
        if table.get("format") == "xy" and "ANALYSIS_VIEW" not in flags:
            xy_sheets.append(sheet)
    for sheet in xy_sheets:
        if sheet.get("title") == expected_title:
            return sheet
    return xy_sheets[0] if xy_sheets else None


def _choose_graph_sheet(graph_sheets: list[dict[str, Any]], expected_title: str) -> dict[str, Any] | None:
    for sheet in graph_sheets:
        if sheet.get("title") == expected_title:
            return sheet
    return graph_sheets[0] if graph_sheets else None


def _choose_analysis(analyses: list[dict[str, Any]], *, analysis_class: str, expected_title: str) -> dict[str, Any] | None:
    exact_matches = [
        sheet for sheet in analyses
        if sheet.get("analysisClass") == analysis_class and sheet.get("title") == expected_title
    ]
    if exact_matches:
        return exact_matches[0]
    class_matches = [sheet for sheet in analyses if sheet.get("analysisClass") == analysis_class]
    if class_matches:
        return class_matches[0]
    title_matches = [sheet for sheet in analyses if sheet.get("title") == expected_title]
    if title_matches:
        return title_matches[0]
    return None


def extract_prism_manifest(agent_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(agent_bytes)) as zf:
        data_sheets = [payload for _, payload in _iter_json_objects(zf, "data/sheets/") if payload.get("@class") == "DataSheet"]
        graph_sheets = [payload for _, payload in _iter_json_objects(zf, "graphs/") if payload.get("@class") == "FENGraphSheet"]
        analyses = [payload for _, payload in _iter_json_objects(zf, "analyses/") if payload.get("@class") == "AnalysisSheet"]

        raw_sheet = _choose_raw_sheet(data_sheets, expected_title="IL13")
        graph_sheet = _choose_graph_sheet(graph_sheets, expected_title="IL13")
        transform_sheet = _choose_analysis(
            analyses,
            analysis_class="TRANSFORM_X_CONCENTRATION",
            expected_title="Transform X of IL13",
        )
        nonlinear_sheet = _choose_analysis(
            analyses,
            analysis_class="NONLINEAR_REGRESSION",
            expected_title="Nonlin fit of Transform X of IL13",
        )

        dataset_titles: list[str] = []
        dose_point_count: int | None = None
        raw_dataset_count = 0
        if raw_sheet is not None:
            table = raw_sheet.get("table", {})
            dataset_ids = list(table.get("dataSets", []))
            raw_dataset_count = len(dataset_ids)
            for dataset_id in dataset_ids:
                dataset = _load_json(zf, f"data/sets/{dataset_id}.json")
                dataset_titles.append(_normalize_title(dataset.get("title")) or "")
                row_count = _replicate_row_count(dataset)
                if row_count is not None:
                    dose_point_count = row_count if dose_point_count is None else max(dose_point_count, row_count)

            x_dataset_id = table.get("xDataSet")
            if x_dataset_id:
                x_dataset = _load_json(zf, f"data/sets/{x_dataset_id}.json")
                row_count = _replicate_row_count(x_dataset)
                if row_count is not None:
                    dose_point_count = row_count if dose_point_count is None else max(dose_point_count, row_count)

        return {
            "graph_sheet_title": graph_sheet.get("title") if graph_sheet else None,
            "graph_input_dataset_count": len(graph_sheet.get("inputDataSets", [])) if graph_sheet else 0,
            "raw_data_sheet_title": raw_sheet.get("title") if raw_sheet else None,
            "raw_dataset_count": raw_dataset_count,
            "dose_point_count": dose_point_count,
            "dataset_titles": dataset_titles,
            "transform_analysis_title": transform_sheet.get("title") if transform_sheet else None,
            "transform_input_sheet_title": (
                transform_sheet.get("inputSheets", [{}])[0].get("title") if transform_sheet and transform_sheet.get("inputSheets") else None
            ),
            "nonlinear_fit_title": nonlinear_sheet.get("title") if nonlinear_sheet else None,
            "nonlinear_fit_input_sheet_title": (
                nonlinear_sheet.get("inputSheets", [{}])[0].get("title") if nonlinear_sheet and nonlinear_sheet.get("inputSheets") else None
            ),
        }


def score_prism_bundle_bytes(agent_bytes: bytes, manifest: dict[str, Any]) -> PrismBundleScoreResult:
    try:
        actual = extract_prism_manifest(agent_bytes)
    except zipfile.BadZipFile as exc:
        return PrismBundleScoreResult(
            score=0.0,
            passed=False,
            reasons=[f"invalid_prism_zip:{exc}"],
        )
    except KeyError as exc:
        return PrismBundleScoreResult(
            score=0.0,
            passed=False,
            reasons=[f"missing_bundle_member:{exc}"],
        )
    except Exception as exc:
        return PrismBundleScoreResult(
            score=0.0,
            passed=False,
            reasons=[f"manifest_extraction_failed:{exc}"],
        )

    reasons: list[str] = []
    for key in [
        "graph_sheet_title",
        "graph_input_dataset_count",
        "raw_data_sheet_title",
        "raw_dataset_count",
        "dose_point_count",
        "transform_analysis_title",
        "transform_input_sheet_title",
        "nonlinear_fit_title",
        "nonlinear_fit_input_sheet_title",
    ]:
        if actual.get(key) != manifest.get(key):
            reasons.append(f"mismatch:{key}")

    if list(actual.get("dataset_titles", [])) != list(manifest.get("dataset_titles", [])):
        reasons.append("mismatch:dataset_titles")

    passed = not reasons
    return PrismBundleScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reasons=reasons,
        actual_manifest=actual,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with open(args.agent, "rb") as f:
        agent_bytes = f.read()
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    result = score_prism_bundle_bytes(agent_bytes=agent_bytes, manifest=manifest)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
