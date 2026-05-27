"""Local scorer for life_sciences/msd_multiplex_cytokine_dose_response_prism."""

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


def _analysis_input_title(sheet: dict[str, Any]) -> str | None:
    input_sheets = sheet.get("inputSheets") or []
    if not input_sheets:
        return None
    return input_sheets[0].get("title")


def _choose_analysis_sheet(
    analyses: list[dict[str, Any]],
    *,
    analysis_class: str,
    expected_title: str,
    expected_input_title: str,
) -> dict[str, Any] | None:
    exact_matches = [
        sheet
        for sheet in analyses
        if sheet.get("analysisClass") == analysis_class
        and sheet.get("title") == expected_title
        and _analysis_input_title(sheet) == expected_input_title
    ]
    if exact_matches:
        return exact_matches[0]

    input_matches = [
        sheet
        for sheet in analyses
        if sheet.get("analysisClass") == analysis_class
        and _analysis_input_title(sheet) == expected_input_title
    ]
    if input_matches:
        return input_matches[0]

    title_matches = [
        sheet
        for sheet in analyses
        if sheet.get("analysisClass") == analysis_class and sheet.get("title") == expected_title
    ]
    if title_matches:
        return title_matches[0]

    return None


def extract_prism_manifest(agent_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(BytesIO(agent_bytes)) as zf:
        data_sheets = [
            payload
            for _, payload in _iter_json_objects(zf, "data/sheets/")
            if payload.get("@class") == "DataSheet"
        ]
        graph_sheets = [
            payload
            for _, payload in _iter_json_objects(zf, "graphs/")
            if payload.get("@class") == "FENGraphSheet"
        ]
        analyses = [
            payload
            for _, payload in _iter_json_objects(zf, "analyses/")
            if payload.get("@class") == "AnalysisSheet"
        ]

        xy_sheets = [
            sheet
            for sheet in data_sheets
            if sheet.get("table", {}).get("format") == "xy"
            and "ANALYSIS_VIEW" not in set(sheet.get("flags", []))
        ]
        graph_titles = sorted(sheet.get("title") for sheet in graph_sheets if sheet.get("title"))
        all_dataset_titles: set[str] = set()
        cytokine_manifests: list[dict[str, Any]] = []

        for raw_sheet in sorted(xy_sheets, key=lambda s: s.get("title") or ""):
            raw_title = raw_sheet.get("title")
            if not raw_title:
                continue

            transform_title = f"Transform X of {raw_title}"
            nonlinear_title = f"Nonlin fit of Transform X of {raw_title}"

            table = raw_sheet.get("table", {})
            dataset_titles: list[str] = []
            dose_point_count: int | None = None
            for dataset_id in table.get("dataSets", []):
                dataset = _load_json(zf, f"data/sets/{dataset_id}.json")
                title = _normalize_title(dataset.get("title")) or ""
                dataset_titles.append(title)
                if title:
                    all_dataset_titles.add(title)
                row_count = _replicate_row_count(dataset)
                if row_count is not None:
                    dose_point_count = row_count if dose_point_count is None else max(
                        dose_point_count, row_count
                    )

            x_dataset_id = table.get("xDataSet")
            if x_dataset_id:
                x_dataset = _load_json(zf, f"data/sets/{x_dataset_id}.json")
                row_count = _replicate_row_count(x_dataset)
                if row_count is not None:
                    dose_point_count = row_count if dose_point_count is None else max(
                        dose_point_count, row_count
                    )

            graph_sheet = next(
                (sheet for sheet in graph_sheets if sheet.get("title") == raw_title),
                None,
            )
            transform_sheet = _choose_analysis_sheet(
                analyses,
                analysis_class="TRANSFORM_X_CONCENTRATION",
                expected_title=transform_title,
                expected_input_title=raw_title,
            )
            nonlinear_sheet = _choose_analysis_sheet(
                analyses,
                analysis_class="NONLINEAR_REGRESSION",
                expected_title=nonlinear_title,
                expected_input_title=transform_title,
            )

            cytokine_manifests.append(
                {
                    "raw_data_sheet_title": raw_title,
                    "graph_sheet_title": graph_sheet.get("title") if graph_sheet else None,
                    "graph_input_dataset_count": (
                        len(graph_sheet.get("inputDataSets", [])) if graph_sheet else 0
                    ),
                    "raw_dataset_count": len(table.get("dataSets", [])),
                    "dose_point_count": dose_point_count,
                    "dataset_titles": dataset_titles,
                    "transform_analysis_title": transform_title if transform_sheet else None,
                    "transform_input_sheet_title": raw_title if transform_sheet else None,
                    "nonlinear_fit_title": nonlinear_title if nonlinear_sheet else None,
                    "nonlinear_fit_input_sheet_title": transform_title if nonlinear_sheet else None,
                }
            )

        return {
            "graph_titles": graph_titles,
            "dataset_titles": sorted(all_dataset_titles),
            "cytokine_count": len(cytokine_manifests),
            "analysis_count": len(analyses),
            "cytokines": cytokine_manifests,
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
    for key in ["graph_titles", "dataset_titles", "cytokine_count", "analysis_count"]:
        if actual.get(key) != manifest.get(key):
            reasons.append(f"mismatch:{key}")

    actual_cytokines = list(actual.get("cytokines", []))
    expected_cytokines = list(manifest.get("cytokines", []))
    if len(actual_cytokines) != len(expected_cytokines):
        reasons.append("mismatch:cytokines:length")
    else:
        compare_keys = [
            "raw_data_sheet_title",
            "graph_sheet_title",
            "graph_input_dataset_count",
            "raw_dataset_count",
            "dose_point_count",
            "dataset_titles",
            "transform_analysis_title",
            "transform_input_sheet_title",
            "nonlinear_fit_title",
            "nonlinear_fit_input_sheet_title",
        ]
        for index, (actual_row, expected_row) in enumerate(
            zip(actual_cytokines, expected_cytokines, strict=True)
        ):
            for key in compare_keys:
                if actual_row.get(key) != expected_row.get(key):
                    reasons.append(f"mismatch:cytokines[{index}].{key}")

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
