"""Score CRF-to-SDTM mapping CSV outputs for crf_sdtm_mapping_1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

OUTPUT_COLUMNS = [
    "crf_form",
    "crf_field_label",
    "crf_item_or_placeholder",
    "sdtm_dataset",
    "sdtm_variable",
    "role",
    "origin",
    "mapping_rule",
    "controlled_terms_or_expected_values",
    "goes_to_suppqual",
    "notes",
]

KEY_COLUMNS = [
    "crf_form",
    "crf_field_label",
    "crf_item_or_placeholder",
    "sdtm_dataset",
    "sdtm_variable",
]

VARIANT_DATASETS = {
    "base": ("CM", "SUPPCM"),
    "vs": ("VS", "SUPPVS"),
    "ds": ("DS", "SUPPDS"),
}


@dataclass
class ScoreResult:
    score: float
    errors: list[str] = field(default_factory=list)
    missing_keys: list[list[str]] = field(default_factory=list)
    extra_keys: list[list[str]] = field(default_factory=list)
    mismatches: list[dict[str, str]] = field(default_factory=list)
    compared_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "errors": self.errors,
            "missing_keys": self.missing_keys,
            "extra_keys": self.extra_keys,
            "mismatches": self.mismatches,
            "compared_rows": self.compared_rows,
        }


def normalize_cell(value: object) -> str:
    """Normalize text values exactly as the task contract states."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _format_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(normalize_cell(row[column]) for column in KEY_COLUMNS)


def _parse_csv(text: str, *, label: str) -> tuple[list[str], list[dict[str, str]], list[str]]:
    errors: list[str] = []
    try:
        reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    except csv.Error as exc:
        return [], [], [f"{label}: CSV parse error: {exc}"]

    if fieldnames != OUTPUT_COLUMNS:
        errors.append(
            f"{label}: columns must exactly match expected order; got {fieldnames!r}"
        )

    normalized_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        if not any(normalize_cell(value) for value in row.values()):
            continue
        if None in row:
            errors.append(f"{label}: row {row_index} has extra unheaded values")
            continue
        normalized_rows.append(
            {column: normalize_cell(row.get(column, "")) for column in OUTPUT_COLUMNS}
        )

    if not normalized_rows:
        errors.append(f"{label}: CSV has no data rows")

    return fieldnames, normalized_rows, errors


def _index_rows(
    rows: list[dict[str, str]], *, label: str, allowed_datasets: tuple[str, str]
) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[str]]:
    errors: list[str] = []
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    primary, suppqual = allowed_datasets

    for row_index, row in enumerate(rows, start=2):
        dataset = row["sdtm_dataset"]
        supp_flag = row["goes_to_suppqual"]
        if dataset not in allowed_datasets:
            errors.append(
                f"{label}: row {row_index} has sdtm_dataset={dataset!r}; "
                f"expected one of {allowed_datasets!r}"
            )
        if supp_flag not in {"YES", "NO"}:
            errors.append(
                f"{label}: row {row_index} has goes_to_suppqual={supp_flag!r}; expected YES or NO"
            )
        if dataset == primary and supp_flag != "NO":
            errors.append(
                f"{label}: row {row_index} maps primary dataset {primary} with goes_to_suppqual={supp_flag!r}"
            )
        if dataset == suppqual and supp_flag != "YES":
            errors.append(
                f"{label}: row {row_index} maps supplemental dataset {suppqual} with goes_to_suppqual={supp_flag!r}"
            )

        key = _format_key(row)
        if not all(key):
            errors.append(f"{label}: row {row_index} has an empty composite-key cell: {key!r}")
            continue
        if key in indexed:
            errors.append(f"{label}: duplicate composite key {key!r}")
            continue
        indexed[key] = row

    return indexed, errors


def score_mapping_csv(agent_csv: str, reference_csv: str, *, variant: str) -> ScoreResult:
    """Return a binary score for a submitted mapping CSV."""

    if variant not in VARIANT_DATASETS:
        return ScoreResult(score=0.0, errors=[f"unknown variant {variant!r}"])

    _, agent_rows, agent_errors = _parse_csv(agent_csv, label="agent")
    _, reference_rows, reference_errors = _parse_csv(reference_csv, label="reference")
    errors = agent_errors + reference_errors
    if errors:
        return ScoreResult(score=0.0, errors=errors)

    allowed_datasets = VARIANT_DATASETS[variant]
    agent_index, agent_index_errors = _index_rows(
        agent_rows, label="agent", allowed_datasets=allowed_datasets
    )
    reference_index, reference_index_errors = _index_rows(
        reference_rows, label="reference", allowed_datasets=allowed_datasets
    )
    errors.extend(agent_index_errors)
    errors.extend(reference_index_errors)
    if errors:
        return ScoreResult(score=0.0, errors=errors)

    agent_keys = set(agent_index)
    reference_keys = set(reference_index)
    missing_keys = sorted(reference_keys - agent_keys)
    extra_keys = sorted(agent_keys - reference_keys)

    mismatches: list[dict[str, str]] = []
    for key in sorted(reference_keys & agent_keys):
        expected = reference_index[key]
        observed = agent_index[key]
        for column in OUTPUT_COLUMNS:
            if observed[column] != expected[column]:
                mismatches.append(
                    {
                        "key": " | ".join(key),
                        "column": column,
                        "expected": expected[column],
                        "observed": observed[column],
                    }
                )
                break

    passed = not missing_keys and not extra_keys and not mismatches
    return ScoreResult(
        score=1.0 if passed else 0.0,
        missing_keys=[list(key) for key in missing_keys],
        extra_keys=[list(key) for key in extra_keys],
        mismatches=mismatches[:25],
        compared_rows=len(reference_index),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_DATASETS))
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = score_mapping_csv(
        args.agent.read_text(encoding="utf-8-sig"),
        args.reference.read_text(encoding="utf-8-sig"),
        variant=args.variant,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if result.score == 1.0 else 1)


if __name__ == "__main__":
    main()
