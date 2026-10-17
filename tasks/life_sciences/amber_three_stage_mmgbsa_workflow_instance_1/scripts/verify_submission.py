"""Verifier for amber_three_stage_mmgbsa_workflow_instance_1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts.workflow_checks import (
    check_results as _check_results,
    check_workflow,
    parse_delta_total as _parse_delta_total,
)

SYSTEM_BASENAME = "GLN_phb2_parl_pgam5_model_0"
REQUIRED_FILES = (
    "submit_min.sh",
    "submit_prod.sh",
    "submit_mmgbsa.sh",
    "FINAL_RESULTS_MMGBSA.dat",
)
IGNORED_FILENAMES = {".gitkeep"}
ACCEPTED_DELTA_RANGE = (-130.0, -100.0)


def evaluate_output_bundle(
    files: Mapping[str, str],
    *,
    present_files: list[str] | None = None,
    hidden_reference_text: str | None = None,
) -> dict:
    reasons: list[str] = []
    visible_files = sorted(
        name
        for name in (present_files if present_files is not None else files.keys())
        if name not in IGNORED_FILENAMES
    )
    expected = sorted(REQUIRED_FILES)

    missing = sorted(set(expected) - set(visible_files))
    if missing:
        reasons.append(f"missing_required_files:{missing}")
    if visible_files != expected:
        reasons.append("output file set must contain exactly the four required files")

    submit_min = files.get("submit_min.sh")
    submit_prod = files.get("submit_prod.sh")
    submit_mmgbsa = files.get("submit_mmgbsa.sh")
    final_results = files.get("FINAL_RESULTS_MMGBSA.dat")

    if submit_min is None:
        reasons.append("missing submit_min.sh")
    if submit_prod is None:
        reasons.append("missing submit_prod.sh")
    if submit_mmgbsa is None:
        reasons.append("missing submit_mmgbsa.sh")
    if final_results is None:
        reasons.append("missing FINAL_RESULTS_MMGBSA.dat")

    reasons.extend(check_workflow(dict(files)))
    if final_results is not None:
        reasons.extend(_check_results(final_results, hidden_reference_text, dict(files)))

    deduped: list[str] = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    passed = not deduped
    delta_total = _parse_delta_total(final_results)
    hidden_delta = _parse_delta_total(hidden_reference_text)
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasons": deduped,
        "delta_total": delta_total,
        "hidden_delta_total": hidden_delta,
    }


def _load_directory(path: Path) -> tuple[dict[str, str], list[str]]:
    files: dict[str, str] = {}
    names: list[str] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        names.append(child.name)
        files[child.name] = child.read_text(encoding="utf-8", errors="replace")
    return files, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify amber_three_stage_mmgbsa_workflow_instance_1 outputs."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--hidden-reference", help="Optional hidden reference FINAL_RESULTS_MMGBSA.dat"
    )
    args = parser.parse_args()

    files, names = _load_directory(Path(args.output_dir))
    hidden_reference_text = None
    if args.hidden_reference:
        hidden_reference_text = Path(args.hidden_reference).read_text(
            encoding="utf-8", errors="replace"
        )
    payload = evaluate_output_bundle(
        files,
        present_files=names,
        hidden_reference_text=hidden_reference_text,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
