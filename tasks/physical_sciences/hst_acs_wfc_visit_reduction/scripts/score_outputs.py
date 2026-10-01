"""Scoring logic for physical_sciences/hst_acs_wfc_visit_reduction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_FIELDS = [
    "source_id",
    "x",
    "y",
    "ra_deg",
    "dec_deg",
    "flux_e_s",
    "mag_ab",
    "snr",
    "sharpness",
    "flags",
]
ALIGN_FIELDS = ["exposure_id", "dx_pix", "dy_pix", "rms_pix", "matched_sources"]
REQUIRED_VISIT_FILES = {
    "source_catalog.csv",
    "alignment_solution.csv",
    "photometry_qc.json",
    "drizzled_image.csv",
    "reduction_report.md",
}
PASS_THRESHOLD = 0.80
FULL_CENTROID_TOLERANCE_PIX = 0.45
PARTIAL_CENTROID_TOLERANCE_PIX = 0.65


def _report_phrase_score(report: str) -> float:
    normalized = "".join(
        " " if unicodedata.category(char) == "Pd" or char in "\u00ad\u2212" else char
        for char in report.casefold()
    )
    normalized = " ".join(normalized.split())
    phrases = (r"calacs style", r"astrodrizzle style", r"astrometric rms", r"cosmic rays?")
    return sum(1.5 for phrase in phrases if re.search(rf"(?<!\w){phrase}(?!\w)", normalized))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _nearest_pairs(
    agent_rows: list[dict[str, str]],
    ref_rows: list[dict[str, str]],
    max_dist: float = 1.0,
) -> list[tuple[dict[str, str], dict[str, str], float]]:
    pairs: list[tuple[dict[str, str], dict[str, str], float]] = []
    used: set[int] = set()
    for ref in ref_rows:
        best: tuple[float, int, dict[str, str]] | None = None
        for index, row in enumerate(agent_rows):
            if index in used:
                continue
            dist = math.hypot(float(row["x"]) - float(ref["x"]), float(row["y"]) - float(ref["y"]))
            if best is None or dist < best[0]:
                best = (dist, index, row)
        if best and best[0] <= max_dist:
            used.add(best[1])
            pairs.append((best[2], ref, best[0]))
    return pairs


def _reference_outputs_root(reference_dir: Path) -> Path:
    candidate = reference_dir / "reference_outputs"
    return candidate if candidate.exists() else reference_dir


def _visit_reference_dirs(reference_dir: Path) -> list[Path]:
    root = _reference_outputs_root(reference_dir)
    refs: list[Path] = []
    for split in ("visible", "hidden"):
        split_dir = root / split
        if split_dir.exists():
            refs.extend(sorted(path for path in split_dir.iterdir() if path.is_dir()))
    if refs:
        return refs
    return sorted(path for path in root.iterdir() if path.is_dir())


def _find_output_visit(output_root: Path, visit_id: str) -> Path:
    direct = output_root / visit_id
    if direct.exists():
        return direct
    nested = output_root / "outputs" / visit_id
    if nested.exists():
        return nested
    return direct


def score_visit(out_visit: Path, ref_visit: Path) -> tuple[float, list[str]]:
    """Return the raw 0-100 visit score and notes for one visit."""
    try:
        ref_sources = _csv_rows(ref_visit / "source_catalog.csv")
        ref_align = _csv_rows(ref_visit / "alignment_solution.csv")
        ref_qc = json.loads((ref_visit / "photometry_qc.json").read_text(encoding="utf-8"))
        ref_img = np.loadtxt(
            ref_visit / "drizzled_image.csv", delimiter=",", encoding="utf-8-sig", quotechar='"'
        )
        for rows, fields, identity, numeric_fields in (
            (ref_sources, SOURCE_FIELDS, "source_id", SOURCE_FIELDS[1:-1]),
            (ref_align, ALIGN_FIELDS, "exposure_id", ALIGN_FIELDS[1:]),
        ):
            if not rows or any(list(row) != fields for row in rows):
                raise ValueError("invalid reference table schema")
            identifiers = [row[identity] for row in rows]
            if not all(identifiers) or len(set(identifiers)) != len(identifiers):
                raise ValueError("invalid reference row identities")
            if not all(
                math.isfinite(float(row[field])) for row in rows for field in numeric_fields
            ):
                raise ValueError("nonfinite reference table measurement")
        if not isinstance(ref_qc, dict) or not all(
            math.isfinite(float(ref_qc[key]))
            for key in (
                "num_sources",
                "background_median_e_s",
                "cosmic_ray_pixels_masked",
                "hot_pixels_masked",
                "aperture_radius_pix",
                "pixfrac",
                "final_scale_arcsec_per_pix",
                "astrometric_rms_pix",
            )
        ):
            raise ValueError("invalid reference photometry QC")
        if ref_img.ndim != 2 or not ref_img.size or not np.isfinite(ref_img).all():
            raise ValueError("invalid reference image")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"invalid evaluator reference for {ref_visit.name}: {exc}") from exc

    try:
        return _score_visit_output(out_visit, ref_visit, ref_sources, ref_align, ref_qc, ref_img)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return 0.0, [f"could not parse outputs for {ref_visit.name}: {exc}"]


def _score_visit_output(
    out_visit: Path,
    ref_visit: Path,
    ref_sources: list[dict[str, str]],
    ref_align: list[dict[str, str]],
    ref_qc: dict[str, Any],
    ref_img: np.ndarray,
) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []
    missing = sorted(name for name in REQUIRED_VISIT_FILES if not (out_visit / name).exists())
    if missing:
        return 0.0, [f"missing outputs for {ref_visit.name}: {missing}"]

    try:
        agent_sources = _csv_rows(out_visit / "source_catalog.csv")
        agent_align = _csv_rows(out_visit / "alignment_solution.csv")
        qc = json.loads((out_visit / "photometry_qc.json").read_text(encoding="utf-8"))
        if not isinstance(qc, dict):
            raise ValueError("photometry_qc.json must be an object")
        agent_img = np.loadtxt(
            out_visit / "drizzled_image.csv", delimiter=",", encoding="utf-8-sig", quotechar='"'
        )
    except Exception as exc:
        return 0.0, [f"could not parse outputs for {ref_visit.name}: {exc}"]

    if agent_sources and list(agent_sources[0]) == SOURCE_FIELDS:
        score += 5.0
    else:
        notes.append("source_catalog.csv schema mismatch")

    if agent_align and list(agent_align[0]) == ALIGN_FIELDS:
        score += 5.0
    else:
        notes.append("alignment_solution.csv schema mismatch")

    if agent_img.shape == ref_img.shape:
        rmse = float(np.sqrt(np.nanmean((agent_img - ref_img) ** 2)))
        if rmse <= 8.0:
            score += 12.0
        elif rmse <= 25.0:
            score += 6.0
        else:
            notes.append(f"drizzled_image RMSE too high: {rmse:.3f}")
    else:
        notes.append(f"drizzled_image shape mismatch: {agent_img.shape} != {ref_img.shape}")

    pairs = _nearest_pairs(agent_sources, ref_sources, max_dist=1.0)
    completeness = len(pairs) / max(len(ref_sources), 1)
    score += min(16.0, 16.0 * completeness)
    if completeness < 0.88:
        notes.append(f"low source completeness: {len(pairs)}/{len(ref_sources)}")

    if pairs:
        xy_med = float(np.median([dist for _, _, dist in pairs]))
        flux_med = float(
            np.median(
                [
                    abs(float(agent["flux_e_s"]) - float(ref["flux_e_s"]))
                    / max(float(ref["flux_e_s"]), 1e-9)
                    for agent, ref, _ in pairs
                ]
            )
        )
        mag_med = float(
            np.median(
                [abs(float(agent["mag_ab"]) - float(ref["mag_ab"])) for agent, ref, _ in pairs]
            )
        )
        sky_med = float(
            np.median(
                [
                    math.hypot(
                        (float(agent["ra_deg"]) - float(ref["ra_deg"])) * 3600.0,
                        (float(agent["dec_deg"]) - float(ref["dec_deg"])) * 3600.0,
                    )
                    for agent, ref, _ in pairs
                ]
            )
        )
        score += (
            10.0
            if xy_med <= FULL_CENTROID_TOLERANCE_PIX
            else 5.0
            if xy_med <= PARTIAL_CENTROID_TOLERANCE_PIX
            else 0.0
        )
        score += 10.0 if flux_med <= 0.08 else 5.0 if flux_med <= 0.16 else 0.0
        score += 8.0 if mag_med <= 0.06 else 4.0 if mag_med <= 0.12 else 0.0
        score += 8.0 if sky_med <= 0.04 else 4.0 if sky_med <= 0.08 else 0.0
        if xy_med > PARTIAL_CENTROID_TOLERANCE_PIX:
            notes.append(f"centroid median error too high: {xy_med:.3f} pix")
        if flux_med > 0.16:
            notes.append(f"flux median relative error too high: {flux_med:.3f}")

    align_score = 0.0
    agent_by_exposure = {row.get("exposure_id"): row for row in agent_align}
    ref_by_exposure = {row.get("exposure_id"): row for row in ref_align}
    if (
        len(agent_align) == len(agent_by_exposure) == len(ref_align) == len(ref_by_exposure)
        and agent_by_exposure.keys() == ref_by_exposure.keys()
        and not ({None, ""} & agent_by_exposure.keys())
    ):
        try:
            errors_by_sign = [
                [
                    abs(sign * float(agent_by_exposure[exposure_id][axis]) - float(reference[axis]))
                    for exposure_id, reference in ref_by_exposure.items()
                    for axis in ("dx_pix", "dy_pix")
                ]
                for sign in (1, -1)
            ]
        except (KeyError, TypeError, ValueError):
            errors_by_sign = []
        if errors_by_sign and all(
            math.isfinite(error) for errors in errors_by_sign for error in errors
        ):
            maximum_error = min(max(errors) for errors in errors_by_sign)
        else:
            maximum_error = math.inf
        if maximum_error <= 0.08:
            align_score = 10.0
        elif maximum_error <= 0.18:
            align_score = 5.0
    score += align_score
    if align_score == 0.0:
        notes.append("alignment solution outside tolerance")

    qc_score = 0.0
    for key, tol in [("cosmic_ray_pixels_masked", 0), ("hot_pixels_masked", 0), ("num_sources", 1)]:
        if abs(float(qc.get(key, -999)) - float(ref_qc.get(key, -111))) <= tol:
            qc_score += 3.0
    if (
        abs(float(qc.get("astrometric_rms_pix", 9)) - float(ref_qc.get("astrometric_rms_pix", 0)))
        <= 0.04
    ):
        qc_score += 3.0
    if float(qc.get("aperture_radius_pix", 0)) == 3.0 and float(qc.get("pixfrac", 0)) == 0.8:
        qc_score += 4.0
    score += qc_score

    report = (out_visit / "reduction_report.md").read_text(encoding="utf-8", errors="ignore")
    score += _report_phrase_score(report)

    return min(score, 100.0), notes


def evaluate_output_directory(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    """Score a static output directory against reference outputs."""
    visit_scores: list[float] = []
    notes: list[str] = []
    reference_visits = _visit_reference_dirs(reference_dir)
    if not reference_visits:
        raise RuntimeError("evaluator reference contains no visits")
    for ref_visit in reference_visits:
        out_visit = _find_output_visit(output_dir, ref_visit.name)
        raw_score, visit_notes = score_visit(out_visit, ref_visit)
        visit_scores.append(raw_score)
        notes.extend(f"{ref_visit.name}: {note}" for note in visit_notes)

    raw_mean = round(float(np.mean(visit_scores)), 2) if visit_scores else 0.0
    score = max(0.0, min(1.0, raw_mean / 100.0))
    return {
        "score": score,
        "raw_score": raw_mean,
        "passed": score >= PASS_THRESHOLD,
        "visit_scores": visit_scores,
        "notes": notes[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score hst_acs_wfc_visit_reduction outputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_output_directory(args.output_dir, args.reference_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
