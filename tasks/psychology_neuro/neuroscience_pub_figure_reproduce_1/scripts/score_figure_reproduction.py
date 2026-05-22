"""Local evaluator for the neuroscience Illustrator schematic task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_pdf(path: Path, *, dpi: int = 200) -> np.ndarray:
    doc = fitz.open(str(path))
    if doc.page_count < 1:
        raise ValueError(f"{path} has no pages")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if pix.n == 1:
        return arr[:, :, 0]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _pad_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = max(a.shape[0], b.shape[0])
    width = max(a.shape[1], b.shape[1])

    def pad(image: np.ndarray) -> np.ndarray:
        out = np.full((height, width), 255, dtype=np.uint8)
        out[: image.shape[0], : image.shape[1]] = image
        return out

    return pad(a), pad(b)


def _visual_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = _pad_pair(a, b)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff * diff).mean()))

    ink_a = a < 245
    ink_b = b < 245
    ink_union = np.logical_or(ink_a, ink_b).sum()
    ink_iou = float(np.logical_and(ink_a, ink_b).sum() / ink_union) if ink_union else 1.0

    edge_a = cv2.Canny(a, 80, 160) > 0
    edge_b = cv2.Canny(b, 80, 160) > 0
    edge_union = np.logical_or(edge_a, edge_b).sum()
    edge_iou = float(np.logical_and(edge_a, edge_b).sum() / edge_union) if edge_union else 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "ink_iou": ink_iou,
        "edge_iou": edge_iou,
        "height_ratio": min(a.shape[0], b.shape[0]) / max(a.shape[0], b.shape[0]),
        "width_ratio": min(a.shape[1], b.shape[1]) / max(a.shape[1], b.shape[1]),
    }


def _crop_ink(image: np.ndarray) -> np.ndarray:
    mask = image < 245
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    padding = 10
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, image.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def _normalized_ink_metrics(
    a: np.ndarray, b: np.ndarray, *, size: tuple[int, int] = (1400, 260)
) -> dict[str, float]:
    a_norm = cv2.resize(_crop_ink(a), size, interpolation=cv2.INTER_AREA)
    b_norm = cv2.resize(_crop_ink(b), size, interpolation=cv2.INTER_AREA)
    return _visual_metrics(a_norm, b_norm)


def _is_pdf_like(path: Path) -> bool:
    head = path.read_bytes()[:32]
    return head.startswith(b"%PDF") or head.startswith(b"%!PS-Adobe")


def _has_illustrator_source_features(path: Path) -> bool:
    data = path.read_bytes()
    if not data.startswith(b"%PDF-1.6"):
        return False
    try:
        doc = fitz.open(str(path))
    except Exception:
        return False

    has_oc_properties = False
    has_pieceinfo_illustrator = False
    has_private_data_manifest = False
    has_creator_info = False

    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref, compressed=False)
        except Exception:
            continue
        if "/OCProperties" in obj and "/OCGs" in obj:
            has_oc_properties = True
        if "/PieceInfo" in obj and "/Illustrator" in obj:
            has_pieceinfo_illustrator = True
        if "/AIMetaData" in obj and "AIPrivateData" in obj and "/RoundtripVersion" in obj:
            has_private_data_manifest = True
        if "/CreatorInfo" in obj and "Adobe Illustrator" in obj and "/Subtype /Artwork" in obj:
            has_creator_info = True

    return all(
        [
            has_oc_properties,
            has_pieceinfo_illustrator,
            has_private_data_manifest,
            has_creator_info,
        ]
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_visual(metrics: dict[str, float]) -> float:
    mae_score = _clamp01(1.0 - metrics["mae"] / 0.10)
    rmse_score = _clamp01(1.0 - metrics["rmse"] / 0.24)
    ink_score = _clamp01(metrics["ink_iou"] / 0.70)
    edge_score = _clamp01(metrics["edge_iou"] / 0.10)
    page_score = min(metrics["height_ratio"], metrics["width_ratio"])
    return float(
        0.30 * mae_score
        + 0.20 * rmse_score
        + 0.25 * ink_score
        + 0.15 * edge_score
        + 0.10 * page_score
    )


def evaluate_files(
    *,
    output_pdf: Path,
    output_ai: Path,
    reference_pdf: Path,
    input_pdf: Path,
    input_template_ai: Path,
) -> dict[str, Any]:
    for path, label in [
        (output_pdf, "output_pdf"),
        (output_ai, "output_ai"),
        (reference_pdf, "reference_pdf"),
        (input_pdf, "input_pdf"),
        (input_template_ai, "input_template_ai"),
    ]:
        if not path.exists():
            return {"score": 0.0, "reason": f"missing_{label}", "details": {}}
        if path.stat().st_size < 1024:
            return {"score": 0.0, "reason": f"{label}_too_small", "details": {}}

    output_pdf_hash = _sha256(output_pdf)
    output_ai_hash = _sha256(output_ai)
    input_pdf_hash = _sha256(input_pdf)
    if output_pdf_hash == input_pdf_hash:
        return {"score": 0.0, "reason": "output_pdf_is_direct_input_copy", "details": {}}
    if output_ai_hash == input_pdf_hash:
        return {"score": 0.0, "reason": "output_ai_is_direct_input_pdf_copy", "details": {}}
    if output_ai_hash == output_pdf_hash:
        return {"score": 0.0, "reason": "output_ai_is_pdf_export_copy", "details": {}}
    if output_ai_hash == _sha256(input_template_ai):
        return {"score": 0.0, "reason": "output_ai_is_direct_template_copy", "details": {}}
    if not _is_pdf_like(output_ai):
        return {"score": 0.0, "reason": "output_ai_not_illustrator_pdf_compatible", "details": {}}
    if not _has_illustrator_source_features(output_ai):
        return {
            "score": 0.0,
            "reason": "output_ai_missing_illustrator_source_features",
            "details": {},
        }

    try:
        out_img = _render_pdf(output_pdf)
        ref_img = _render_pdf(reference_pdf)
        input_img = _render_pdf(input_pdf)
        ai_img = _render_pdf(output_ai)
    except Exception as exc:
        return {"score": 0.0, "reason": f"pdf_render_failed: {exc}", "details": {}}

    ref_metrics = _visual_metrics(out_img, ref_img)
    input_metrics = _visual_metrics(out_img, input_img)
    ai_output_metrics = _normalized_ink_metrics(ai_img, out_img)

    if (
        input_metrics["mae"] < 0.035 or input_metrics["mae"] + 0.01 < ref_metrics["mae"]
    ) and ref_metrics["mae"] > 0.03:
        return {
            "score": 0.0,
            "reason": "output_pdf_visually_matches_visible_input_copy",
            "details": {"ref_metrics": ref_metrics, "input_metrics": input_metrics},
        }
    if ai_output_metrics["mae"] > 0.10 or ai_output_metrics["ink_iou"] < 0.50:
        return {
            "score": 0.0,
            "reason": "output_ai_visual_content_mismatch",
            "details": {
                "ref_metrics": ref_metrics,
                "input_metrics": input_metrics,
                "ai_output_metrics": ai_output_metrics,
            },
        }

    visual_score = _score_visual(ref_metrics)
    ai_score = min(
        1.0,
        0.5 * (1.0 - min(ai_output_metrics["mae"] / 0.10, 1.0))
        + 0.5 * min(ai_output_metrics["ink_iou"] / 0.75, 1.0),
    )
    score = _clamp01(0.80 * visual_score + 0.20 * ai_score)
    if output_pdf_hash == _sha256(reference_pdf):
        score = 1.0

    return {
        "score": float(score),
        "reason": "scored",
        "details": {
            "visual_score": visual_score,
            "ai_score": ai_score,
            "ref_metrics": ref_metrics,
            "input_metrics": input_metrics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-pdf", required=True, type=Path)
    parser.add_argument("--output-ai", required=True, type=Path)
    parser.add_argument("--reference-pdf", required=True, type=Path)
    parser.add_argument("--input-pdf", required=True, type=Path)
    parser.add_argument("--input-template-ai", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_files(
        output_pdf=args.output_pdf,
        output_ai=args.output_ai,
        reference_pdf=args.reference_pdf,
        input_pdf=args.input_pdf,
        input_template_ai=args.input_template_ai,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
