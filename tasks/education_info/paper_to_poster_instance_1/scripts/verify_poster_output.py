# Deterministic verifier for education_info/paper_to_poster_instance_1.
#
# The verifier scores the submitted poster itself, not by pixel matching a hidden
# expert poster. It uses PyMuPDF/Pillow/OpenCV from the local benchmark runtime.

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TITLE_TOKENS = ("humanity", "last", "exam")
AUTHOR_KEYWORDS = [
    "long phan",
    "alice gatti",
    "ziwen han",
    "nathaniel li",
    "josephina hu",
    "hugh zhang",
    "calvin zhang",
    "mohamed shaaban",
    "john ling",
    "sean shi",
    "michael choi",
    "anish agrawal",
    "arnav chopra",
    "adam khoja",
    "ryan kim",
    "richard ren",
    "jason hausenloy",
    "oliver zhang",
    "mantas mazeika",
    "summer yue",
    "alexandr wang",
    "dan hendrycks",
    "center for ai safety",
    "scale ai",
]
CONTENT_GROUPS = {
    "benchmark_saturation": ["mmlu", "benchmark saturation", "frontier", "low accuracy"],
    "dataset_composition": ["2500", "2 500", "multi modal", "vision text", "subjects"],
    "creation_pipeline": ["70 000", "70000", "13 000", "13000", "expert review", "multi stage"],
    "table1_model_results": ["gpt 4o", "deepseek", "o3 mini", "rms calibration", "accuracy"],
    "token_compute": ["token", "compute", "reasoning models", "completion"],
    "rolling_refinement": ["rolling", "public set", "disagreement", "bio chem health"],
}
TABLE_KEYWORDS = ["accuracy", "rms calibration", "gpt 4o", "deepseek", "o3 mini"]


def _normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_phrase(norm_text: str, phrase: str) -> bool:
    return _normalize_text(phrase) in norm_text


def _title_present(norm_text: str) -> bool:
    tokens = set(norm_text.split())
    return all(token in tokens or (token == "humanity" and "humanitys" in tokens) for token in TITLE_TOKENS)


def _author_score(norm_text: str) -> float:
    hits = sum(1 for phrase in AUTHOR_KEYWORDS if _contains_phrase(norm_text, phrase))
    return hits / len(AUTHOR_KEYWORDS)


def _content_group_hits(norm_text: str) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for group, phrases in CONTENT_GROUPS.items():
        count = sum(1 for phrase in phrases if _contains_phrase(norm_text, phrase))
        hits[group] = count >= 2
    return hits


def _table_present(norm_text: str) -> bool:
    return sum(1 for phrase in TABLE_KEYWORDS if _contains_phrase(norm_text, phrase)) >= 3


def _render_first_page(doc: fitz.Document, *, max_dim: int = 1800) -> np.ndarray:
    page = doc[0]
    rect = page.rect
    scale = max_dim / max(float(rect.width), float(rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def _region_stats(arr: np.ndarray, mask: np.ndarray, bounds: tuple[float, float, float, float]) -> dict[str, float]:
    h, w = mask.shape
    x0, x1, y0, y1 = bounds
    xs = slice(max(0, int(x0 * w)), min(w, int(x1 * w)))
    ys = slice(max(0, int(y0 * h)), min(h, int(y1 * h)))
    region_mask = mask[ys, xs]
    region_rgb = arr[ys, xs]
    if region_mask.size == 0:
        return {"ink": 0.0, "dark": 0.0, "colored": 0.0}
    gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2HSV)
    return {
        "ink": float(region_mask.mean()),
        "dark": float((gray < 180).mean()),
        "colored": float((hsv[:, :, 1] > 45).mean()),
    }


def _visual_metrics(arr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = ((gray < 246) | ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 252))).astype(np.uint8)
    h, w = mask.shape
    area = float(w * h)

    kernel_size = max(3, int(min(w, h) / 120))
    if kernel_size % 2 == 0:
        kernel_size += 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))

    dilate_size = max(5, int(min(w, h) / 70))
    if dilate_size % 2 == 0:
        dilate_size += 1
    grouped = cv2.dilate(closed, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(grouped, 8)
    components: list[dict[str, float]] = []
    for idx in range(1, count):
        x, y, width, height, comp_area = stats[idx]
        frac = comp_area / area
        if frac > 0.002 and width > w * 0.03 and height > h * 0.02:
            components.append(
                {
                    "area_frac": float(frac),
                    "x_frac": float(x / w),
                    "y_frac": float(y / h),
                    "w_frac": float(width / w),
                    "h_frac": float(height / h),
                }
            )

    return {
        "ink_ratio": float(mask.mean()),
        "visual_section_count": len(components),
        "top_left_logo_region": _region_stats(arr, mask, (0.0, 0.22, 0.0, 0.18)),
        "top_right_logo_region": _region_stats(arr, mask, (0.78, 1.0, 0.0, 0.18)),
        "largest_components": sorted(components, key=lambda item: item["area_frac"], reverse=True)[:10],
    }


def _read_png_info(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return {"valid": False, "reason": "missing_or_empty"}
    try:
        data = p.read_bytes()[:24]
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            width = int.from_bytes(data[16:20], "big")
            height = int.from_bytes(data[20:24], "big")
            return {"valid": True, "width": width, "height": height, "format": "PNG", "size_bytes": p.stat().st_size}
        return {"valid": False, "reason": "not_png_or_missing_ihdr", "size_bytes": p.stat().st_size}
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}


def _remote_png_score(png_info: dict[str, Any] | None, page_aspect: float) -> tuple[float, str]:
    if not png_info:
        return 0.5, "png_not_inspected"
    if not png_info.get("valid"):
        return 0.0, "png_invalid"
    width = float(png_info.get("width", 0) or 0)
    height = float(png_info.get("height", 0) or 0)
    if width < 900 or height < 500:
        return 0.0, "png_too_small"
    png_aspect = width / max(height, 1.0)
    if abs(png_aspect - page_aspect) > 0.35:
        return 0.35, "png_aspect_mismatch"
    return 1.0, "png_valid"


def _png_pdf_similarity(png_path: str | Path | None, pdf_render_rgb: np.ndarray) -> dict[str, Any]:
    if png_path is None:
        return {"checked": False, "score": 0.5, "reason": "png_content_not_inspected"}
    try:
        with Image.open(png_path) as image:
            image = image.convert("RGB")
            image.thumbnail((900, 900), Image.Resampling.LANCZOS)
            png_arr = np.asarray(image).copy()
    except Exception as exc:
        return {"checked": True, "score": 0.0, "reason": f"png_unreadable:{exc}"}

    if png_arr.size == 0:
        return {"checked": True, "score": 0.0, "reason": "png_empty_after_decode"}

    target_size = (png_arr.shape[1], png_arr.shape[0])
    pdf_small = cv2.resize(pdf_render_rgb, target_size, interpolation=cv2.INTER_AREA)
    png_gray = cv2.cvtColor(png_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pdf_gray = cv2.cvtColor(pdf_small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    png_vec = (png_gray - float(png_gray.mean())).ravel()
    pdf_vec = (pdf_gray - float(pdf_gray.mean())).ravel()
    denom = float(np.linalg.norm(png_vec) * np.linalg.norm(pdf_vec))
    corr = float(np.dot(png_vec, pdf_vec) / denom) if denom else 0.0

    hist_png = cv2.calcHist([png_gray.astype(np.uint8)], [0], None, [32], [0, 256])
    hist_pdf = cv2.calcHist([pdf_gray.astype(np.uint8)], [0], None, [32], [0, 256])
    cv2.normalize(hist_png, hist_png)
    cv2.normalize(hist_pdf, hist_pdf)
    hist_corr = float(cv2.compareHist(hist_png, hist_pdf, cv2.HISTCMP_CORREL))

    score = max(0.0, min(1.0, 0.75 * max(0.0, corr) + 0.25 * max(0.0, hist_corr)))
    return {
        "checked": True,
        "score": score,
        "gray_correlation": corr,
        "histogram_correlation": hist_corr,
        "thumbnail_size": [int(png_arr.shape[1]), int(png_arr.shape[0])],
        "reason": "png_matches_pdf_render" if score >= 0.60 else "png_does_not_match_pdf_render",
    }


def score_poster_pdf(
    pdf_path: str | Path,
    png_info: dict[str, Any] | None = None,
    png_path: str | Path | None = None,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    details: dict[str, Any] = {"score": 0.0, "hard_failures": []}
    if not pdf_path.exists() or pdf_path.stat().st_size <= 0:
        details["hard_failures"].append("poster_pdf_missing_or_empty")
        return details

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        details["hard_failures"].append(f"poster_pdf_unreadable:{exc}")
        return details

    try:
        page_count = doc.page_count
        page = doc[0] if page_count else None
        if page is None:
            details["hard_failures"].append("poster_pdf_has_no_pages")
            return details
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        page_aspect = page_width / max(page_height, 1.0)
        text = "\n".join(page.get_text("text") for page in doc)
        norm_text = _normalize_text(text)
        arr = _render_first_page(doc)
    finally:
        doc.close()

    visual = _visual_metrics(arr)
    png_similarity = _png_pdf_similarity(png_path, arr)
    title_ok = _title_present(norm_text)
    author_score = _author_score(norm_text)
    content_hits = _content_group_hits(norm_text)
    content_hit_count = sum(1 for ok in content_hits.values() if ok)
    table_ok = _table_present(norm_text)
    png_score, png_reason = _remote_png_score(png_info, page_aspect)

    top_left = visual["top_left_logo_region"]
    top_right = visual["top_right_logo_region"]
    hle_logo_ok = top_left["ink"] >= 0.03 and top_left["dark"] >= 0.025
    nature_logo_ok = top_right["ink"] >= 0.03 and top_right["dark"] >= 0.025
    visual_sections = int(visual["visual_section_count"])
    ink_ratio = float(visual["ink_ratio"])

    details.update(
        {
            "page_count": page_count,
            "page_size_pts": [page_width, page_height],
            "page_aspect": page_aspect,
            "text_characters": len(text),
            "title_ok": title_ok,
            "author_score": author_score,
            "content_group_hits": content_hits,
            "content_hit_count": content_hit_count,
            "table_ok": table_ok,
            "visual_metrics": visual,
            "hle_logo_ok": hle_logo_ok,
            "nature_logo_ok": nature_logo_ok,
            "png_info": png_info,
            "png_score": png_score,
            "png_reason": png_reason,
            "png_pdf_similarity": png_similarity,
        }
    )

    hard_failures = details["hard_failures"]
    if page_count != 1:
        hard_failures.append("poster_pdf_must_be_single_page")
    if not (1.25 <= page_aspect <= 2.2):
        hard_failures.append("poster_pdf_must_be_landscape_poster")
    if len(text) < 900:
        hard_failures.append("poster_text_layer_too_sparse")
    if not title_ok:
        hard_failures.append("title_not_detected")
    if author_score < 0.70:
        hard_failures.append("author_affiliation_block_not_detected")
    if content_hit_count < 5:
        hard_failures.append("required_figure_content_not_detected")
    if not table_ok:
        hard_failures.append("table1_model_results_not_detected")
    if not hle_logo_ok:
        hard_failures.append("hle_logo_top_left_not_detected")
    if not nature_logo_ok:
        hard_failures.append("nature_logo_top_right_not_detected")
    if visual_sections < 5 or ink_ratio < 0.14:
        hard_failures.append("poster_layout_too_sparse")
    if png_score <= 0.0:
        hard_failures.append(png_reason)
    if png_similarity.get("checked") and float(png_similarity.get("score", 0.0)) < 0.60:
        hard_failures.append(png_similarity.get("reason", "png_does_not_match_pdf_render"))

    if hard_failures:
        details["score"] = 0.0
        return details

    content_score = 0.20 * float(title_ok) + 0.22 * min(1.0, author_score / 0.90) + 0.38 * (content_hit_count / len(CONTENT_GROUPS)) + 0.20 * float(table_ok)
    layout_score = (
        0.25 * min(1.0, max(0.0, (ink_ratio - 0.14) / 0.10))
        + 0.25 * min(1.0, visual_sections / 10.0)
        + 0.25 * float(hle_logo_ok)
        + 0.25 * float(nature_logo_ok)
    )
    export_score = min(png_score, float(png_similarity.get("score", png_score)))
    final_score = 0.58 * content_score + 0.34 * layout_score + 0.08 * export_score
    details["subscores"] = {"content": content_score, "layout": layout_score, "export": export_score}
    details["score"] = float(max(0.0, min(1.0, final_score)))
    return details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--png")
    parser.add_argument("--png-info-json")
    args = parser.parse_args()

    png_info = None
    if args.png_info_json:
        png_info = json.loads(args.png_info_json)
    elif args.png:
        png_info = _read_png_info(args.png)

    payload = score_poster_pdf(args.pdf, png_info, png_path=args.png)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("score", 0.0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
