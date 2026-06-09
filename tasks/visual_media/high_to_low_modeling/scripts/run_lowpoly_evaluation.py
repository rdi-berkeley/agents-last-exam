"""Single-script high->low benchmark pipeline.

This file is the only workflow script for the Blender high->low benchmark.
It exposes a callable class, `LowpolyBenchmark`, and can also run its Blender
stage by reinvoking itself through Blender with `--mode blender-stage`.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
BLENDER_DIR = ROOT_DIR / "blender_tools"
REPO_ROOT = THIS_DIR.parents[4]
for path in (THIS_DIR, BLENDER_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass(frozen=True)
class LowpolyPaths:
    asset_root: Path | None
    config_path: Path
    config_exists_on_disk: bool
    high_obj: Path
    low_obj: Path


class LowpolyBenchmark:
    DEFAULT_BLENDER = os.environ.get("BLENDER_BINARY", "/Applications/Blender.app/Contents/MacOS/Blender")
    DEFAULT_CONFIG: dict[str, Any] = {
        "triangle_ratio_cap": 0.05,
        "sample_count_per_mesh": 20_000,
        "judge_model": "gpt-5.2",
        "views_metric": ["front", "back", "left", "right", "top"],
        "views_evidence": ["front", "back", "left", "right", "top", "perspective"],
        "normalize_distances_by": "high_bbox_diagonal",
        "temperature": 0,
    }
    VIEW_ORDER = ["front", "back", "left", "right", "top", "perspective"]
    VIEW_LABELS = {
        "front": "Front",
        "back": "Back",
        "left": "Left",
        "right": "Right",
        "top": "Top",
        "perspective": "Perspective",
    }
    VIEW_SPECS = {
        "front": {"azimuth_deg": 0, "elevation_deg": 0},
        "back": {"azimuth_deg": 180, "elevation_deg": 0},
        "left": {"azimuth_deg": 90, "elevation_deg": 0},
        "right": {"azimuth_deg": 270, "elevation_deg": 0},
        "top": {"azimuth_deg": 0, "elevation_deg": 89},
        "perspective": {"azimuth_deg": 45, "elevation_deg": 25},
    }

    def __init__(
        self,
        *,
        output_dir: str,
        asset_root: str = "",
        high_obj: str = "",
        low_obj: str = "",
        evaluation_config: str = "",
        blender_binary: str = DEFAULT_BLENDER,
        judge_backend: str = "openai",
        judge_model: str = "",
        skip_judge: bool = False,
        seed: int = 1337,
    ) -> None:
        self.paths = self.resolve_paths(
            asset_root=asset_root,
            high_obj=high_obj,
            low_obj=low_obj,
            evaluation_config=evaluation_config,
        )
        self.config = self.load_config(self.paths)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blender_binary = blender_binary
        self.judge_backend = judge_backend
        self.judge_model_override = judge_model
        self.skip_judge = skip_judge
        self.seed = int(seed)

    @staticmethod
    def load_json(path: Path) -> dict[str, Any] | list[Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def dump_json(path: Path, payload: dict[str, Any]) -> str:
        LowpolyBenchmark.write_json(path, payload)
        return str(path)

    @staticmethod
    def mean(values: Iterable[float]) -> float:
        vals = [float(value) for value in values]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    @staticmethod
    def score_from_error(value: float, tolerance: float) -> float:
        return max(0.0, 1.0 - min(1.0, float(value) / float(tolerance)))

    @staticmethod
    def image_to_data_url(path: Path) -> str:
        mime = "image/png"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @classmethod
    def resolve_paths(
        cls,
        *,
        asset_root: str = "",
        high_obj: str = "",
        low_obj: str = "",
        evaluation_config: str = "",
    ) -> LowpolyPaths:
        root = Path(asset_root).expanduser().resolve() if asset_root else None
        high_path = Path(high_obj).expanduser().resolve() if high_obj else None
        low_path = Path(low_obj).expanduser().resolve() if low_obj else None
        config_path = Path(evaluation_config).expanduser().resolve() if evaluation_config else None

        if root is not None:
            high_path = high_path or (root / "input" / "high.obj")
            low_path = low_path or (root / "output" / "low.obj")
            config_path = config_path or (root / "evaluation_config.json")

        if high_path is None or low_path is None:
            raise RuntimeError("Need either --asset-root or both --high-obj and --low-obj")
        if config_path is None:
            parent = high_path.parent.parent if high_path.parent.name == "input" else high_path.parent
            config_path = parent / "evaluation_config.json"

        return LowpolyPaths(
            asset_root=root,
            config_path=config_path,
            config_exists_on_disk=config_path.exists(),
            high_obj=high_path,
            low_obj=low_path,
        )

    @classmethod
    def load_config_from_path(cls, config_path: str | Path | None) -> dict[str, Any]:
        config = dict(cls.DEFAULT_CONFIG)
        if config_path:
            path = Path(config_path).expanduser().resolve()
            if path.exists():
                raw = cls.load_json(path)
                if not isinstance(raw, dict):
                    raise RuntimeError(f"Lowpoly evaluation config must be an object: {path}")
                config.update(raw)
        config["triangle_ratio_cap"] = float(config["triangle_ratio_cap"])
        config["sample_count_per_mesh"] = int(config["sample_count_per_mesh"])
        config["temperature"] = float(config.get("temperature", 0))
        config["judge_model"] = str(config["judge_model"])
        config["views_metric"] = [str(v) for v in config.get("views_metric", cls.DEFAULT_CONFIG["views_metric"])]
        config["views_evidence"] = [str(v) for v in config.get("views_evidence", cls.DEFAULT_CONFIG["views_evidence"])]
        config["normalize_distances_by"] = str(config.get("normalize_distances_by", "high_bbox_diagonal"))
        return config

    @classmethod
    def load_config(cls, paths: LowpolyPaths) -> dict[str, Any]:
        return cls.load_config_from_path(paths.config_path if paths.config_exists_on_disk else None)

    @classmethod
    def compute_scores(cls, raw_metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
        distance_cfg = dict(config.get("distance_thresholds", {}))
        silhouette_cfg = dict(config.get("silhouette_thresholds", {}))
        alignment_cfg = dict(config.get("alignment_thresholds", {}))
        mesh_cfg = dict(config.get("mesh_health_thresholds", {}))

        distance_pass = (
            float(raw_metrics["chamfer_low_to_high_norm"]) <= float(distance_cfg.get("good", 0.01))
            and float(raw_metrics["chamfer_high_to_low_norm"]) <= float(distance_cfg.get("ok", distance_cfg.get("good", 0.03)))
            and float(raw_metrics["p95_distance_norm"]) <= float(distance_cfg.get("bad", distance_cfg.get("ok", 0.08)))
            and float(raw_metrics["within_tolerance_ratio"]) >= 0.95
        )
        silhouette_pass = float(raw_metrics["silhouette_iou_mean"]) >= float(silhouette_cfg.get("good", 0.95))
        alignment_pass = (
            float(raw_metrics["center_offset_norm"]) <= float(alignment_cfg.get("center_good", 0.02))
            and float(raw_metrics["max_bbox_axis_diff_ratio"]) <= float(alignment_cfg.get("scale_good", 0.05))
        )
        nonmanifold_score = 1.0 if int(raw_metrics["nonmanifold_edge_count"]) == 0 else 0.0
        mesh_health_raw = cls.mean(
            [
                nonmanifold_score,
                cls.score_from_error(float(raw_metrics["degenerate_face_ratio"]), 0.002),
                cls.score_from_error(float(raw_metrics["loose_vert_ratio"]), 0.001),
            ]
        )
        mesh_health_pass = mesh_health_raw >= float(mesh_cfg.get("min_health_score", 0.5))

        distance_score = 1.0 if distance_pass else 0.0
        silhouette_score = 1.0 if silhouette_pass else 0.0
        alignment_score = 1.0 if alignment_pass else 0.0
        mesh_health_score = 1.0 if mesh_health_pass else 0.0
        geometry_score = (
            0.45 * distance_score
            + 0.20 * silhouette_score
            + 0.20 * alignment_score
            + 0.15 * mesh_health_score
        )
        return {
            "distance_score": float(distance_score),
            "silhouette_score": float(silhouette_score),
            "alignment_score": float(alignment_score),
            "mesh_health_score": float(mesh_health_score),
            "nonmanifold_score": float(nonmanifold_score),
            "distance_pass": float(distance_pass),
            "silhouette_pass": float(silhouette_pass),
            "alignment_pass": float(alignment_pass),
            "mesh_health_pass": float(mesh_health_pass),
            "mesh_health_raw": float(mesh_health_raw),
            "geometry_score": float(geometry_score),
        }

    @staticmethod
    def _pil_modules() -> tuple[Any, Any, Any, Any]:
        from PIL import Image, ImageChops, ImageDraw, ImageFont

        return Image, ImageChops, ImageDraw, ImageFont

    @classmethod
    def _font(cls) -> Any:
        _, _, _, image_font = cls._pil_modules()
        return image_font.load_default()

    @classmethod
    def _labeled(cls, image: Any, label: str) -> Any:
        image_mod, _, image_draw, _ = cls._pil_modules()
        pad = 32
        out = image_mod.new("RGBA", (image.width, image.height + pad), (255, 255, 255, 255))
        out.paste(image, (0, pad))
        draw = image_draw.Draw(out)
        draw.text((12, 8), label, fill=(10, 10, 10, 255), font=cls._font())
        return out

    @classmethod
    def compose_contact_sheet(cls, image_map: dict[str, str], output_path: Path, *, columns: int = 3) -> str:
        image_mod, _, _, _ = cls._pil_modules()
        ordered = [view for view in cls.VIEW_ORDER if view in image_map]
        if not ordered:
            raise RuntimeError("No images provided for contact sheet")
        cells = [
            cls._labeled(
                image_mod.open(image_map[view]).convert("RGBA"),
                cls.VIEW_LABELS.get(view, view.title()),
            )
            for view in ordered
        ]
        cell_w = max(image.width for image in cells)
        cell_h = max(image.height for image in cells)
        rows = math.ceil(len(cells) / columns)
        sheet = image_mod.new("RGBA", (cell_w * columns, cell_h * rows), (255, 255, 255, 255))
        for idx, cell in enumerate(cells):
            x = (idx % columns) * cell_w
            y = (idx // columns) * cell_h
            sheet.paste(cell, (x, y))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        return str(output_path)

    @classmethod
    def compose_overlay_views(
        cls,
        high_map: dict[str, str],
        low_map: dict[str, str],
        output_dir: Path,
    ) -> dict[str, str]:
        image_mod, _, _, _ = cls._pil_modules()
        output_dir.mkdir(parents=True, exist_ok=True)
        merged: dict[str, str] = {}
        for view in cls.VIEW_ORDER:
            if view not in high_map or view not in low_map:
                continue
            high = image_mod.open(high_map[view]).convert("RGBA")
            low = image_mod.open(low_map[view]).convert("RGBA")
            if high.size != low.size:
                low = low.resize(high.size)
            composite = image_mod.alpha_composite(high, low)
            out_path = output_dir / f"{view}.png"
            composite.save(out_path)
            merged[view] = str(out_path)
        return merged

    @classmethod
    def _mask_from_image(cls, path: Path) -> Any:
        image_mod, _, _, _ = cls._pil_modules()
        image = image_mod.open(path).convert("L")
        return image.point(lambda value: 0 if value > 240 else 255, mode="1")

    @classmethod
    def silhouette_iou(cls, high_path: Path, low_path: Path) -> float:
        high_mask = cls._mask_from_image(high_path)
        low_mask = cls._mask_from_image(low_path)
        if high_mask.size != low_mask.size:
            low_mask = low_mask.resize(high_mask.size)
        high_data = list(high_mask.getdata())
        low_data = list(low_mask.getdata())
        inter = sum(1 for left, right in zip(high_data, low_data) if left and right)
        union = sum(1 for left, right in zip(high_data, low_data) if left or right)
        return 1.0 if union == 0 else float(inter / union)

    @classmethod
    def compose_silhouette_sheet(
        cls,
        high_map: dict[str, str],
        low_map: dict[str, str],
        output_path: Path,
    ) -> tuple[str, float]:
        image_mod, image_chops, image_draw, _ = cls._pil_modules()
        ordered = [view for view in cls.VIEW_ORDER if view in high_map and view in low_map]
        if not ordered:
            raise RuntimeError("No silhouette views provided")

        rows: list[Any] = []
        ious: list[float] = []
        for view in ordered:
            high = image_mod.open(high_map[view]).convert("RGBA")
            low = image_mod.open(low_map[view]).convert("RGBA")
            if high.size != low.size:
                low = low.resize(high.size)
            iou = cls.silhouette_iou(Path(high_map[view]), Path(low_map[view]))
            ious.append(iou)

            diff = image_chops.difference(high.convert("RGB"), low.convert("RGB")).convert("RGBA")
            row = image_mod.new("RGBA", (high.width * 3, high.height + 32), (255, 255, 255, 255))
            row.paste(high, (0, 32))
            row.paste(low, (high.width, 32))
            row.paste(diff, (high.width * 2, 32))
            draw = image_draw.Draw(row)
            draw.text(
                (12, 8),
                f"{cls.VIEW_LABELS.get(view, view.title())} | IoU={iou:.4f}",
                fill=(10, 10, 10, 255),
                font=cls._font(),
            )
            rows.append(row)

        sheet_w = max(row.width for row in rows)
        sheet_h = sum(row.height for row in rows)
        sheet = image_mod.new("RGBA", (sheet_w, sheet_h), (255, 255, 255, 255))
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        return str(output_path), cls.mean(ious)

    @classmethod
    def render_heatmap_view(cls, base_image_path: Path, points: list[dict[str, float]], output_path: Path) -> str:
        image_mod, _, image_draw, _ = cls._pil_modules()
        base = image_mod.open(base_image_path).convert("RGBA")
        draw = image_draw.Draw(base, "RGBA")
        width, height = base.size
        radius = max(2, int(min(width, height) * 0.005))
        for point in points:
            x = max(0.0, min(1.0, float(point["x"])))
            y = max(0.0, min(1.0, float(point["y"])))
            distance_norm = float(point["distance_norm"])
            if distance_norm < 0.0025:
                color = (52, 201, 36, 210)
            elif distance_norm <= 0.005:
                color = (245, 204, 0, 210)
            else:
                color = (220, 55, 42, 210)
            px = x * width
            py = (1.0 - y) * height
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(output_path)
        return str(output_path)

    @classmethod
    def format_metrics_summary(cls, metrics: dict[str, Any]) -> str:
        lines = [
            "MODEL EVALUATION",
            "",
            f"High poly triangles: {int(metrics['triangle_count_high']):,}",
            f"Low poly triangles: {int(metrics['triangle_count_low']):,}",
            f"Triangle ratio: {float(metrics['triangle_ratio']):.6f}",
            "",
            f"Chamfer low->high: {float(metrics['chamfer_low_to_high_norm']):.6f}",
            f"Chamfer high->low: {float(metrics['chamfer_high_to_low_norm']):.6f}",
            f"p95 distance: {float(metrics['p95_distance_norm']):.6f}",
            f"Within tolerance: {100.0 * float(metrics['within_tolerance_ratio']):.2f}%",
            "",
            f"Silhouette IoU mean: {float(metrics['silhouette_iou_mean']):.6f}",
            f"Center offset norm: {float(metrics['center_offset_norm']):.6f}",
            f"Max bbox axis diff ratio: {float(metrics['max_bbox_axis_diff_ratio']):.6f}",
            "",
            "Mesh health:",
            f"non-manifold edges: {int(metrics['nonmanifold_edge_count'])}",
            f"degenerate face ratio: {float(metrics['degenerate_face_ratio']):.6f}",
            f"loose vert ratio: {float(metrics['loose_vert_ratio']):.6f}",
            "",
            f"Distance score: {float(metrics['distance_score']):.4f}",
            f"Silhouette score: {float(metrics['silhouette_score']):.4f}",
            f"Alignment score: {float(metrics['alignment_score']):.4f}",
            f"Mesh health score: {float(metrics['mesh_health_score']):.4f}",
            f"Geometry score: {float(metrics['geometry_score']):.4f}",
        ]
        return "\n".join(lines)

    @classmethod
    def build_judge_bundle(
        cls,
        metrics: dict[str, Any],
        overlay_sheet: str,
        heatmap_sheet: str,
        silhouette_sheet: str,
    ) -> dict[str, Any]:
        return {
            "summary_text": cls.format_metrics_summary(metrics),
            "metrics": metrics,
            "images": {
                "overlay_sheet": overlay_sheet,
                "heatmap_sheet": heatmap_sheet,
                "silhouette_sheet": silhouette_sheet,
            },
        }

    @classmethod
    def judge_content_payload(cls, bundle: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": f"{bundle['summary_text']}",
            },
            {"type": "text", "text": "Overlay sheet"},
            {"type": "image_url", "image_url": {"url": cls.image_to_data_url(Path(bundle["images"]["overlay_sheet"]))}},
            {"type": "text", "text": "Error heatmap sheet"},
            {"type": "image_url", "image_url": {"url": cls.image_to_data_url(Path(bundle["images"]["heatmap_sheet"]))}},
            {"type": "text", "text": "Silhouette comparison sheet"},
            {"type": "image_url", "image_url": {"url": cls.image_to_data_url(Path(bundle["images"]["silhouette_sheet"]))}},
        ]

    @staticmethod
    def judge_questions() -> list[str]:
        return [
            "Does the candidate preserve the overall highpoly shape well enough to pass?",
            "Does the candidate preserve the main silhouettes across the sampled views well enough to pass?",
            "Does the candidate achieve a meaningful lowpoly reduction rather than effectively submitting the highpoly again?",
            "Are obvious visual artifacts or shape collapses absent enough for the result to pass?",
        ]

    @classmethod
    def _result_from_question_scores(
        cls,
        *,
        scores: list[float],
        reason: str,
        question_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        judge_score = cls._clamp_judge_score(100.0 * sum(scores) / len(scores))
        verdict = "yes" if judge_score >= 50.0 else "no"
        result = {
            "shape_fidelity": cls._clamp_judge_score(100.0 * scores[0]),
            "silhouette_preservation": cls._clamp_judge_score(100.0 * scores[1]),
            "triangle_efficiency": cls._clamp_judge_score(100.0 * scores[2]),
            "visible_artifacts": cls._clamp_judge_score(100.0 * scores[3]),
            "judge_score": judge_score,
            "verdict": verdict,
            "reason": reason,
        }
        if question_results is not None:
            result["question_results"] = question_results
        return result

    @classmethod
    def build_final_report(
        cls,
        *,
        run_status: str,
        gate_passed: bool,
        gate_fail_reasons: list[str],
        metrics: dict[str, Any],
        geometry_score: float,
        vlm_score: float,
        final_score: float,
        judge_model: str | None,
        evidence_paths: dict[str, str],
        judge_report_path: str | None,
        config: dict[str, Any],
        input_paths: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "run_status": run_status,
            "gate_passed": gate_passed,
            "gate_fail_reasons": gate_fail_reasons,
            "metrics": metrics,
            "geometry_score": float(geometry_score),
            "vlm_score": float(vlm_score),
            "final_score": float(final_score),
            "judge_model": judge_model,
            "judge_report_path": judge_report_path,
            "evidence_paths": evidence_paths,
            "config": config,
            "input_paths": input_paths,
        }

    @classmethod
    def write_markdown_summary(cls, report: dict[str, Any], output_path: Path) -> str:
        lines = [
            "# Lowpoly Evaluation Report",
            "",
            f"- Run status: `{report['run_status']}`",
            f"- Gate passed: `{report['gate_passed']}`",
            f"- Final score: `{report['final_score']:.4f}`",
            f"- Geometry score: `{report['geometry_score']:.4f}`",
            f"- VLM score: `{report['vlm_score']:.4f}`",
            f"- Judge model: `{report['judge_model']}`",
        ]
        if report["gate_fail_reasons"]:
            lines.append(f"- Gate fail reasons: `{', '.join(report['gate_fail_reasons'])}`")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(output_path)

    @staticmethod
    def _clamp_judge_score(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @classmethod
    def heuristic_judge(cls, bundle: dict[str, Any]) -> dict[str, Any]:
        metrics = bundle["metrics"]
        triangle_ratio = float(metrics["triangle_ratio"])
        distance_score = float(metrics["distance_score"])
        silhouette_score = float(metrics["silhouette_score"])
        mesh_health_score = float(metrics["mesh_health_score"])
        alignment_score = float(metrics["alignment_score"])

        question_scores = [
            1.0 if (0.65 * distance_score + 0.35 * alignment_score) >= 0.60 else 0.0,
            1.0 if silhouette_score >= 0.60 else 0.0,
            1.0 if triangle_ratio <= 0.05 else 0.0,
            1.0
            if (0.45 * distance_score + 0.25 * silhouette_score + 0.15 * mesh_health_score + 0.15 * alignment_score)
            >= 0.60
            else 0.0,
        ]
        question_results = [
            {
                "question": question,
                "result": "YES" if score >= 1.0 else "NO",
                "score": score,
                "raw_response": "heuristic",
            }
            for question, score in zip(cls.judge_questions(), question_scores)
        ]
        return cls._result_from_question_scores(
            scores=question_scores,
            reason="Heuristic judge based on geometry, silhouette, alignment, mesh health, and triangle ratio.",
            question_results=question_results,
        )

    @classmethod
    def openai_judge(cls, bundle: dict[str, Any], *, model: str, temperature: float) -> dict[str, Any]:
        from tasks.utils.evaluation import llm_multimodal_binary_questions_sync

        prompt_context = (
            "You are an expert 3D artist evaluating whether a lowpoly model preserves the highpoly shape and silhouette "
            "while using significantly fewer polygons.\n\n"
            "You will receive exactly three evidence images in this order: overlay sheet, error heatmap sheet, and "
            "silhouette comparison sheet. Use the metrics summary and images only. Judge each question independently "
            "using only YES or NO."
        )
        data = llm_multimodal_binary_questions_sync(
            prompt_context=prompt_context,
            questions=cls.judge_questions(),
            content=cls.judge_content_payload(bundle),
            model=model,
            max_tokens=32,
            temperature=temperature,
        )
        question_scores = [float(item["score"]) for item in data["results"]]
        return cls._result_from_question_scores(
            scores=question_scores,
            reason="OpenAI binary-question judge over evidence sheets.",
            question_results=data["results"],
        )

    @classmethod
    def run_judge(
        cls,
        *,
        judge_bundle_path: Path,
        output_dir: Path,
        backend: str,
        model: str,
        temperature: float,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = cls.load_json(judge_bundle_path)
        if not isinstance(bundle, dict):
            raise RuntimeError(f"Judge bundle must be an object: {judge_bundle_path}")
        if backend == "heuristic":
            judged = cls.heuristic_judge(bundle)
        else:
            judged = cls.openai_judge(bundle, model=model, temperature=temperature)
        report = {
            "summary": {
                "backend": backend,
                "model": model if backend == "openai" else None,
                "judge_score": float(judged["judge_score"]) / 100.0,
                "verdict": judged["verdict"],
            },
            "result": judged,
            "judge_bundle": str(judge_bundle_path),
        }
        cls.write_json(output_dir / "judge_report.json", report)
        return report

    @classmethod
    def write_stub_judge_report(cls, output_dir: Path, reason: str) -> str:
        report = {
            "summary": {
                "backend": None,
                "model": None,
                "judge_score": 0.0,
                "verdict": "no_pass",
            },
            "result": {
                "shape_fidelity": 0.0,
                "silhouette_preservation": 0.0,
                "triangle_efficiency": 0.0,
                "visible_artifacts": 0.0,
                "judge_score": 0.0,
                "verdict": "no_pass",
                "reason": reason,
            },
            "judge_bundle": None,
        }
        out_path = output_dir / "judge_report.json"
        cls.write_json(out_path, report)
        return str(out_path)

    @classmethod
    def determine_run_status(cls, *, gate_passed: bool, gate_fail_reasons: list[str], low_obj_exists: bool) -> str:
        if gate_passed:
            return "ok"
        if not low_obj_exists or any(reason in {"low_mesh_empty", "surface_sampling_failed"} for reason in gate_fail_reasons):
            return "invalid_submission"
        if any("No mesh objects imported" in reason for reason in gate_fail_reasons):
            return "invalid_submission"
        return "gate_failed"

    @staticmethod
    def zero_scores() -> dict[str, float]:
        return {
            "distance_score": 0.0,
            "silhouette_score": 0.0,
            "alignment_score": 0.0,
            "mesh_health_score": 0.0,
            "nonmanifold_score": 0.0,
            "distance_pass": 0.0,
            "silhouette_pass": 0.0,
            "alignment_pass": 0.0,
            "mesh_health_pass": 0.0,
            "mesh_health_raw": 0.0,
            "geometry_score": 0.0,
        }

    def run_blender_stage(self) -> Path:
        cmd = [
            self.blender_binary,
            "-b",
            "--factory-startup",
            "--python",
            str(Path(__file__).resolve()),
            "--",
            "--mode",
            "blender-stage",
            "--high-obj",
            str(self.paths.high_obj),
            "--low-obj",
            str(self.paths.low_obj),
            "--output-dir",
            str(self.output_dir),
            "--seed",
            str(self.seed),
        ]
        if self.paths.config_path.exists():
            cmd.extend(["--evaluation-config", str(self.paths.config_path)])
        subprocess.run(cmd, check=True)
        report_path = self.output_dir / "blender_eval.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Blender stage did not create {report_path}")
        return report_path

    @classmethod
    def load_blender_eval(cls, report_path: Path) -> dict[str, Any]:
        data = cls.load_json(report_path)
        if not isinstance(data, dict):
            raise RuntimeError(f"Blender evaluation report must be an object: {report_path}")
        return data

    @classmethod
    def build_heatmap_views(cls, raw_eval: dict[str, Any], output_dir: Path) -> dict[str, str]:
        points_path = Path(raw_eval["raw_evidence"]["heatmap_points"])
        payload = cls.load_json(points_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Heatmap projection payload must be an object: {points_path}")
        views_payload = payload.get("views", {})
        rendered: dict[str, str] = {}
        for view, base_path in raw_eval["raw_evidence"]["heatmap_base"].items():
            out_path = output_dir / f"{view}.png"
            rendered[view] = cls.render_heatmap_view(
                Path(base_path),
                list(views_payload.get(view, [])),
                out_path,
            )
        return rendered

    def build_missing_low_report(self) -> dict[str, Any]:
        report = self.build_final_report(
            run_status="invalid_submission",
            gate_passed=False,
            gate_fail_reasons=[f"low_obj_missing:{self.paths.low_obj}"],
            metrics={},
            geometry_score=0.0,
            vlm_score=0.0,
            final_score=0.0,
            judge_model=None,
            evidence_paths={},
            judge_report_path=self.write_stub_judge_report(
                self.output_dir / "judge_eval",
                "Judge skipped because low OBJ is missing.",
            ),
            config=self.config,
            input_paths={"high_obj": str(self.paths.high_obj), "low_obj": str(self.paths.low_obj)},
        )
        self.dump_json(self.output_dir / "final_report.json", report)
        self.write_markdown_summary(report, self.output_dir / "final_report.md")
        return report

    def build_failed_stage_report(
        self,
        *,
        raw_eval: dict[str, Any] | None,
        fallback_reason: str,
    ) -> dict[str, Any]:
        if raw_eval is not None:
            metrics = dict(raw_eval)
            metrics["silhouette_iou_mean"] = 0.0
            metrics.update(self.zero_scores())
            gate_fail_reasons = [str(reason) for reason in raw_eval.get("gate_fail_reasons", [])]
            report = self.build_final_report(
                run_status=self.determine_run_status(
                    gate_passed=False,
                    gate_fail_reasons=gate_fail_reasons,
                    low_obj_exists=self.paths.low_obj.exists(),
                ),
                gate_passed=False,
                gate_fail_reasons=gate_fail_reasons,
                metrics=metrics,
                geometry_score=0.0,
                vlm_score=0.0,
                final_score=0.0,
                judge_model=None,
                evidence_paths={},
                judge_report_path=self.write_stub_judge_report(
                    self.output_dir / "judge_eval",
                    "Judge skipped because Blender stage exited with validation failure.",
                ),
                config=self.config,
                input_paths={
                    "high_obj": str(self.paths.high_obj),
                    "low_obj": str(self.paths.low_obj),
                    "evaluation_config": str(self.paths.config_path),
                },
            )
            self.dump_json(self.output_dir / "metrics.json", metrics)
        else:
            report = self.build_final_report(
                run_status="evaluation_error",
                gate_passed=False,
                gate_fail_reasons=[fallback_reason],
                metrics={},
                geometry_score=0.0,
                vlm_score=0.0,
                final_score=0.0,
                judge_model=None,
                evidence_paths={},
                judge_report_path=self.write_stub_judge_report(
                    self.output_dir / "judge_eval",
                    "Judge skipped because Blender stage failed.",
                ),
                config=self.config,
                input_paths={"high_obj": str(self.paths.high_obj), "low_obj": str(self.paths.low_obj)},
            )
        self.dump_json(self.output_dir / "final_report.json", report)
        self.write_markdown_summary(report, self.output_dir / "final_report.md")
        return report

    def run(self) -> dict[str, Any]:
        if not self.paths.high_obj.exists():
            raise FileNotFoundError(f"High OBJ not found: {self.paths.high_obj}")
        if not self.paths.low_obj.exists():
            return self.build_missing_low_report()

        try:
            blender_report_path = self.run_blender_stage()
        except subprocess.CalledProcessError as exc:
            raw_path = self.output_dir / "blender_eval.json"
            raw_eval = self.load_blender_eval(raw_path) if raw_path.exists() else None
            report = self.build_failed_stage_report(
                raw_eval=raw_eval,
                fallback_reason=f"render_stage_failed:{exc}",
            )
            print(json.dumps(report, indent=2))
            return report

        raw_eval = self.load_blender_eval(blender_report_path)

        overlay_views = self.compose_overlay_views(
            raw_eval["raw_evidence"]["overlay_high"],
            raw_eval["raw_evidence"]["overlay_low"],
            self.output_dir / "evidence" / "overlay_views",
        )
        heatmap_views = self.build_heatmap_views(raw_eval, self.output_dir / "evidence" / "heatmap_views")
        silhouette_sheet_path, silhouette_iou_mean = self.compose_silhouette_sheet(
            raw_eval["raw_evidence"]["silhouette_high"],
            raw_eval["raw_evidence"]["silhouette_low"],
            self.output_dir / "evidence" / "silhouette_sheet.png",
        )
        overlay_sheet_path = self.compose_contact_sheet(overlay_views, self.output_dir / "evidence" / "overlay_sheet.png")
        heatmap_sheet_path = self.compose_contact_sheet(heatmap_views, self.output_dir / "evidence" / "heatmap_sheet.png")

        metrics = dict(raw_eval)
        metrics["silhouette_iou_mean"] = float(silhouette_iou_mean)
        metrics.update(self.compute_scores(metrics, self.config) if raw_eval.get("triangle_count_high") else self.zero_scores())
        metrics_path = self.dump_json(self.output_dir / "metrics.json", metrics)

        evidence_paths = {
            "overlay_sheet": overlay_sheet_path,
            "heatmap_sheet": heatmap_sheet_path,
            "silhouette_sheet": silhouette_sheet_path,
        }
        judge_bundle = self.build_judge_bundle(metrics, overlay_sheet_path, heatmap_sheet_path, silhouette_sheet_path)
        judge_bundle_path = self.dump_json(self.output_dir / "judge_bundle.json", judge_bundle)

        gate_passed = bool(raw_eval.get("gate_passed", False))
        gate_fail_reasons = [str(reason) for reason in raw_eval.get("gate_fail_reasons", [])]
        run_status = self.determine_run_status(
            gate_passed=gate_passed,
            gate_fail_reasons=gate_fail_reasons,
            low_obj_exists=self.paths.low_obj.exists(),
        )

        judge_model = self.judge_model_override or str(self.config["judge_model"])
        judge_weight = float(self.config.get("score_weights", {}).get("judge_score", 0.25))
        if self.skip_judge or not gate_passed:
            judge_report_path = self.write_stub_judge_report(
                self.output_dir / "judge_eval",
                "Judge skipped because gate failed." if not gate_passed else "Judge skipped by request.",
            )
            vlm_score = 0.0
        else:
            judge_report = self.run_judge(
                judge_bundle_path=Path(judge_bundle_path),
                output_dir=self.output_dir / "judge_eval",
                backend=self.judge_backend,
                model=judge_model,
                temperature=float(self.config.get("temperature", 0.0)),
            )
            judge_report_path = str((self.output_dir / "judge_eval" / "judge_report.json").resolve())
            vlm_score = float(judge_report["summary"]["judge_score"])

        geometry_score = float(metrics.get("geometry_score", 0.0)) if gate_passed else 0.0
        if not gate_passed:
            final_score = 0.0
        elif self.skip_judge:
            final_score = geometry_score
        else:
            final_score = (1.0 - judge_weight) * geometry_score + judge_weight * vlm_score

        report = self.build_final_report(
            run_status=run_status,
            gate_passed=gate_passed,
            gate_fail_reasons=gate_fail_reasons,
            metrics=metrics,
            geometry_score=geometry_score,
            vlm_score=vlm_score,
            final_score=final_score,
            judge_model=judge_model if not self.skip_judge else None,
            evidence_paths=evidence_paths,
            judge_report_path=judge_report_path,
            config=self.config,
            input_paths={
                "high_obj": str(self.paths.high_obj),
                "low_obj": str(self.paths.low_obj),
                "evaluation_config": str(self.paths.config_path),
                "metrics_path": metrics_path,
                "judge_bundle_path": judge_bundle_path,
            },
        )
        self.dump_json(self.output_dir / "final_report.json", report)
        self.write_markdown_summary(report, self.output_dir / "final_report.md")
        print(json.dumps(report, indent=2))
        return report

    @classmethod
    def blender_stage_main(
        cls,
        *,
        high_obj_path: Path,
        low_obj_path: Path,
        output_dir: Path,
        evaluation_config: str = "",
        seed: int = 1337,
    ) -> None:
        import bmesh  # type: ignore
        import bpy  # type: ignore
        import mathutils  # type: ignore
        from bpy_extras.object_utils import world_to_camera_view  # type: ignore
        from mathutils.bvhtree import BVHTree  # type: ignore
        from render_parts_dataset import (  # type: ignore
            clear_scene,
            fit_distance,
            get_or_create_camera_rig,
            import_obj,
            set_viewports_to_camera,
            setup_workbench,
        )

        def set_pose(rig: Any, cam: Any, center: Any, azimuth_deg: int, elevation_deg: float, distance: float) -> None:
            rig.location = center
            rig.rotation_euler = (0.0, 0.0, math.radians(azimuth_deg))
            elev_rad = math.radians(elevation_deg)
            cam.location = mathutils.Vector((0.0, -distance * math.cos(elev_rad), distance * math.sin(elev_rad)))
            bpy.context.view_layer.update()

        def import_obj_group(filepath: Path, object_name: str) -> Any:
            before = {obj.name for obj in bpy.data.objects}
            import_obj(filepath)
            imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
            if not imported:
                raise RuntimeError(f"No mesh objects imported from {filepath}")
            if len(imported) == 1:
                obj = imported[0]
                obj.name = object_name
                return obj

            bpy.ops.object.select_all(action="DESELECT")
            for obj in imported:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = imported[0]
            bpy.ops.object.join()
            obj = bpy.context.view_layer.objects.active
            obj.name = object_name
            return obj

        def all_mesh_objects() -> list[Any]:
            return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

        def set_visibility(visible: list[Any]) -> None:
            visible_names = {obj.name for obj in visible}
            for obj in all_mesh_objects():
                show = obj.name in visible_names
                obj.hide_set(not show)
                obj.hide_render = not show

        def set_colors(color_map: dict[str, tuple[float, float, float, float]]) -> None:
            for obj in all_mesh_objects():
                obj.color = color_map.get(obj.name, (0.8, 0.8, 0.8, 1.0))

        def duplicate_wireframe(obj: Any, *, name: str, thickness: float) -> Any:
            dup = obj.copy()
            dup.data = obj.data.copy()
            dup.name = name
            bpy.context.scene.collection.objects.link(dup)
            modifier = dup.modifiers.new(name="EvalWireframe", type="WIREFRAME")
            modifier.thickness = thickness
            modifier.use_even_offset = True
            modifier.use_relative_offset = False
            modifier.use_replace = True
            return dup

        def triangulated_mesh_data(obj: Any, mesh_name: str) -> dict[str, Any]:
            raw_mesh = obj.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(raw_mesh)
            bm.transform(obj.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

            tri_mesh = bpy.data.meshes.new(mesh_name)
            bm.to_mesh(tri_mesh)
            tri_mesh.calc_loop_triangles()
            # Blender 4.5 removed ``calc_normals_split``; loop triangle normals are
            # still available after the standard normal update.
            if hasattr(tri_mesh, "calc_normals_split"):
                tri_mesh.calc_normals_split()
            elif hasattr(tri_mesh, "calc_normals"):
                tri_mesh.calc_normals()

            verts = [vert.co.copy() for vert in tri_mesh.vertices]
            if not verts:
                bbox_min = mathutils.Vector((0.0, 0.0, 0.0))
                bbox_max = mathutils.Vector((0.0, 0.0, 0.0))
            else:
                bbox_min = mathutils.Vector(
                    (
                        min(vert.x for vert in verts),
                        min(vert.y for vert in verts),
                        min(vert.z for vert in verts),
                    )
                )
                bbox_max = mathutils.Vector(
                    (
                        max(vert.x for vert in verts),
                        max(vert.y for vert in verts),
                        max(vert.z for vert in verts),
                    )
                )

            triangle_count = len(tri_mesh.loop_triangles)
            nonmanifold_edge_count = sum(1 for edge in bm.edges if not edge.is_manifold)
            loose_vert_count = sum(1 for vert in bm.verts if len(vert.link_faces) == 0)
            degenerate_face_count = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)

            return {
                "mesh": tri_mesh,
                "bmesh": bm,
                "raw_mesh": raw_mesh,
                "bvh": BVHTree.FromBMesh(bm),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_center": (bbox_min + bbox_max) * 0.5,
                "bbox_size": bbox_max - bbox_min,
                "triangle_count": triangle_count,
                "nonmanifold_edge_count": nonmanifold_edge_count,
                "loose_vert_count": loose_vert_count,
                "degenerate_face_count": degenerate_face_count,
            }

        def cleanup_mesh_data(mesh_data: dict[str, Any] | None, owner: Any | None) -> None:
            if mesh_data is None or owner is None:
                return
            bm = mesh_data.get("bmesh")
            mesh = mesh_data.get("mesh")
            raw_mesh = mesh_data.get("raw_mesh")
            if bm is not None:
                bm.free()
            if mesh is not None:
                bpy.data.meshes.remove(mesh)
            if raw_mesh is not None:
                owner.to_mesh_clear()

        def sample_surface_points(mesh: Any, sample_count: int, rng: random.Random) -> list[dict[str, Any]]:
            triangles = mesh.loop_triangles
            if not triangles:
                return []
            cumulative_areas: list[float] = []
            total = 0.0
            for tri in triangles:
                a = mesh.vertices[tri.vertices[0]].co
                b = mesh.vertices[tri.vertices[1]].co
                c = mesh.vertices[tri.vertices[2]].co
                area = mathutils.geometry.area_tri(a, b, c)
                total += max(area, 1e-12)
                cumulative_areas.append(total)
            samples: list[dict[str, Any]] = []
            for _ in range(sample_count):
                target = rng.random() * total
                idx = bisect.bisect_left(cumulative_areas, target)
                tri = triangles[min(idx, len(triangles) - 1)]
                a = mesh.vertices[tri.vertices[0]].co
                b = mesh.vertices[tri.vertices[1]].co
                c = mesh.vertices[tri.vertices[2]].co
                r1 = math.sqrt(rng.random())
                r2 = rng.random()
                point = (1.0 - r1) * a + r1 * (1.0 - r2) * b + r1 * r2 * c
                normal = tri.normal.copy()
                if normal.length <= 1e-12:
                    normal = (b - a).cross(c - a)
                if normal.length > 1e-12:
                    normal.normalize()
                samples.append({"point": point, "normal": normal})
            return samples

        def distances_to_bvh(samples: list[dict[str, Any]], bvh: Any) -> list[float]:
            distances: list[float] = []
            for sample in samples:
                hit = bvh.find_nearest(sample["point"])
                nearest = hit[0]
                if nearest is None:
                    distances.append(1e9)
                    continue
                distances.append((sample["point"] - nearest).length)
            return distances

        def percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
            return float(ordered[idx])

        def mean_value(values: list[float]) -> float:
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        def render_image(
            *,
            filepath: Path,
            resolution: int,
            background: tuple[float, float, float],
            transparent: bool,
            visible: list[Any],
            color_map: dict[str, tuple[float, float, float, float]],
        ) -> None:
            scene = bpy.context.scene
            setup_workbench(background, "OBJECT", transparent=transparent, show_cavity=False, show_shadows=False)
            scene.render.resolution_x = resolution
            scene.render.resolution_y = resolution
            set_visibility(visible)
            set_colors(color_map)
            scene.render.filepath = str(filepath)
            bpy.ops.render.render(write_still=True)

        def project_points_for_view(*, scene: Any, cam: Any, samples: list[dict[str, Any]]) -> list[dict[str, float]]:
            cam_pos = cam.matrix_world.translation.copy()
            points: list[dict[str, float]] = []
            for sample in samples:
                world_point = sample["point"]
                coord = world_to_camera_view(scene, cam, world_point)
                if coord.z <= 0 or coord.x < 0 or coord.x > 1 or coord.y < 0 or coord.y > 1:
                    continue
                to_camera = cam_pos - world_point
                if to_camera.length <= 1e-12:
                    continue
                to_camera.normalize()
                normal = sample["normal"]
                if normal.length > 1e-12 and normal.dot(to_camera) <= 0:
                    continue
                points.append(
                    {
                        "x": float(coord.x),
                        "y": float(coord.y),
                        "distance_norm": float(sample["distance_norm"]),
                    }
                )
            return points

        def ensure_output_dirs(root: Path) -> dict[str, Path]:
            paths = {
                "root": root,
                "raw": root / "evidence" / "raw",
                "overlay_high": root / "evidence" / "raw" / "overlay" / "high",
                "overlay_low": root / "evidence" / "raw" / "overlay" / "low",
                "heatmap_base": root / "evidence" / "raw" / "heatmap" / "base",
                "sil_high": root / "evidence" / "raw" / "silhouette" / "high",
                "sil_low": root / "evidence" / "raw" / "silhouette" / "low",
            }
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
            return paths

        output_dir.mkdir(parents=True, exist_ok=True)
        dirs = ensure_output_dirs(output_dir)
        config = cls.load_config_from_path(evaluation_config or None)
        result: dict[str, Any] = {
            "gate_passed": False,
            "gate_fail_reasons": [],
            "config_used": config,
            "high_obj": str(high_obj_path),
            "low_obj": str(low_obj_path),
            "raw_evidence": {
                "overlay_high": {},
                "overlay_low": {},
                "heatmap_base": {},
                "silhouette_high": {},
                "silhouette_low": {},
                "heatmap_points": str(output_dir / "evidence" / "raw" / "heatmap_points.json"),
            },
        }

        high_obj = None
        low_obj = None
        low_wire = None
        high_data = None
        low_data = None
        try:
            clear_scene()
            high_obj = import_obj_group(high_obj_path, "HighPoly")
            low_obj = import_obj_group(low_obj_path, "LowPoly")
            high_data = triangulated_mesh_data(high_obj, "HighPolyEval")
            low_data = triangulated_mesh_data(low_obj, "LowPolyEval")

            diag = max((high_data["bbox_size"]).length, 1e-8)
            triangle_ratio = low_data["triangle_count"] / max(high_data["triangle_count"], 1)
            center_offset = (high_data["bbox_center"] - low_data["bbox_center"]).length
            size_delta = high_data["bbox_size"] - low_data["bbox_size"]
            axis_scale_diffs: list[float] = []
            for idx in range(3):
                ref = abs(high_data["bbox_size"][idx])
                axis_scale_diffs.append(abs(size_delta[idx]) / max(ref, 1e-8))
            max_bbox_axis_diff_ratio = max(axis_scale_diffs) if axis_scale_diffs else 0.0
            center_offset_norm = center_offset / diag

            if low_data["triangle_count"] <= 0:
                result["gate_fail_reasons"].append("low_mesh_empty")
            if triangle_ratio > float(config["triangle_ratio_cap"]):
                result["gate_fail_reasons"].append("triangle_ratio_cap_exceeded")
            if center_offset_norm > 0.02:
                result["gate_fail_reasons"].append("center_offset_too_large")
            if max_bbox_axis_diff_ratio > 0.05:
                result["gate_fail_reasons"].append("bbox_scale_diff_too_large")

            rng = random.Random(seed)
            sample_count = int(config["sample_count_per_mesh"])
            low_samples = sample_surface_points(low_data["mesh"], sample_count, rng)
            high_samples = sample_surface_points(high_data["mesh"], sample_count, rng)
            if not low_samples or not high_samples:
                result["gate_fail_reasons"].append("surface_sampling_failed")

            low_to_high = distances_to_bvh(low_samples, high_data["bvh"]) if low_samples else []
            high_to_low = distances_to_bvh(high_samples, low_data["bvh"]) if high_samples else []
            low_to_high_norm = [dist / diag for dist in low_to_high]
            high_to_low_norm = [dist / diag for dist in high_to_low]
            within_tolerance_ratio = mean_value([1.0 if dist <= 0.005 else 0.0 for dist in low_to_high_norm])

            for sample, dist_norm in zip(low_samples, low_to_high_norm):
                sample["distance_norm"] = dist_norm

            result.update(
                {
                    "triangle_count_high": int(high_data["triangle_count"]),
                    "triangle_count_low": int(low_data["triangle_count"]),
                    "triangle_ratio": float(triangle_ratio),
                    "high_bbox_diagonal": float(diag),
                    "chamfer_low_to_high_norm": float(mean_value(low_to_high_norm)),
                    "chamfer_high_to_low_norm": float(mean_value(high_to_low_norm)),
                    "p95_distance_norm": float(percentile(low_to_high_norm, 0.95)),
                    "within_tolerance_ratio": float(within_tolerance_ratio),
                    "bbox_scale_diff": float(mean_value(axis_scale_diffs)),
                    "max_bbox_axis_diff_ratio": float(max_bbox_axis_diff_ratio),
                    "center_offset_norm": float(center_offset_norm),
                    "nonmanifold_edge_count": int(low_data["nonmanifold_edge_count"]),
                    "degenerate_face_ratio": float(low_data["degenerate_face_count"] / max(low_data["triangle_count"], 1)),
                    "loose_vert_ratio": float(low_data["loose_vert_count"] / max(len(low_data["mesh"].vertices), 1)),
                }
            )

            scene = bpy.context.scene
            rig, cam = get_or_create_camera_rig()
            cam.data.type = "PERSP"
            cam.data.lens = 85.0
            cam.data.clip_start = 0.01
            cam.data.clip_end = 10000.0
            set_viewports_to_camera()

            high_points = [high_obj.matrix_world @ mathutils.Vector(corner) for corner in high_obj.bound_box]
            target_center = high_data["bbox_center"]
            wire_thickness = max(diag * 0.0015, 1e-4)
            low_wire = duplicate_wireframe(low_obj, name="LowPolyWire", thickness=wire_thickness)

            heatmap_projection_payload: dict[str, list[dict[str, float]]] = {}
            heatmap_samples = low_samples[: min(len(low_samples), 5000)]
            for view_name in config["views_evidence"]:
                spec = cls.VIEW_SPECS[view_name]
                distance = fit_distance(
                    scene,
                    rig,
                    cam,
                    target_center,
                    high_points,
                    int(spec["azimuth_deg"]),
                    math.radians(float(spec["elevation_deg"])),
                    0.80,
                )
                set_pose(rig, cam, target_center, int(spec["azimuth_deg"]), float(spec["elevation_deg"]), distance)

                overlay_high_path = dirs["overlay_high"] / f"{view_name}.png"
                overlay_low_path = dirs["overlay_low"] / f"{view_name}.png"
                heatmap_base_path = dirs["heatmap_base"] / f"{view_name}.png"
                result["raw_evidence"]["overlay_high"][view_name] = str(overlay_high_path)
                result["raw_evidence"]["overlay_low"][view_name] = str(overlay_low_path)
                result["raw_evidence"]["heatmap_base"][view_name] = str(heatmap_base_path)

                render_image(
                    filepath=overlay_high_path,
                    resolution=1024,
                    background=(1.0, 1.0, 1.0),
                    transparent=False,
                    visible=[high_obj],
                    color_map={high_obj.name: (0.60, 0.60, 0.60, 1.0)},
                )
                render_image(
                    filepath=overlay_low_path,
                    resolution=1024,
                    background=(1.0, 1.0, 1.0),
                    transparent=True,
                    visible=[low_wire],
                    color_map={low_wire.name: (0.10, 0.85, 0.20, 1.0)},
                )
                render_image(
                    filepath=heatmap_base_path,
                    resolution=1024,
                    background=(1.0, 1.0, 1.0),
                    transparent=False,
                    visible=[low_obj],
                    color_map={low_obj.name: (0.82, 0.82, 0.82, 1.0)},
                )
                heatmap_projection_payload[view_name] = project_points_for_view(scene=scene, cam=cam, samples=heatmap_samples)

            for view_name in config["views_metric"]:
                spec = cls.VIEW_SPECS[view_name]
                distance = fit_distance(
                    scene,
                    rig,
                    cam,
                    target_center,
                    high_points,
                    int(spec["azimuth_deg"]),
                    math.radians(float(spec["elevation_deg"])),
                    0.80,
                )
                set_pose(rig, cam, target_center, int(spec["azimuth_deg"]), float(spec["elevation_deg"]), distance)

                high_path = dirs["sil_high"] / f"{view_name}.png"
                low_path = dirs["sil_low"] / f"{view_name}.png"
                result["raw_evidence"]["silhouette_high"][view_name] = str(high_path)
                result["raw_evidence"]["silhouette_low"][view_name] = str(low_path)

                render_image(
                    filepath=high_path,
                    resolution=1024,
                    background=(1.0, 1.0, 1.0),
                    transparent=False,
                    visible=[high_obj],
                    color_map={high_obj.name: (0.0, 0.0, 0.0, 1.0)},
                )
                render_image(
                    filepath=low_path,
                    resolution=1024,
                    background=(1.0, 1.0, 1.0),
                    transparent=False,
                    visible=[low_obj],
                    color_map={low_obj.name: (0.0, 0.0, 0.0, 1.0)},
                )

            Path(result["raw_evidence"]["heatmap_points"]).write_text(
                json.dumps({"views": heatmap_projection_payload}, indent=2),
                encoding="utf-8",
            )
            result["gate_passed"] = not result["gate_fail_reasons"]
            Path(output_dir / "blender_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception as exc:
            result["gate_passed"] = False
            result["gate_fail_reasons"].append(str(exc))
            Path(output_dir / "blender_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise
        finally:
            cleanup_mesh_data(high_data, high_obj)
            cleanup_mesh_data(low_data, low_obj)
            if low_wire is not None and low_wire.name in bpy.data.objects:
                bpy.data.objects.remove(low_wire, do_unlink=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_argv = argv
    if cli_argv is None:
        cli_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run the lowpoly benchmark")
    parser.add_argument("--mode", default="run", choices=["run", "blender-stage"], help="Execution mode")
    parser.add_argument("--asset-root", default="", help="Asset root with input/high.obj and output/low.obj")
    parser.add_argument("--high-obj", default="", help="Optional explicit high.obj path")
    parser.add_argument("--low-obj", default="", help="Optional explicit low.obj path")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--evaluation-config", default="", help="Optional evaluation_config.json")
    parser.add_argument("--blender-binary", default=LowpolyBenchmark.DEFAULT_BLENDER, help="Path to Blender binary")
    parser.add_argument("--judge-backend", default="openai", choices=["heuristic", "openai"], help="Judge backend")
    parser.add_argument("--judge-model", default="", help="Optional judge model override")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge and leave vlm_score at zero")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    return parser.parse_args(cli_argv)


def main() -> None:
    args = parse_args()
    if args.mode == "blender-stage":
        LowpolyBenchmark.blender_stage_main(
            high_obj_path=Path(args.high_obj).expanduser().resolve(),
            low_obj_path=Path(args.low_obj).expanduser().resolve(),
            output_dir=Path(args.output_dir).expanduser().resolve(),
            evaluation_config=args.evaluation_config,
            seed=args.seed,
        )
        return

    benchmark = LowpolyBenchmark(
        output_dir=args.output_dir,
        asset_root=args.asset_root,
        high_obj=args.high_obj,
        low_obj=args.low_obj,
        evaluation_config=args.evaluation_config,
        blender_binary=args.blender_binary,
        judge_backend=args.judge_backend,
        judge_model=args.judge_model,
        skip_judge=args.skip_judge,
        seed=args.seed,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
