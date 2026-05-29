"""Score VAST skeleton CSV output against the hidden segmentation stack."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
import numpy as np
from PIL import Image

VAST_CSV_COLUMNS = [
    "ObjectIDNr",
    "NodeNr",
    "Flags",
    "NodeX",
    "NodeY",
    "NodeZ",
    "ParentNodeID",
    "ChildNode1ID",
    "ChildNode2ID",
    "Diameter",
    "NodeType",
    "ParentEdgeType",
    "ParentEdgeWeight",
    "EXT",
    "XFlags",
    "NodeName",
    "ParentEdgeName",
]

MASK_FILENAME_TEMPLATE = "gt_bbox_1_segmentation_mask_s{slice_num:04d}.png"
MASK_SLICE_NUMBERS = list(range(600, 651))
COVERAGE_THRESHOLD = 0.80
PURITY_THRESHOLD = 0.90


class EvaluationError(ValueError):
    """Raised when the predicted CSV or reference stack is malformed."""


@dataclass
class NodeRecord:
    skeleton_id: int
    node_id: int
    x: float
    y: float
    z: float


@dataclass
class SkeletonDiagnostic:
    skeleton_id: int
    total_nodes: int
    majority_gt_label: int
    majority_count: int
    purity: float
    precision_pass: bool


@dataclass
class LabelDiagnostic:
    gt_label: int
    visible_slice_count: int
    covered_slice_count: int
    coverage: float
    recall_pass: bool


@dataclass
class ScoreResult:
    score: float
    recall_pass: bool
    precision_pass: bool
    final_pass: bool
    skeleton_count: int
    gt_label_count: int
    skeleton_diagnostics: list[SkeletonDiagnostic]
    label_diagnostics: list[LabelDiagnostic]
    notes: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def _normalize_csv_text(text: str) -> list[list[str]]:
    rows = []
    reader = csv.reader(StringIO(text.lstrip("\ufeff")))
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        rows.append(row)
    return rows


def parse_vast_csv(text: str) -> dict[int, list[NodeRecord]]:
    rows = _normalize_csv_text(text)
    if not rows:
        return {}

    header = rows[0]
    data_rows = rows[1:] if header == VAST_CSV_COLUMNS else rows

    if not data_rows:
        return {}

    skeletons: dict[int, list[NodeRecord]] = defaultdict(list)
    for row_idx, row in enumerate(data_rows, start=1):
        if len(row) < len(VAST_CSV_COLUMNS):
            raise EvaluationError(
                f"row {row_idx} has {len(row)} columns; expected at least {len(VAST_CSV_COLUMNS)}"
            )
        try:
            skeleton_id = int(float(row[0]))
            node_id = int(float(row[1]))
            x = float(row[3])
            y = float(row[4])
            z = float(row[5])
        except ValueError as exc:
            raise EvaluationError(f"row {row_idx} contains non-numeric VAST coordinates or IDs") from exc
        skeletons[skeleton_id].append(
            NodeRecord(
                skeleton_id=skeleton_id,
                node_id=node_id,
                x=x,
                y=y,
                z=z,
            )
        )

    for nodes in skeletons.values():
        nodes.sort(key=lambda node: (round(node.z), node.node_id))
    return dict(sorted(skeletons.items()))


def load_reference_stack(reference_dir: Path) -> list[np.ndarray]:
    mask_dir = reference_dir / "mask_stack"
    masks: list[np.ndarray] = []
    for slice_num in MASK_SLICE_NUMBERS:
        path = mask_dir / MASK_FILENAME_TEMPLATE.format(slice_num=slice_num)
        if not path.exists():
            raise EvaluationError(f"missing reference mask: {path}")
        masks.append(np.asarray(Image.open(path)))
    return masks


def _normalize_z_convention(skeletons: dict[int, list[NodeRecord]]) -> tuple[dict[int, list[NodeRecord]], str]:
    rounded_values = [int(round(node.z)) for nodes in skeletons.values() for node in nodes]
    if not rounded_values:
        return skeletons, "empty"

    conventions = [
        (0, "zero_based"),
        (1, "one_based"),
        (600, "filename_offset_600"),
    ]
    for offset, name in conventions:
        normalized = [value - offset for value in rounded_values]
        if all(0 <= value <= 50 for value in normalized):
            normalized_skeletons: dict[int, list[NodeRecord]] = {}
            for skeleton_id, nodes in skeletons.items():
                normalized_skeletons[skeleton_id] = [
                    NodeRecord(
                        skeleton_id=node.skeleton_id,
                        node_id=node.node_id,
                        x=node.x,
                        y=node.y,
                        z=int(round(node.z)) - offset,
                    )
                    for node in nodes
                ]
            return normalized_skeletons, name

    raise EvaluationError(
        "predicted NodeZ values do not match any accepted slice convention "
        "(0..50, 1..51, or 600..650)"
    )


def _lookup_label(mask_stack: list[np.ndarray], node: NodeRecord) -> int:
    z = int(round(node.z))
    x = int(round(node.x))
    y = int(round(node.y))
    if z < 0 or z >= len(mask_stack):
        return 0
    mask = mask_stack[z]
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return 0
    return int(mask[y, x])


def _visible_slices_by_label(mask_stack: list[np.ndarray]) -> dict[int, set[int]]:
    visible: dict[int, set[int]] = defaultdict(set)
    for z, mask in enumerate(mask_stack):
        for label in np.unique(mask):
            label_int = int(label)
            if label_int != 0:
                visible[label_int].add(z)
    return dict(sorted(visible.items()))


def score_skeletons(
    skeletons: dict[int, list[NodeRecord]],
    mask_stack: list[np.ndarray],
) -> ScoreResult:
    skeletons, z_convention = _normalize_z_convention(skeletons)
    visible = _visible_slices_by_label(mask_stack)
    coverage_hits: dict[int, set[int]] = {label: set() for label in visible}
    skeleton_diagnostics: list[SkeletonDiagnostic] = []

    all_precision_pass = True
    for skeleton_id, nodes in skeletons.items():
        labels = [_lookup_label(mask_stack, node) for node in nodes]
        foreground = [label for label in labels if label != 0]
        majority_label = Counter(foreground).most_common(1)[0][0] if foreground else 0
        majority_count = sum(1 for label in labels if label == majority_label and majority_label != 0)
        purity = majority_count / len(nodes) if nodes else 0.0
        precision_pass = purity >= PURITY_THRESHOLD
        all_precision_pass &= precision_pass
        if majority_label != 0:
            for node, label in zip(nodes, labels, strict=True):
                if label == majority_label:
                    coverage_hits[majority_label].add(int(round(node.z)))
        skeleton_diagnostics.append(
            SkeletonDiagnostic(
                skeleton_id=skeleton_id,
                total_nodes=len(nodes),
                majority_gt_label=majority_label,
                majority_count=majority_count,
                purity=purity,
                precision_pass=precision_pass,
            )
        )

    label_diagnostics: list[LabelDiagnostic] = []
    all_recall_pass = True
    for label, visible_slices in visible.items():
        covered = coverage_hits[label]
        coverage = len(covered) / len(visible_slices) if visible_slices else 0.0
        recall_pass = coverage >= COVERAGE_THRESHOLD
        all_recall_pass &= recall_pass
        label_diagnostics.append(
            LabelDiagnostic(
                gt_label=label,
                visible_slice_count=len(visible_slices),
                covered_slice_count=len(covered),
                coverage=coverage,
                recall_pass=recall_pass,
            )
        )

    notes: list[str] = []
    if not skeleton_diagnostics:
        notes.append("no predicted skeletons")
    if not all_precision_pass:
        notes.append("one or more predicted skeletons failed purity")
    if not all_recall_pass:
        notes.append("one or more GT neurons failed coverage")
    if z_convention != "zero_based":
        notes.append(f"normalized NodeZ from {z_convention} convention")

    final_pass = all_precision_pass and all_recall_pass and bool(skeleton_diagnostics)
    return ScoreResult(
        score=1.0 if final_pass else 0.0,
        recall_pass=all_recall_pass,
        precision_pass=all_precision_pass,
        final_pass=final_pass,
        skeleton_count=len(skeleton_diagnostics),
        gt_label_count=len(label_diagnostics),
        skeleton_diagnostics=skeleton_diagnostics,
        label_diagnostics=label_diagnostics,
        notes=notes,
    )


def score_submission_file(output_file: Path, reference_dir: Path) -> ScoreResult:
    if not output_file.exists():
        raise EvaluationError(f"missing predicted CSV: {output_file}")
    skeletons = parse_vast_csv(output_file.read_text(encoding="utf-8"))
    mask_stack = load_reference_stack(reference_dir)
    return score_skeletons(skeletons, mask_stack)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Score a VAST skeleton CSV against a mask stack.")
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = score_submission_file(args.submission, args.reference_dir)
    except Exception as exc:
        print(json.dumps({"score": 0.0, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.final_pass else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
