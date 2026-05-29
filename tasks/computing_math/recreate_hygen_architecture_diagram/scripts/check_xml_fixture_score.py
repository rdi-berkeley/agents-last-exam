#!/usr/bin/env python
"""Score a Draw.io XML against the normalized Stage 1 rubric."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

SYNTHETIC_LABELS = [
    "stream-a buffer",
    "stream-b buffer",
    "alpha dispatcher",
    "beta dispatcher",
    "delta estimator",
    "unified core",
]

SCORED_PATHS = [
    ("stream-a buffer", "alpha dispatcher"),
    ("stream-b buffer", "beta dispatcher"),
    ("alpha dispatcher", "unified core"),
    ("delta estimator", "unified core"),
    ("unified core", "workers"),
]

FORBIDDEN_ORIGINAL_LABELS = [
    "online request queue",
    "offline request queue",
    "online selector",
    "offline selector",
    "latency predictor",
    "continuous batch",
]


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.sub("<.*?>", " ", text).split()).lower()


def evaluate(xml_path: Path) -> dict[str, object]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    graph_model = root if root.tag == "mxGraphModel" else root.find(".//mxGraphModel")
    if graph_model is None:
        return {"score": 0.0, "gate_failed": "missing_mxGraphModel"}

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    has_image = False

    for cell in root.iter("mxCell"):
        cid = cell.get("id")
        style = cell.get("style", "")
        if "image=data:image" in style:
            has_image = True
        if cell.get("vertex") == "1" and cid:
            nodes[cid] = strip_html(cell.get("value"))
        elif cell.get("edge") == "1":
            source = cell.get("source")
            target = cell.get("target")
            if source and target:
                edges.append((source, target))

    found_labels = [
        label for label in SYNTHETIC_LABELS
        if any(label in text for text in nodes.values())
    ]
    mapping_score = len(found_labels) / len(SYNTHETIC_LABELS)

    forbidden_labels = [
        label for label in FORBIDDEN_ORIGINAL_LABELS
        if any(label in text for text in nodes.values())
    ]
    if forbidden_labels:
        mapping_score = 0.0

    path_matches = 0
    for source_label, target_label in SCORED_PATHS:
        source_ids = [node_id for node_id, text in nodes.items() if source_label in text]
        target_ids = [node_id for node_id, text in nodes.items() if target_label in text]
        if any(source in source_ids and target in target_ids for source, target in edges):
            path_matches += 1
    topology_score = path_matches / len(SCORED_PATHS)
    asset_score = 1.0 if has_image else 0.0
    final_score = (mapping_score * 0.3) + (topology_score * 0.5) + (asset_score * 0.2)

    return {
        "score": final_score,
        "mapping_score": mapping_score,
        "found_labels": found_labels,
        "forbidden_labels_present": forbidden_labels,
        "topology_score": topology_score,
        "path_matches": path_matches,
        "asset_score": asset_score,
        "has_image": has_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.xml_path), indent=2))


if __name__ == "__main__":
    main()
