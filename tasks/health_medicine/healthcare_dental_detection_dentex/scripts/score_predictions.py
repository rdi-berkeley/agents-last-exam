"""Score a DENTEX prediction file against hidden COCO ground truth."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VALID_CATEGORY_IDS = {0, 1, 2, 3}
REQUIRED_KEYS = {"image_id", "category_id", "bbox", "score"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(reason: str, *, prediction_count: int = 0, summary: str = "") -> dict[str, Any]:
    return {
        "score": 0.0,
        "ap50": 0.0,
        "passed_snapshot_threshold": False,
        "prediction_count": prediction_count,
        "reason": reason,
        "summary": summary,
    }


def _validate_predictions(predictions: Any, allowed_image_ids: set[int]) -> tuple[bool, str]:
    if not isinstance(predictions, list):
        return False, "predictions_json_must_be_a_list"

    for index, detection in enumerate(predictions):
        prefix = f"prediction_{index}"
        if not isinstance(detection, dict):
            return False, f"{prefix}_must_be_an_object"

        missing_keys = REQUIRED_KEYS - set(detection)
        if missing_keys:
            return False, f"{prefix}_missing_keys:{','.join(sorted(missing_keys))}"

        image_id = detection["image_id"]
        if not isinstance(image_id, int):
            return False, f"{prefix}_image_id_must_be_int"
        if image_id not in allowed_image_ids:
            return False, f"{prefix}_image_id_out_of_range"

        category_id = detection["category_id"]
        if not isinstance(category_id, int):
            return False, f"{prefix}_category_id_must_be_int"
        if category_id not in VALID_CATEGORY_IDS:
            return False, f"{prefix}_category_id_out_of_range"

        bbox = detection["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, f"{prefix}_bbox_must_be_length_4_list"
        try:
            bbox_values = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return False, f"{prefix}_bbox_must_be_numeric"
        if any(not math.isfinite(value) for value in bbox_values):
            return False, f"{prefix}_bbox_must_be_finite"
        if bbox_values[2] <= 0 or bbox_values[3] <= 0:
            return False, f"{prefix}_bbox_width_height_must_be_positive"

        score = detection["score"]
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            return False, f"{prefix}_score_must_be_numeric"
        if not math.isfinite(score_value):
            return False, f"{prefix}_score_must_be_finite"
        if score_value < 0 or score_value > 1:
            return False, f"{prefix}_score_must_be_between_0_and_1"

    return True, "ok"


def _compute_ap50(reference_json: Path, predictions_json: Path) -> tuple[float, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_gt = COCO(str(reference_json))
        coco_dt = coco_gt.loadRes(str(predictions_json))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return float(coco_eval.stats[1]), buffer.getvalue().strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--pass-threshold", type=float, default=0.10)
    args = parser.parse_args()

    reference_path = Path(args.reference_json)
    predictions_path = Path(args.predictions_json)

    reference = _load_json(reference_path)
    predictions = _load_json(predictions_path)

    allowed_image_ids = {int(image["id"]) for image in reference.get("images", [])}
    valid, reason = _validate_predictions(predictions, allowed_image_ids)
    if not valid:
        print(
            json.dumps(
                _fail(
                    reason,
                    prediction_count=len(predictions) if isinstance(predictions, list) else 0,
                )
            )
        )
        return 0

    if not predictions:
        print(json.dumps(_fail("no_predictions", prediction_count=0)))
        return 0

    ap50, summary = _compute_ap50(reference_path, predictions_path)
    payload = {
        "score": ap50,
        "ap50": ap50,
        "passed_snapshot_threshold": ap50 >= float(args.pass_threshold),
        "prediction_count": len(predictions),
        "reason": "ok",
        "summary": summary,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
