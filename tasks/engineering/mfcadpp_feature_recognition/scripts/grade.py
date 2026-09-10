"""Grader for engineering/mfcadpp_feature_recognition.

Reference: {part_id: {face_id: class_int}} for every ADVANCED_FACE in every staged part.
Prediction: same shape, produced by the agent at output/labels.json. Missing parts or faces count as wrong.
Score = macro-F1 over the classes present in the reference (25 classes: 0-23 machining features, 24 = stock face).
Macro-F1 keeps the trivial "everything is stock" answer near zero while a complete, correct labelling scores 1.0.
"""
from __future__ import annotations

import json
from collections import defaultdict


def parse_prediction(text: str) -> dict[str, dict[str, int]] | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict[str, int]] = {}
    for part, faces in data.items():
        if not isinstance(faces, dict):
            continue
        clean = {}
        for fid, lab in faces.items():
            try:
                clean[str(fid)] = int(lab)
            except (TypeError, ValueError):
                continue
        out[str(part)] = clean
    return out


def score_labels(pred: dict[str, dict[str, int]] | None, ref: dict[str, dict[str, int]]) -> dict:
    total = sum(len(f) for f in ref.values())
    if not pred or total == 0:
        return {"score": 0.0, "macro_f1": 0.0, "accuracy": 0.0, "faces": total, "labelled": 0, "per_part": {}}
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0; labelled = 0; per_part = {}
    for part, faces in ref.items():
        p = pred.get(part, {}); ok = 0
        for fid, truth in faces.items():
            guess = p.get(fid)
            if guess is None:
                fn[truth] += 1
                continue
            labelled += 1
            if guess == truth:
                tp[truth] += 1; ok += 1; correct += 1
            else:
                fp[guess] += 1; fn[truth] += 1
        per_part[part] = round(ok / len(faces), 4) if faces else 0.0
    classes = sorted({c for faces in ref.values() for c in faces.values()})
    f1s = []
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return {"score": round(max(0.0, min(1.0, macro)), 4), "macro_f1": round(macro, 4), "accuracy": round(correct / total, 4), "faces": total, "labelled": labelled, "classes": len(classes), "per_part": per_part}
