"""engineering/mfcadpp_feature_recognition: macro-F1 grader calibration and the three hooks with a fake session."""
from __future__ import annotations

import json

import pytest

from tasks.engineering.mfcadpp_feature_recognition import main as T
from tasks.engineering.mfcadpp_feature_recognition.scripts.grade import parse_prediction, score_labels
from tests.tasks._cad_fake_session import FakeSession

REF = {"p1": {"f000": 24, "f001": 24, "f002": 1, "f003": 1, "f004": 13, "f005": 13, "f006": 0}, "p2": {"f000": 24, "f001": 5, "f002": 5, "f003": 12}}


def test_perfect_all_stock_missing_part_and_garbage():
    assert score_labels(parse_prediction(json.dumps(REF)), REF)["score"] == 1.0
    assert score_labels({p: {f: 24 for f in faces} for p, faces in REF.items()}, REF)["score"] < 0.2
    assert 0.3 < score_labels({"p1": REF["p1"]}, REF)["score"] < 1.0
    assert score_labels(parse_prediction("not json"), REF)["score"] == 0.0 and score_labels(parse_prediction("[1,2]"), REF)["score"] == 0.0


@pytest.mark.asyncio
async def test_hooks():
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    files = {m["classes_path"]: "0 - Chamfer\n", m["readme_path"]: "x", m["reference_labels_path"]: json.dumps(REF)}
    with pytest.raises(RuntimeError):                      # reference visible during start -> refuse
        await T.start(t, FakeSession(files, {m["parts_dir"]: 30}))
    hidden = {k: v for k, v in files.items() if k != m["reference_labels_path"]}
    await T.start(t, FakeSession(hidden, {m["parts_dir"]: 30}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]                                   # no output
    files[m["output_labels_path"]] = json.dumps(REF); assert await T.evaluate(t, FakeSession(files)) == [1.0]
    files[m["output_labels_path"]] = "garbage"; assert await T.evaluate(t, FakeSession(files)) == [0.0]
