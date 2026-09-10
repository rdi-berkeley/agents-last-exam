"""engineering/phm2010_tool_wear: MAE-to-score mapping, gates, and the three hooks with a fake session."""
from __future__ import annotations

import random

import pytest

from tasks.engineering.phm2010_tool_wear import main as T
from tasks.engineering.phm2010_tool_wear.scripts.grade import FULL_CREDIT_MAE_UM, ZERO_CREDIT_MAE_UM, parse_wear_csv, score_wear
from tests.tasks._cad_fake_session import FakeSession


def _wear(n=315, seed=0):
    rng = random.Random(seed); return {c: (30 + c * 0.4 + rng.random(), 40 + c * 0.35, 35 + c * 0.45) for c in range(1, n + 1)}


def _csv(d):
    return "cut,flute_1,flute_2,flute_3\n" + "\n".join(f"{c},{a},{b},{e}" for c, (a, b, e) in sorted(d.items())) + "\n"


REF = _wear()


def test_perfect_noisy_constant_and_gates():
    assert score_wear(parse_wear_csv(_csv(REF), 315), REF)["score"] == 1.0
    pred = {c: tuple(v + 20 for v in vals) for c, vals in REF.items()}
    r = score_wear(parse_wear_csv(_csv(pred), 315), REF)
    assert abs(r["mae_um"] - 20) < 1e-6 and abs(r["score"] - (ZERO_CREDIT_MAE_UM - 20) / (ZERO_CREDIT_MAE_UM - FULL_CREDIT_MAE_UM)) < 1e-4
    mean = sum(sum(v) for v in REF.values()) / (3 * 315)
    assert score_wear(parse_wear_csv(_csv({c: (mean, mean, mean) for c in REF}), 315), REF)["score"] < 0.2
    assert score_wear(parse_wear_csv("cut,flute_1\n1,2\n", 315), REF)["score"] == 0.0
    assert score_wear(parse_wear_csv(_csv(dict(list(REF.items())[:300])), 315), REF)["score"] == 0.0
    assert score_wear(parse_wear_csv(_csv(REF).replace("\n2,", "\n2,nan,", 1), 315), REF)["score"] == 0.0


@pytest.mark.asyncio
async def test_hooks():
    t = T.load()[0]; m = t.metadata; assert m["reference_dir"] not in t.description
    ref_csv = _csv(REF)
    files = {f"{m['train_dir']}/c1/wear.csv": "x", f"{m['train_dir']}/c6/wear.csv": "x", m["readme_path"]: "x", m["reference_csv"]: ref_csv}
    with pytest.raises(RuntimeError):
        await T.start(t, FakeSession(files, {m["test_cuts_dir"]: 315}))
    hidden = {k: v for k, v in files.items() if k != m["reference_csv"]}
    await T.start(t, FakeSession(hidden, {m["test_cuts_dir"]: 315}))
    with pytest.raises(RuntimeError):                      # too few cut files -> staging error
        await T.start(t, FakeSession(hidden, {m["test_cuts_dir"]: 10}))
    assert await T.evaluate(t, FakeSession(files)) == [0.0]
    files[m["output_csv"]] = ref_csv; assert await T.evaluate(t, FakeSession(files)) == [1.0]
    files[m["output_csv"]] = ref_csv.replace("\n2,", "\n2,x,", 1); assert await T.evaluate(t, FakeSession(files)) == [0.0]
