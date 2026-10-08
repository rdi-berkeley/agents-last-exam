import copy
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_judge
from tasks.visual_media.project_migration.replay_audio import replay_evaluation


@pytest.fixture
def float_pair(tmp_path):
    times = np.arange(960) / 48000
    samples = np.column_stack(
        (0.1 * np.sin(2 * np.pi * 440 * times), 0.07 * np.cos(2 * np.pi * 330 * times))
    )
    reference, delivered, native = [
        tmp_path / name for name in ("reference.wav", "delivery.wav", "native.wav")
    ]
    sf.write(reference, samples[::-1], 48000, subtype="FLOAT")
    sf.write(delivered, samples, 48000, format="WAV", subtype="FLOAT")
    sf.write(native, samples, 48000, format="WAVEX", subtype="FLOAT")
    assert delivered.read_bytes() != native.read_bytes()
    return reference, delivered, native


def raw_reply(delivery_rating, native_rating):
    component = {
        "assessable": True,
        "candidate_character": "pitched acoustic tone",
        "agreements": ["shared attack"],
        "differences": ["different sustain"],
        "limitations": [],
    }
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "reference_character": "sustained tone",
                            "candidate_a": {**component, "rating": delivery_rating},
                            "candidate_b": {**component, "rating": native_rating},
                        }
                    )
                }
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("ratings", [(1, 4), (3, 2)])
async def test_different_float_containers_reuse_delivery_not_best(float_pair, tmp_path, ratings):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=raw_reply(*ratings)))
    result = await audio_judge.judge_window(
        *float_pair, tmp_path / "judge.json", transport=transport
    )
    assert result["score"] == ratings[0] / 4
    assert result["native"]["score"] == result["delivery"]["score"]
    assert result["native"]["reused_identical_audio"]
    assert result["native"]["reused_exact_decoded_pcm"]
    assert result["raw"] == raw_reply(*ratings)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["one_ulp", "rate", "frames", "channels"])
async def test_decoded_identity_rejects_near_or_structurally_different_audio(
    float_pair, tmp_path, change
):
    _, delivered, native = float_pair
    samples, rate = sf.read(delivered, always_2d=True, dtype="float32")
    if change == "one_ulp":
        samples[100, 0] = np.nextafter(samples[100, 0], np.float32(1))
    elif change == "rate":
        rate = 44100
    elif change == "frames":
        samples = samples[:-1]
    else:
        samples = samples[:, :1]
    sf.write(native, samples, rate, format="WAVEX", subtype="FLOAT")
    result = await audio_judge.judge_window(
        *float_pair,
        tmp_path / "judge.json",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw_reply(3, 1))),
    )
    assert result["score"] == 0.25
    assert not result["native"].get("reused_identical_audio")
    assert not result["native"].get("reused_exact_decoded_pcm")


@pytest.mark.asyncio
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
async def test_equal_nonfinite_audio_is_rejected(float_pair, tmp_path, nonfinite):
    samples, rate = sf.read(float_pair[1], always_2d=True)
    samples[100, 0] = nonfinite
    for path, container in zip(float_pair[1:], ("WAV", "WAVEX"), strict=True):
        sf.write(path, samples, rate, format=container, subtype="FLOAT")

    def forbidden(request):
        pytest.fail("Nonfinite audio must fail before a provider request")

    with pytest.raises(JudgeInfrastructureError, match="Invalid audio"):
        await audio_judge.judge_window(
            *float_pair, tmp_path / "invalid.json", transport=httpx.MockTransport(forbidden)
        )


@pytest.fixture
def retained_evaluation():
    path = os.environ.get("MIGRATION_JOINT_EVALUATION")
    if not path:
        pytest.skip("Retained R2 witnesses require MIGRATION_JOINT_EVALUATION")
    evaluation = json.loads(Path(path).read_text())
    return next(
        entry["result"]
        for entry in evaluation["evaluations"]
        if entry["identifier"] == "audio_semantic_rubric"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [0, 1, 2])
async def test_actual_mallets_exact_float_witnesses(retained_evaluation, tmp_path, window):
    stem = next(
        stem
        for stem in retained_evaluation["delivery"]["stems"]
        if stem["reference"] == "Percussion Mallets.flac"
    )
    passage, old = stem["passages"][window], stem["judgments"][window]
    paths = [Path(passage[key]) for key in ("reference_path", "candidate_path", "native_path")]
    assert paths[1].read_bytes() != paths[2].read_bytes()
    before, before_rate = sf.read(paths[1], always_2d=True)
    after, after_rate = sf.read(paths[2], always_2d=True)
    assert before_rate == after_rate and np.array_equal(before, after)
    assert np.isfinite(before).all() and old["score"] == 0.5

    async def recorded(*args, **kwargs):
        return {
            key: copy.deepcopy(old[key]) for key in ("raw", "normalization_gains", "request_sha256")
        }

    result = await audio_judge.judge_window(
        *paths, tmp_path / "witness.json", comparison_request=recorded
    )
    assert result["score"] == result["delivery"]["score"] == result["native"]["score"] == 0.75
    assert result["native"]["reused_exact_decoded_pcm"]
    assert result["raw"] == old["raw"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tampered", [False, True])
async def test_recorded_replay_is_offline_preserves_sources_and_checks_bindings(
    float_pair, tmp_path, monkeypatch, tampered
):
    delivery = {
        "delivery_score": 1,
        "stems": [
            {
                "reference": "control.wav",
                "reference_audio": {"audible": True},
                "valid": True,
                "passages": [
                    dict(
                        zip(
                            ("reference_path", "candidate_path", "native_path"),
                            map(str, float_pair),
                            strict=True,
                        )
                    )
                ],
            }
        ],
    }
    scored = await audio_judge.score_delivery(
        delivery,
        tmp_path,
        window_judge=partial(
            audio_judge.judge_window,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=raw_reply(1, 4))
            ),
        ),
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(scored))
    if tampered:
        path = tmp_path / "judge-000-0.request.json"
        value = json.loads(path.read_text())
        value["normalized_audio_sha256"][1] = "wrong"
        path.write_text(json.dumps(value))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Replay must never create an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    destination = tmp_path / "correction"
    if tampered:
        with pytest.raises(JudgeInfrastructureError, match="does not bind"):
            await replay_evaluation(source, destination)
        assert not (destination / "result.json").exists()
    else:
        result = await replay_evaluation(source, destination)
        assert result["weighted_score"] == pytest.approx(0.4)
        assert result["exact_weighted_score"] == "2/5"
        assert result["recorded_responses_used"] == 1 and result["api_calls"] == 0
        assert result["changed_windows"] == []
        assert len(list(destination.glob("*.response-*.json"))) == 1
        with pytest.raises(FileExistsError):
            await replay_evaluation(source, destination)
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items()
    )
