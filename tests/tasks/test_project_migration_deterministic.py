"""Musical invariants and clear counterexamples, without fitting to solver scores.

Encoding-only positives should retain at least 95% similarity. Deliberate noise,
reordering and struck-to-sustained changes must lose at least half the credit.
"""

import asyncio
import copy
import json
import socket

import httpx
import numpy as np
import pytest
import soundfile as sf

from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_score
from tasks.visual_media.project_migration.scripts import audio_metrics


RATE = 48000
NOTE_SECONDS = 0.7
PITCHES = (220.0, 329.6276, 261.6256, 440.0, 391.9954, 293.6648)
VELOCITIES = (0.8, 0.35, 0.65, 0.45, 0.9, 0.55)


def musical_notes(rate=RATE, *, sustained=False):
    time = np.arange(round(NOTE_SECONDS * rate), dtype=np.float64) / rate
    local_time = np.maximum(time - 0.03, 0)
    notes = []
    for index, (pitch, velocity) in enumerate(zip(PITCHES, VELOCITIES, strict=True)):
        if sustained:
            attack = np.sin(np.minimum(local_time / 0.18, 1) * np.pi / 2) ** 2
            release = np.clip((0.62 - local_time) / 0.08, 0, 1)
            envelope = attack * release
        else:
            envelope = (1 - np.exp(-local_time / 0.003)) * np.exp(
                -local_time / (0.055 + 0.018 * (index % 3))
            )
            envelope *= np.clip((0.62 - local_time) / 0.03, 0, 1)
        phase = 2 * np.pi * pitch * local_time
        wave = sum(
            amplitude * np.sin(harmonic * phase)
            for harmonic, amplitude in ((1, 1.0), (2, 0.32), (3, 0.16), (5, 0.07), (7, 0.025))
        )
        notes.append(0.2 * velocity * envelope * wave)
    mono = np.concatenate(notes)
    return np.column_stack((0.8 * mono, 0.6 * mono))


def write_audio(path, samples, rate=RATE, *, subtype="FLOAT"):
    sf.write(path, samples, rate, subtype=subtype)
    return path


@pytest.fixture
def phrase():
    return musical_notes()


@pytest.fixture
def reference_file(tmp_path, phrase):
    return write_audio(tmp_path / "reference.wav", phrase)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Deterministic audio verification attempted network access")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.mark.parametrize(
    "change",
    ["identity", "quiet_gain", "loud_gain", "faint", "pan", "antiphase", "mono", "dc_offset"],
)
def test_feature_similarity_preserves_nonmusical_changes(phrase, change):
    mono = phrase[:, 0] / 0.8
    candidates = {
        "identity": phrase.copy(),
        "quiet_gain": phrase * 0.05,
        "loud_gain": phrase * 3,
        "faint": phrase * 1e-12,
        "pan": np.column_stack((0.15 * mono, 0.95 * mono)),
        "antiphase": phrase * np.array([1, -1]),
        "mono": mono[:, None],
        "dc_offset": phrase + np.array([0.15, -0.2]),
    }
    result = audio_metrics.compare_audio(phrase, RATE, candidates[change], RATE)
    assert 0.99 <= result["score"] <= 1, result


@pytest.mark.parametrize("candidate_rate", [22050, 44100, 96000])
@pytest.mark.parametrize("dc_offset", [0.0, 0.2])
def test_same_music_at_other_sample_rates_preserves_similarity(phrase, candidate_rate, dc_offset):
    candidate = musical_notes(candidate_rate) + dc_offset
    result = audio_metrics.compare_audio(phrase, RATE, candidate, candidate_rate)
    assert 0.95 <= result["score"] <= 1, result


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT"])
def test_pcm_depth_is_not_a_timbre_penalty(reference_file, phrase, tmp_path, subtype):
    candidate = write_audio(tmp_path / "candidate.wav", phrase, subtype=subtype)
    result = audio_score.score_window(reference_file, candidate, candidate, tmp_path / "score.json")
    assert result["score"] >= 0.95, result
    assert result["delivery"]["score"] == result["native"]["score"]


def test_faint_float_delivery_remains_audible(reference_file, phrase, tmp_path):
    candidate = write_audio(tmp_path / "faint.wav", phrase * 1e-12)
    result = audio_score.score_window(reference_file, candidate, candidate, tmp_path / "score.json")
    assert result["score"] >= 0.99, result


@pytest.mark.parametrize("constant", [0.0, 5e-10, -0.125])
def test_silence_and_dc_are_zero_against_active_music(reference_file, phrase, tmp_path, constant):
    candidate = write_audio(tmp_path / "constant.wav", np.full_like(phrase, constant))
    result = audio_score.score_window(reference_file, candidate, candidate, tmp_path / "score.json")
    assert result["delivery"]["score"] == 0
    assert result["native"]["score"] == 0
    assert result["score"] == 0


@pytest.mark.parametrize("change", ["noise", "reordered_notes", "reversed_time", "wrong_attack"])
def test_clear_musical_changes_do_not_receive_high_similarity(phrase, change):
    if change == "noise":
        candidate = np.random.default_rng(732).normal(size=phrase.shape)
        candidate *= np.sqrt(np.mean(phrase**2) / np.mean(candidate**2))
    elif change == "reordered_notes":
        slots = phrase.reshape(len(PITCHES), -1, 2)
        candidate = slots[[3, 5, 1, 4, 0, 2]].reshape(phrase.shape)
    elif change == "reversed_time":
        candidate = phrase[::-1].copy()
    else:
        candidate = musical_notes(sustained=True)
    result = audio_metrics.compare_audio(phrase, RATE, candidate, RATE)
    assert 0 <= result["score"] < 0.5, result


def test_time_varying_dynamics_are_not_treated_as_global_gain(phrase):
    gain = np.repeat([0.05, 1.0, 0.1, 1.0, 0.1, 1.0], round(NOTE_SECONDS * RATE))
    changed = audio_metrics.compare_audio(phrase, RATE, phrase * gain[:, None], RATE)
    unchanged = audio_metrics.compare_audio(phrase, RATE, phrase * 0.1, RATE)
    assert unchanged["score"] >= 0.99
    assert changed["score"] < 0.85, changed


@pytest.mark.parametrize("wrong_side", ["delivery", "native"])
def test_one_correct_recording_cannot_mask_wrong_other_recording(
    reference_file, tmp_path, wrong_side
):
    wrong = write_audio(tmp_path / "wrong-attack.wav", musical_notes(sustained=True))
    delivered, native = (
        (wrong, reference_file) if wrong_side == "delivery" else (reference_file, wrong)
    )
    result = audio_score.score_window(reference_file, delivered, native, tmp_path / "score.json")
    correct_side = "native" if wrong_side == "delivery" else "delivery"
    assert result[correct_side]["score"] == 1
    assert result[wrong_side]["score"] < 0.5, result
    assert result["score"] == result[wrong_side]["score"]


def test_missing_active_stems_stay_in_denominator(reference_file, tmp_path):
    passage = {
        "reference_path": str(reference_file),
        "candidate_path": str(reference_file),
        "native_path": str(reference_file),
    }
    delivery = {
        "delivery_score": 0.5,
        "stems": [
            {"valid": True, "reference_audio": {"audible": True}, "passages": [passage]},
            {"valid": False, "reference_audio": {"audible": True}, "passages": []},
            {"valid": False, "reference_audio": {"audible": False}, "passages": []},
        ],
    }
    result = asyncio.run(audio_score.score_delivery(delivery, tmp_path / "receipts"))
    assert result["timbre_score"] == pytest.approx(0.5)
    assert result["weighted_score"] == pytest.approx(0.5)


def test_windows_are_averaged_within_stem_before_stem_average(reference_file, tmp_path):
    silence = write_audio(
        tmp_path / "silent.wav", np.zeros((round(len(PITCHES) * NOTE_SECONDS * RATE), 2))
    )
    good = {
        "reference_path": str(reference_file),
        "candidate_path": str(reference_file),
        "native_path": str(reference_file),
    }
    bad = {**good, "candidate_path": str(silence), "native_path": str(silence)}
    delivery = {
        "delivery_score": 1.0,
        "stems": [
            {"valid": True, "reference_audio": {"audible": True}, "passages": [good, bad]},
            {
                "valid": True,
                "reference_audio": {"audible": True},
                "passages": [copy.deepcopy(good)],
            },
        ],
    }
    result = asyncio.run(audio_score.score_delivery(delivery, tmp_path / "receipts"))
    assert result["timbre_score"] == pytest.approx(0.75)
    assert result["weighted_score"] == pytest.approx(0.8)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "empty", "silence", "dc", "nonfinite"])
def test_unusable_reference_raises_instead_of_scoring_zero(
    reference_file, phrase, tmp_path, failure
):
    broken = tmp_path / "broken-reference.wav"
    if failure == "corrupt":
        broken.write_bytes(b"not a WAV recording")
    elif failure != "missing":
        samples = {
            "empty": np.empty((0, 2)),
            "silence": np.zeros_like(phrase),
            "dc": np.full_like(phrase, 1e-9),
            "nonfinite": np.full_like(phrase, np.nan),
        }[failure]
        write_audio(broken, samples)
    with pytest.raises(JudgeInfrastructureError):
        audio_score.score_window(broken, reference_file, reference_file, tmp_path / "score.json")


def test_missing_native_file_is_an_evaluator_error(reference_file, tmp_path):
    with pytest.raises(JudgeInfrastructureError):
        audio_score.score_window(
            reference_file, reference_file, tmp_path / "missing-native.wav", tmp_path / "score.json"
        )


def test_decoder_runtime_failure_is_not_a_zero(reference_file, tmp_path, monkeypatch):
    def unavailable_decoder(path):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(audio_metrics, "read_audio", unavailable_decoder)
    with pytest.raises(JudgeInfrastructureError, match="decoder unavailable"):
        audio_score.score_window(
            reference_file, reference_file, reference_file, tmp_path / "score.json"
        )


@pytest.mark.parametrize("failure", ["missing_native_path", "no_windows", "no_active_references"])
def test_incomplete_evaluator_manifest_does_not_return_zero(reference_file, tmp_path, failure):
    passage = {"reference_path": str(reference_file), "candidate_path": str(reference_file)}
    stem = {"valid": True, "reference_audio": {"audible": True}, "passages": [passage]}
    if failure == "no_windows":
        stem["passages"] = []
    elif failure == "no_active_references":
        stem["reference_audio"]["audible"] = False
    with pytest.raises(JudgeInfrastructureError):
        asyncio.run(audio_score.score_delivery({"delivery_score": 1.0, "stems": [stem]}, tmp_path))


def test_scoring_is_repeatable_and_needs_no_api_credentials(reference_file, tmp_path, monkeypatch):
    for name in (
        "JUDGE_API_KEY",
        "OPENAI_API_KEY",
        "AUDIO_JUDGE_API_KEY",
        "JUDGE_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    candidate = write_audio(
        tmp_path / "candidate.wav", musical_notes(44100), 44100, subtype="PCM_24"
    )
    receipts = [tmp_path / f"repeat-{index}.json" for index in range(3)]
    results = [
        audio_score.score_window(reference_file, candidate, candidate, receipt)
        for receipt in receipts
    ]
    assert results[0]["score"] >= 0.95
    assert results[0] == results[1] == results[2]
    assert all(json.loads(receipt.read_text()) == results[0] for receipt in receipts)


def test_feature_comparison_is_repeatable_for_nonidentical_music(phrase):
    changed = musical_notes(sustained=True)
    results = [audio_metrics.compare_audio(phrase, RATE, changed, RATE) for _ in range(3)]
    assert results[0] == results[1] == results[2]
