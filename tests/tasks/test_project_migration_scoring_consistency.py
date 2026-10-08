import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_judge
from tasks.visual_media.project_migration.scripts.equivalence import compare_pcm


@pytest.fixture
def signals(tmp_path):
    times = np.arange(960) / 48000
    samples = np.column_stack(
        (0.1 * np.sin(2 * np.pi * 440 * times), 0.07 * np.cos(2 * np.pi * 330 * times))
    )
    integer = tmp_path / "delivery.wav"
    native = tmp_path / "native.wav"
    reference = tmp_path / "reference.wav"
    sf.write(integer, np.rint(samples * 32768) / 32768, 48000, subtype="PCM_16")
    sf.write(native, samples, 48000, subtype="FLOAT")
    sf.write(reference, 0.1 * np.sin(2 * np.pi * 660 * times), 48000, subtype="PCM_16")
    return reference, integer, native


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["zero", "tiny_dc", "large_dc"])
async def test_no_signal_cannot_be_promoted_by_normalization(signals, tmp_path, signal):
    reference, candidate, _ = signals
    values = {
        "zero": [0.0, 0.0],
        "tiny_dc": [-5.064150210287721e-10, -3.6223249400002544e-10],
        "large_dc": [0.2, -0.3],
    }
    samples = np.tile(values[signal], (960, 1))
    sf.write(candidate, samples, 48000, subtype="FLOAT")

    def forbidden(request):
        pytest.fail("Normalization must not amplify a silent candidate for judging")

    result = audio_judge.deterministic_score(reference, candidate)
    assert result["score"] == 0 and result["route"] == "silent_candidate"
    with pytest.raises(JudgeInfrastructureError, match="zero samples or constant DC"):
        audio_judge.normalized_payload(candidate)
    same = audio_judge.deterministic_score(candidate, candidate)
    assert same["score"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_24"])
async def test_faint_varying_signal_preserves_gain_invariance(tmp_path, subtype):
    times = np.arange(960) / 48000
    envelope = np.linspace(0.2, 1, len(times))
    waveform = envelope[:, None] * np.column_stack(
        (0.25 * np.sin(2 * np.pi * 440 * times), 0.125 * np.cos(2 * np.pi * 330 * times))
    )
    samples = waveform * 2**-20 if subtype == "FLOAT" else np.rint(waveform * 256) * 2**-23
    faint, louder = tmp_path / "faint.wav", tmp_path / "louder.wav"
    sf.write(faint, samples, 48000, subtype=subtype)
    decoded, rate = sf.read(faint, always_2d=True)
    assert 0 < np.abs(decoded).max() < 2**-16
    assert audio_judge.silence_reason(decoded) is None
    sf.write(louder, decoded * 2**16, rate, subtype=subtype)
    faint_payload, faint_gain = audio_judge.normalized_payload(faint)
    louder_payload, louder_gain = audio_judge.normalized_payload(louder)
    assert faint_payload == louder_payload
    assert faint_gain == louder_gain * 2**16

    def forbidden(request):
        pytest.fail("Gain-invariance controls must not call the provider")

    result = audio_judge.deterministic_score(faint, faint)
    assert result["score"] == 1 and result["route"] == "exact_decoded_pcm"
    if subtype == "PCM_24":
        scaled = audio_judge.deterministic_score(louder, faint)
        assert scaled["score"] == 1 and scaled["route"] == "analytic_quantization"
    zero_delivery = tmp_path / "zero-pcm16.wav"
    sf.write(zero_delivery, np.rint(decoded * 32768) / 32768, rate, subtype="PCM_16")
    zero = audio_judge.deterministic_score(louder, zero_delivery)
    assert zero["score"] == 0 and zero["reason"] == "zero_samples"


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_native_float_integer_rounding_requires_exact_cells(signals, tmp_path, bits):
    _, integer, native = signals
    samples, rate = sf.read(native, always_2d=True)
    scale = 2 ** (bits - 1)
    quantized = np.rint(samples * scale) / scale
    sf.write(integer, quantized, rate, subtype=f"PCM_{bits}")
    for before, after in ((integer, native), (native, integer)):
        proof = compare_pcm(before, after, allow_float_quantization=True)
        assert proof["equivalent"] and proof["shared_gain"] == 1
        assert proof["quantization_bound_full_scale"] == 2**-bits
    assert not compare_pcm(integer, native)["equivalent"]
    outside = tmp_path / "outside.wav"
    changed = samples.copy()
    changed[120, 0] = quantized[120, 0] + 1 / scale
    sf.write(outside, changed, rate, subtype="DOUBLE")
    assert not compare_pcm(integer, outside, allow_float_quantization=True)["equivalent"]


@pytest.mark.parametrize(
    "change", ["note", "time_gain", "channel_gain", "collapse", "shift", "rate"]
)
def test_native_equivalence_does_not_accept_changed_signal(signals, tmp_path, change):
    _, integer, native = signals
    samples, rate = sf.read(native, always_2d=True)
    if change == "note":
        samples[:, 0] = 0.1 * np.sin(2 * np.pi * 880 * np.arange(len(samples)) / rate)
    elif change == "time_gain":
        samples[len(samples) // 2 :] *= 0.5
    elif change == "channel_gain":
        samples[:, 0] *= 0.5
    elif change == "collapse":
        samples = np.repeat(samples.mean(axis=1, keepdims=True), 2, axis=1)
    elif change == "shift":
        samples = np.roll(samples, 1, axis=0)
    elif change == "rate":
        rate = 44100
    changed = tmp_path / "changed.wav"
    sf.write(changed, samples, rate, subtype="FLOAT")
    assert not compare_pcm(integer, changed, allow_float_quantization=True)["equivalent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("equivalent", [True, False])
async def test_joint_judgment_retains_strict_equivalence_guard(signals, tmp_path, equivalent):
    reference, integer, native = signals
    if not equivalent:
        samples, rate = sf.read(native, always_2d=True)
        sf.write(native, samples[::-1], rate, subtype="FLOAT")
    calls = []

    def handle(request):
        calls.append(request)
        component = {
            "assessable": True,
            "candidate_character": "different",
            "agreements": [],
            "differences": ["different"],
            "limitations": [],
            "rating": 1,
        }
        judgment = {
            "reference_character": "sustained",
            "candidate_a": component,
            "candidate_b": {**component, "rating": 0},
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(judgment)}}]}
        )

    result = await audio_judge.judge_window(
        reference, integer, native, tmp_path / "joint.json", transport=httpx.MockTransport(handle)
    )
    assert result["score"] == (0.25 if equivalent else 0)
    assert len(calls) == 1
    assert bool(result["native"].get("reused_analytic_equivalence")) == equivalent


@pytest.fixture
def frozen_score():
    path = os.environ.get("MIGRATION_FROZEN_SCORE")
    if not path:
        pytest.skip("Private retained native witnesses require MIGRATION_FROZEN_SCORE")
    return json.loads(Path(path).read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("window_index", [0, 1])
async def test_actual_mallets_dc_overrides_cached_false_positive(
    frozen_score, tmp_path, window_index
):
    stem = next(
        stem
        for stem in frozen_score["delivery"]["stems"]
        if stem["reference"] == "Percussion Mallets.flac"
    )
    passage = stem["passages"][window_index]
    old = stem["judgments"][window_index]["native"]
    assert old["score"] == 0.75
    receipt = tmp_path / "judge.json"
    receipt.write_text(json.dumps(old))

    def forbidden(request):
        pytest.fail("The retained exact DC must never invoke the provider")

    result = audio_judge.deterministic_score(
        Path(passage["reference_path"]), Path(passage["native_path"])
    )
    assert json.loads(receipt.read_text()) == old
    assert result["score"] == 0
    assert result["reason"] == "constant_per_channel_dc"


@pytest.mark.asyncio
async def test_actual_trombones_half_lsb_reuses_first_judgment(frozen_score, tmp_path):
    stem = next(
        stem for stem in frozen_score["delivery"]["stems"] if stem["reference"] == "Trombones.flac"
    )
    window_index = next(
        index for index, passage in enumerate(stem["passages"]) if passage["start"] == 603.45
    )
    old = stem["judgments"][window_index]
    assert old["delivery"]["score"] == 0.25 and old["native"]["score"] == 0.75
    passage = stem["passages"][window_index]
    proof = compare_pcm(
        Path(passage["candidate_path"]),
        Path(passage["native_path"]),
        allow_float_quantization=True,
    )
    assert proof["equivalent"]
    assert proof["quantization_bound_full_scale"] == 2**-16
