import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tasks/visual_media/project_migration/scripts/score_audio_remote.py"
)
SPEC = importlib.util.spec_from_file_location("migration_audio", SOURCE)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def test_antiphase_stereo_is_not_silence(tmp_path):
    tone = 0.25 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([tone, -tone]), 48000, subtype="FLOAT")
    assert scorer.read_wav_rms_db(path) == pytest.approx(-15.0515, abs=1e-4)
    assert scorer.check_audio_quality(path)


def test_clipping_in_one_channel_cannot_cancel(tmp_path):
    path = tmp_path / "clipped.wav"
    sf.write(path, [[1.2, -1.2]] * 100, 48000, subtype="FLOAT")
    assert not scorer.check_audio_quality(path)


def test_empty_audio_is_invalid(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty((0, 2)), 48000)
    assert scorer.read_wav_rms_db(path) == -120
    assert not scorer.check_audio_quality(path)


def test_nonfinite_audio_is_invalid(tmp_path):
    path = tmp_path / "invalid.wav"
    sf.write(path, [[np.nan, 0]], 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Non-finite"):
        scorer.read_wav_rms_db(path)
    assert not scorer.check_audio_quality(path)


def test_ambiguous_fuzzy_match_is_not_arbitrary():
    assert scorer.find_stem_file(["Violin 1.wav", "Violin 2.wav"], "Violin") is None


def test_one_stem_cannot_satisfy_two_references():
    assert scorer.match_stem_files(
        ["Bass Trombone.wav"], ["Bass Trombone.wav", "Trombone.wav"]
    ) == {
        "Bass Trombone.wav": "Bass Trombone.wav",
        "Trombone.wav": None,
    }


def test_native_cubase_names_match_exactly():
    name = "Suite - 0001 - 乐器 - Harp.wav"
    assert scorer.find_stem_file([name, "Piano.wav"], "Harp") == name


def test_short_nonempty_timbre_control_is_finite(tmp_path):
    path = tmp_path / "short.wav"
    tone = 0.25 * np.sin(2 * np.pi * 440 * np.arange(2205) / 22050)
    sf.write(path, tone, 22050)
    assert scorer.compute_timbre_similarity(path, path) == 1.0
