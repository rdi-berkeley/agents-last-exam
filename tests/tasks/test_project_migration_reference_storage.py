import numpy as np
import pytest
import soundfile as sf

from tasks.visual_media.project_migration.scripts.audio_delivery import check_delivery


def test_lossless_reference_storage_preserves_windows_and_samples(tmp_path):
    reference_wav = tmp_path / "wav"
    reference_flac = tmp_path / "flac"
    output = tmp_path / "output"
    for directory in (reference_wav, reference_flac, output / "stems"):
        directory.mkdir(parents=True)
    positions = np.arange(44100 * 13)
    samples = (np.sin(positions * 0.017) * (2**22)).astype(np.int32) * 256
    samples = np.column_stack((samples, samples // 2 // 256 * 256))
    sf.write(reference_wav / "Harp.wav", samples, 44100, subtype="PCM_24")
    sf.write(reference_flac / "Harp.flac", samples, 44100, subtype="PCM_24")
    sf.write(output / "stems/Harp.wav", samples, 44100, subtype="PCM_24")
    sf.write(output / "mixdown.wav", samples, 44100, subtype="PCM_24")
    original = check_delivery(output, reference_wav, tmp_path / "evidence-wav")
    packed = check_delivery(output, reference_flac, tmp_path / "evidence-flac")
    assert original["delivery_score"] == packed["delivery_score"] == 1
    assert original["expected_stems"] == packed["expected_stems"] == 1
    before, after = original["stems"][0], packed["stems"][0]
    assert before["reference_audio"] == after["reference_audio"]
    assert [row["sha256"] for row in before["passages"]] == [
        row["sha256"] for row in after["passages"]
    ]


def test_duplicate_reference_encodings_are_infrastructure_error(tmp_path):
    (tmp_path / "Harp.wav").touch()
    (tmp_path / "Harp.flac").touch()
    with pytest.raises(RuntimeError, match="Ambiguous evaluator reference"):
        check_delivery(tmp_path, tmp_path, tmp_path / "evidence")
