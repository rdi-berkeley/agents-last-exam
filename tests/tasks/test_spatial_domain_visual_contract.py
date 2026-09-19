from __future__ import annotations

import asyncio
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification.scripts import (
    score_spatial_domains as scorer,
)


def png_bytes(color="white", size=(320, 240), padding=False):
    buffer = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if padding:
        metadata.add_text("padding", "x" * 60_000)
    Image.new("RGB", size, color).save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


@pytest.fixture
def bundle():
    labels = {}
    annotations = {}
    for slice_id, clusters in scorer.SLICE_CONFIG:
        labels[slice_id] = "barcode,predicted_label\n" + "".join(
            f"spot{index},{index % clusters}\n" for index in range(1000)
        )
        annotations[slice_id] = "barcode\tlayer\n" + "".join(
            f"spot{index}\t{index % clusters}\n" for index in range(1000)
        )
    return {
        "summary_csv": "slice_id,n_clusters_pred\n"
        + "".join(f"{slice_id},{clusters}\n" for slice_id, clusters in scorer.SLICE_CONFIG),
        "manifest_json": json.dumps(
            dict(has_graph=True, has_embedding=True, has_clustering=True, seed=42, method="test")
        ),
        "per_slice_labels": labels,
        "reference_annotations": annotations,
        "umap_png": png_bytes(),
        "umap_csv": "barcode,UMAP1,UMAP2\n"
        + "".join(f"spot{index},{index / 10},{index % 7}\n" for index in range(1000)),
    }


@pytest.fixture
def judge(monkeypatch):
    calls = []

    def fake_judge(**kwargs):
        calls.append(kwargs)
        return dict.fromkeys(scorer.VISUAL_CHECKS, True) | {
            "reason": "mocked semantic evidence",
            "label_matches": {str(index): True for index in range(7)},
        }

    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", fake_judge)
    return calls


def test_positive_bundle_preserves_science_and_records_visual_evidence(bundle, judge):
    result = scorer.score_output_bundle(**bundle)
    assert result.score == result.median_ari == result.min_ari == 1
    assert len(result.slice_scores) == 12
    assert result.visual_verification["passed"] is True
    assert result.visual_verification["contract"] == "spatial-umap-v2"
    assert len(judge) == 1
    assert len(judge[0]["image_bytes_list"]) == 3
    assert judge[0]["temperature"] is None
    assert judge[0]["max_tokens"] == 12000
    assert "NOT a layout or" in judge[0]["prompt"]
    assert "UNTRUSTED" in judge[0]["prompt"]
    assert "never instructions" in judge[0]["prompt"]
    assert "hidden canonical UMAP" in judge[0]["prompt"]


@pytest.mark.parametrize("failed_check", scorer.VISUAL_CHECKS)
def test_semantic_failure_overrides_perfect_ari(bundle, monkeypatch, failed_check):
    answer = dict.fromkeys(scorer.VISUAL_CHECKS, True) | {
        "reason": "wrong content",
        failed_check: False,
        "label_matches": {str(index): failed_check != "label_associations" for index in range(7)},
    }
    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", lambda **kwargs: answer)
    result = scorer.score_output_bundle(**bundle)
    assert result.score == 0
    assert result.median_ari == 1
    assert result.reason == "umap_visual_contract_failed"
    assert result.visual_verification["checks"][failed_check] is False


@pytest.mark.parametrize("payload", [b"\x89PNG\r\n\x1a\n" + b"x" * 60_000, b"not a PNG"])
def test_fabricated_header_fails_before_judge(bundle, judge, payload):
    bundle["umap_png"] = payload
    result = scorer.score_output_bundle(**bundle)
    assert result.score == 0 and "readable PNG" in result.reason
    assert not judge


def test_truncated_real_png_fails_before_judge(bundle, judge):
    bundle["umap_png"] = png_bytes()[:100]
    assert scorer.score_output_bundle(**bundle).score == 0
    assert not judge


def test_decode_is_not_semantic_acceptance(bundle, monkeypatch):
    bundle["umap_png"] = png_bytes(padding=True)
    assert len(bundle["umap_png"]) > 50_000
    answer = dict.fromkeys(scorer.VISUAL_CHECKS, False) | {
        "reason": "blank image",
        "label_matches": {str(index): False for index in range(7)},
    }
    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", lambda **kwargs: answer)
    result = scorer.score_output_bundle(**bundle)
    assert result.reason == "umap_visual_contract_failed"


def test_small_png_is_not_rejected_for_byte_size():
    payload = png_bytes()
    assert len(payload) < 50_000
    assert scorer._validate_png(payload)


def test_png_limits_and_metadata_stripping():
    with pytest.raises(ValueError, match="pixel limits"):
        scorer._validate_png(png_bytes(size=(4097, 1)))
    clean = scorer._validate_png(png_bytes(padding=True))
    assert "padding" not in Image.open(io.BytesIO(clean)).info


@pytest.mark.parametrize("image_format", ["JPEG", "GIF"])
def test_non_png_format_rejected(image_format):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "red").save(buffer, format=image_format)
    with pytest.raises(ValueError, match="single-frame PNG"):
        scorer._validate_png(buffer.getvalue())


def test_animated_png_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "red").save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (200, 200), "blue")],
        duration=100,
        loop=0,
    )
    with pytest.raises(ValueError, match="single-frame PNG"):
        scorer._validate_png(buffer.getvalue())


def test_barcode_join_not_row_order(bundle):
    coordinates, labels = scorer._parse_umap(
        bundle["umap_csv"], bundle["per_slice_labels"]["151673"]
    )
    lines = bundle["umap_csv"].splitlines()
    reversed_csv = lines[0] + "\n" + "\n".join(reversed(lines[1:]))
    reversed_coordinates, reversed_labels = scorer._parse_umap(
        reversed_csv, bundle["per_slice_labels"]["151673"]
    )
    assert reversed_coordinates == list(reversed(coordinates))
    assert reversed_labels == list(reversed(labels))


@pytest.mark.parametrize(
    "mutation",
    ["header", "duplicate", "missing", "foreign", "nan", "infinity", "flat", "extra", "short"],
)
def test_invalid_embedding_rejected_without_judge(bundle, judge, mutation):
    lines = bundle["umap_csv"].splitlines()
    if mutation == "header":
        lines[0] = "barcode,x,y"
    elif mutation == "duplicate":
        lines.append(lines[1])
    elif mutation == "missing":
        lines.pop()
    elif mutation == "foreign":
        lines[1] = "foreign,0,1"
    elif mutation == "nan":
        lines[1] = "spot0,nan,1"
    elif mutation == "infinity":
        lines[1] = "spot0,1,inf"
    elif mutation == "flat":
        lines[1:] = [f"spot{index},0,{index}" for index in range(1000)]
    elif mutation == "extra":
        lines[1] += ",extra"
    else:
        lines[1] = "spot0,1"
    bundle["umap_csv"] = "\n".join(lines)
    assert scorer.score_output_bundle(**bundle).score == 0
    assert not judge


@pytest.mark.parametrize(
    "failure", [RuntimeError("API down"), ValueError("invalid JSON"), TimeoutError("timeout")]
)
def test_api_failures_raise_infrastructure(bundle, monkeypatch, failure):
    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", fail)
    with pytest.raises(scorer.JudgeInfrastructureError):
        scorer.score_output_bundle(**bundle)


def test_trusted_render_failure_is_infrastructure(bundle, judge, monkeypatch):
    def fail(*args):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(scorer, "_render_umap_reference", fail)
    with pytest.raises(scorer.JudgeInfrastructureError, match="renderer unavailable"):
        scorer.score_output_bundle(**bundle)
    assert not judge


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"score": 1},
        dict.fromkeys(scorer.VISUAL_CHECKS, "true"),
        dict.fromkeys(scorer.VISUAL_CHECKS, 1),
        dict.fromkeys(scorer.VISUAL_CHECKS, True),
        dict.fromkeys(scorer.VISUAL_CHECKS, True) | {"reason": ""},
    ],
)
def test_judge_schema_failure_is_infrastructure(bundle, monkeypatch, answer):
    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", lambda **kwargs: answer)
    with pytest.raises(scorer.JudgeInfrastructureError):
        scorer.score_output_bundle(**bundle)


@pytest.mark.parametrize("mutation", ["missing", "extra", "string", "contradiction"])
def test_judge_label_schema_and_consistency(bundle, monkeypatch, mutation):
    answer = dict.fromkeys(scorer.VISUAL_CHECKS, True) | {
        "reason": "comparison evidence",
        "label_matches": {str(index): True for index in range(7)},
    }
    if mutation == "missing":
        del answer["label_matches"]["0"]
    elif mutation == "extra":
        answer["label_matches"]["other"] = True
    elif mutation == "string":
        answer["label_matches"]["0"] = "true"
    else:
        answer["label_matches"]["0"] = False
    monkeypatch.setattr(scorer, "llm_vision_json_judge_sync", lambda **kwargs: answer)
    with pytest.raises(scorer.JudgeInfrastructureError):
        scorer.score_output_bundle(**bundle)


@pytest.mark.parametrize(
    "median,minimum,expected",
    [
        (0.3335, 0.10, 1),
        (0.33340001, 0.10, 1),
        (0.3334, 0.10, 0.9994011976047901),
        (0.25, 0.10, 0.5),
        (0.2499, 0.10, 0),
        (0.5, 0.0999, 0.5),
        (0.29175, 0.10, 0.75),
    ],
)
def test_scientific_thresholds_unchanged(median, minimum, expected):
    assert scorer.MIN_COVERAGE == 0.999
    assert scorer._score_from_median(median, minimum) == pytest.approx(expected)


@pytest.mark.parametrize("missing,passes", [(1, True), (2, False)])
def test_coverage_boundary_unchanged(bundle, judge, missing, passes):
    lines = bundle["per_slice_labels"]["151507"].splitlines()
    bundle["per_slice_labels"]["151507"] = "\n".join(lines[:-missing])
    assert (scorer.score_output_bundle(**bundle).score == 1) is passes


def test_exact_cluster_count_still_required(bundle, judge):
    bundle["per_slice_labels"]["151673"] = bundle["per_slice_labels"]["151673"].replace(
        ",6\n", ",5\n"
    )
    result = scorer.score_output_bundle(**bundle)
    assert result.score == 0 and "expected 7" in result.reason
    assert not judge


def test_scoped_import_and_runner():
    script = """
import importlib, sys, types
from pathlib import Path
from ale_run.tasks.loader import TaskLoader
name = 'tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification'
unrelated = types.ModuleType('score_spatial_domains')
sys.modules['score_spatial_domains'] = unrelated
before = list(sys.path)
main = importlib.import_module(name + '.main')
scorer = importlib.import_module(name + '.scripts.score_spatial_domains')
assert main.score_output_bundle is scorer.score_output_bundle
assert sys.modules['score_spatial_domains'] is unrelated
assert sys.path == before
loader = TaskLoader(str(Path(*name.split('.'))))
loaded = loader.load()
assert loader._load_module().score_output_bundle is scorer.score_output_bundle
assert 'barcode,UMAP1,UMAP2' in loaded['description']
assert loaded['metadata']['umap_csv'].endswith(scorer.REQUIRED_UMAP_CSV)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("override", [None, "task-specific-judge"])
def test_task_judge_model_default_and_override(bundle, judge, monkeypatch, override):
    monkeypatch.delenv("SPATIAL_DOMAIN_VISION_MODEL", raising=False)
    if override:
        monkeypatch.setenv("SPATIAL_DOMAIN_VISION_MODEL", override)
    result = scorer.score_output_bundle(**bundle)
    expected = override or "gpt-5.5-2026-04-23"
    assert judge[0]["model"] == expected
    assert result.visual_verification["model"] == expected


@pytest.fixture
def remote_bundle(bundle):
    from types import SimpleNamespace

    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    meta = main.config.to_metadata()
    files = {
        meta["summary_file"]: bundle["summary_csv"].encode(),
        meta["manifest_file"]: bundle["manifest_json"].encode(),
        meta["umap_png"]: bundle["umap_png"],
        meta["umap_csv"]: bundle["umap_csv"].encode(),
    }
    for slice_id, path in meta["label_files"].items():
        files[path] = bundle["per_slice_labels"][slice_id].encode()
    for slice_id, path in meta["annotation_files"].items():
        files[path] = bundle["reference_annotations"][slice_id].encode()

    class Session:
        async def file_exists(self, path):
            return path in files

        async def directory_exists(self, path):
            return False

        async def read_bytes(self, path):
            return files[path]

    return SimpleNamespace(metadata=meta), Session(), files


def test_main_reads_embedding_and_propagates_judge_failure(bundle, monkeypatch, remote_bundle):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, files = remote_bundle

    def fail(**kwargs):
        assert kwargs["umap_csv"] == bundle["umap_csv"]
        raise scorer.JudgeInfrastructureError("unavailable")

    monkeypatch.setattr(main, "score_output_bundle", fail)
    with pytest.raises(scorer.JudgeInfrastructureError):
        asyncio.run(main.evaluate(task, session))
    files.pop(task.metadata["umap_csv"])
    assert asyncio.run(main.evaluate(task, session)) == [0.0]


@pytest.fixture
def output_dirs(tmp_path, bundle):
    output = tmp_path / "output"
    reference = tmp_path / "reference"
    (output / "per_slice").mkdir(parents=True)
    (reference / "manual_annotations").mkdir(parents=True)
    for name, key in [
        ("summary.csv", "summary_csv"),
        ("manifest.json", "manifest_json"),
        (scorer.REQUIRED_UMAP_CSV, "umap_csv"),
    ]:
        (output / name).write_text(bundle[key])
    (output / scorer.REQUIRED_PNG).write_bytes(bundle["umap_png"])
    for slice_id, _ in scorer.SLICE_CONFIG:
        (output / "per_slice" / f"{slice_id}_labels.csv").write_text(
            bundle["per_slice_labels"][slice_id]
        )
        (reference / "manual_annotations" / f"{slice_id}_manual_annotations.tsv").write_text(
            bundle["reference_annotations"][slice_id]
        )
    return output, reference


def test_output_dir_reads_embedding(output_dirs, judge):
    output, reference = output_dirs
    assert scorer.score_output_dir(output, reference).score == 1


@pytest.mark.parametrize("missing_candidate", [False, True])
@pytest.mark.parametrize(
    "reference_payload",
    [
        None,
        b"wrong\theader\nspot0\t0\n",
        b"barcode\tlayer\nspot0\n",
        b"barcode\tlayer\nspot0\t0\textra\n",
        b"barcode\tlayer\nspot0\t\n",
        b"barcode\tlayer\nspot0\t0\nspot0\t1\n",
        b"barcode\tlayer\n",
        b"barcode\tlayer\nspot0\tNA\n",
        b'barcode\tlayer\n"spot0\t0\n',
        b"\xff",
    ],
)
def test_main_reference_failures_precede_candidate_validation(
    remote_bundle, judge, reference_payload, missing_candidate
):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, files = remote_bundle
    path = task.metadata["annotation_files"]["151672"]
    if reference_payload is None:
        del files[path]
    else:
        files[path] = reference_payload
    if missing_candidate:
        del files[task.metadata["umap_csv"]]
    else:
        files[task.metadata["manifest_file"]] = b"\xff"
    with pytest.raises(scorer.ReferenceValidationError):
        asyncio.run(main.evaluate(task, session))
    assert not judge


@pytest.mark.parametrize(
    "candidate",
    ["summary_file", "manifest_file", "umap_csv"]
    + [slice_id for slice_id, _ in scorer.SLICE_CONFIG],
)
def test_main_candidate_invalid_utf8_is_zero(remote_bundle, judge, candidate):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, files = remote_bundle
    path = (
        task.metadata["label_files"][candidate]
        if candidate in task.metadata["label_files"]
        else task.metadata[candidate]
    )
    files[path] = b"\xff"
    assert asyncio.run(main.evaluate(task, session)) == [0.0]
    assert not judge


@pytest.mark.parametrize("payload", [b"not,csv\n", b"slice_id,n_clusters_pred\n151507\n"])
def test_main_candidate_parse_failure_is_zero(remote_bundle, judge, payload):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, files = remote_bundle
    files[task.metadata["summary_file"]] = payload
    assert asyncio.run(main.evaluate(task, session)) == [0.0]
    assert not judge


@pytest.mark.parametrize("failure_type", [RuntimeError, ValueError, KeyError, OSError])
def test_main_metric_programming_and_infrastructure_errors_propagate(
    remote_bundle, judge, monkeypatch, failure_type
):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, _ = remote_bundle

    def fail(*args):
        raise failure_type("metric failure")

    monkeypatch.setattr(scorer, "adjusted_rand_score", fail)
    with pytest.raises(failure_type, match="metric failure"):
        asyncio.run(main.evaluate(task, session))
    assert not judge


@pytest.mark.parametrize("failure_type", [OSError, TimeoutError, PermissionError])
@pytest.mark.parametrize("reference_read", [False, True])
def test_main_read_infrastructure_errors_propagate(
    remote_bundle, monkeypatch, failure_type, reference_read
):
    from tasks.life_sciences.spatial_transcriptomics_spatial_domain_identification import main

    task, session, _ = remote_bundle
    original_read = session.read_bytes
    failing_path = (
        task.metadata["annotation_files"]["151673"]
        if reference_read
        else task.metadata["summary_file"]
    )

    async def fail(path):
        if path == failing_path:
            raise failure_type("transport failure")
        return await original_read(path)

    monkeypatch.setattr(session, "read_bytes", fail)
    with pytest.raises(failure_type, match="transport failure"):
        asyncio.run(main.evaluate(task, session))


@pytest.mark.parametrize("missing", [False, True])
def test_bundle_reference_failures_precede_candidate_validation(bundle, judge, missing):
    if missing:
        del bundle["reference_annotations"]["151672"]
    else:
        bundle["reference_annotations"]["151672"] = "barcode\tlayer\nspot0\n"
    bundle["manifest_json"] = "bad JSON"
    with pytest.raises(scorer.ReferenceValidationError):
        scorer.score_output_bundle(**bundle)
    assert not judge


@pytest.mark.parametrize("mutation", ["missing_labels", "short", "extra", "empty", "quote", "json"])
def test_bundle_candidate_format_failures_are_zero(bundle, judge, mutation):
    if mutation == "missing_labels":
        del bundle["per_slice_labels"]["151672"]
    elif mutation == "json":
        bundle["manifest_json"] = "not JSON"
    else:
        rows = {
            "short": "spot0\n",
            "extra": "spot0,0,extra\n",
            "empty": "spot0,\n",
            "quote": '"spot0,0\n',
        }
        bundle["per_slice_labels"]["151672"] = "barcode,predicted_label\n" + rows[mutation]
    assert scorer.score_output_bundle(**bundle).score == 0
    assert not judge


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unicode"])
def test_output_dir_reference_failure_precedes_missing_candidate(output_dirs, judge, mutation):
    output, reference = output_dirs
    (output / scorer.REQUIRED_UMAP_CSV).unlink()
    path = reference / "manual_annotations/151672_manual_annotations.tsv"
    if mutation == "missing":
        path.unlink()
        failure = FileNotFoundError
    elif mutation == "unicode":
        path.write_bytes(b"\xff")
        failure = UnicodeError
    else:
        path.write_text("barcode\tlayer\nspot0\n")
        failure = scorer.ReferenceValidationError
    with pytest.raises(failure):
        scorer.score_output_dir(output, reference)
    assert not judge


@pytest.mark.parametrize("mutation", ["missing", "directory", "unicode"])
def test_output_dir_candidate_file_failure_is_zero(output_dirs, judge, mutation):
    output, reference = output_dirs
    path = output / scorer.REQUIRED_UMAP_CSV
    if mutation == "unicode":
        path.write_bytes(b"\xff")
    else:
        path.unlink()
        if mutation == "directory":
            path.mkdir()
    assert scorer.score_output_dir(output, reference).score == 0
    assert not judge


def test_trusted_render_has_all_label_panels(bundle):
    coordinates, labels = scorer._parse_umap(
        bundle["umap_csv"], bundle["per_slice_labels"]["151673"]
    )
    image = scorer._render_umap_reference(coordinates, labels)
    with Image.open(io.BytesIO(image)) as picture:
        assert picture.size == (1440, 720)
        assert len(picture.convert("RGB").getcolors(maxcolors=1_000_000)) > 100
    assert len(set(labels)) == 7
    assert len(list(csv.DictReader(io.StringIO(bundle["umap_csv"])))) == len(coordinates)


def test_trusted_render_uses_actual_coordinates_and_label_associations(bundle, monkeypatch):
    import numpy as np
    from matplotlib.figure import Figure

    coordinates, labels = scorer._parse_umap(
        bundle["umap_csv"], bundle["per_slice_labels"]["151673"]
    )
    original_save = Figure.savefig
    inspected = []

    def inspect_and_save(figure, *args, **kwargs):
        assert len(figure.axes) == 8
        classes = sorted(set(labels))
        for index, label in enumerate(classes):
            expected = [
                point for point, predicted in zip(coordinates, labels) if predicted == label
            ]
            overlay = figure.axes[0].collections[index]
            assert overlay.get_label() == label
            np.testing.assert_array_equal(overlay.get_offsets(), expected)
            background, highlighted = figure.axes[index + 1].collections
            np.testing.assert_array_equal(background.get_offsets(), coordinates)
            np.testing.assert_array_equal(highlighted.get_offsets(), expected)
        inspected.append(True)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    scorer._render_umap_reference(coordinates, labels)
    assert inspected == [True]


def test_orientation_evidence_preserves_points_and_labels(bundle, monkeypatch):
    import numpy as np
    from matplotlib.figure import Figure

    coordinates, labels = scorer._parse_umap(
        bundle["umap_csv"], bundle["per_slice_labels"]["151673"]
    )
    original_save = Figure.savefig
    inspected = []

    def inspect_and_save(figure, *args, **kwargs):
        assert len(figure.axes) == 16
        for axis in figure.axes:
            assert len(axis.collections) == 7
            for collection, label in zip(axis.collections, sorted(set(labels))):
                assert collection.get_label() == label
                original = np.asarray(
                    [point for point, predicted in zip(coordinates, labels) if predicted == label]
                )
                transformed = collection.get_offsets()
                np.testing.assert_allclose(
                    np.linalg.norm(transformed - transformed[0], axis=1),
                    np.linalg.norm(original - original[0], axis=1),
                    atol=1e-10,
                )
        inspected.append(True)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    scorer._render_umap_orientations(coordinates, labels)
    assert inspected == [True]
