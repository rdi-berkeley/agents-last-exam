"""Opt-in, read-only replays of the selected post-validation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.spatial.distance import cdist

from tasks.life_sciences.merfish_image_decoding_segmentation_1.scripts import score_merfish
from tasks.physical_sciences.hst_acs_wfc_visit_reduction.scripts import score_outputs


@pytest.fixture
def release():
    value = os.environ.get("MERFISH_HST_RELEASE_ROOT")
    if not value:
        pytest.skip("set MERFISH_HST_RELEASE_ROOT for retained-artifact replays")
    return Path(value)


@pytest.fixture
def data_root():
    return Path(__file__).resolve().parents[2] / "task-data-hf/extracted"


@pytest.fixture
def frozen_scorers(release, monkeypatch):
    archive = (
        release / "validation-source-context-qemu-r02-20260917/private-validation-context.tar.gz"
    )
    modules = {}
    with tarfile.open(archive) as source_archive:
        for name, member, digest in (
            (
                "merfish",
                "tasks/life_sciences/merfish_image_decoding_segmentation_1/scripts/score_merfish.py",
                "1ac7ef904d84bc68a3ecddb56d2e0f4e8aec50bf83f11731f3f45a4b69090923",
            ),
            (
                "hst",
                "tasks/physical_sciences/hst_acs_wfc_visit_reduction/scripts/score_outputs.py",
                "24f899b9e196bb0231ab3fbd4003e40893e19c87c227ea7f6b637ab446da0f73",
            ),
        ):
            source = source_archive.extractfile(member).read()
            assert hashlib.sha256(source).hexdigest() == digest
            module = ModuleType(f"frozen_post_validation_{name}")
            monkeypatch.setitem(sys.modules, module.__name__, module)
            exec(compile(source, f"{archive}:{member}", "exec"), module.__dict__)
            modules[name] = module
    return modules


def record(release, name, result):
    root = release / "post-validation-20260918/merfish-hst-evidence"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"after-{name}.json").write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize("selection", ["kimi_after", "codex_after_repair"])
def test_merfish_retained_outputs(release, data_root, frozen_scorers, selection):
    evidence = json.loads((release / "validation-checks/merfish-r02-evidence.json").read_text())
    artifacts = evidence["runs"][selection]["artifacts"]
    payloads = {}
    for filename, identity in artifacts.items():
        payloads[filename] = Path(identity["path"]).read_bytes()
        assert hashlib.sha256(payloads[filename]).hexdigest() == identity["sha256"]
    base = data_root / "life_sciences/merfish_image_decoding_segmentation_1/base"
    reference = (base / "reference/benchmark_results.csv").read_bytes()
    codebook = (base / "input/codebook.json").read_bytes()
    assert hashlib.sha256(reference).hexdigest() == (
        "8b32453bfdf6b539b45ec223eab6bdef4e5af3d083974a34d15b44c2ab156fc6"
    )
    arguments = dict(
        decoded_csv_bytes=payloads["decoded_transcripts.csv"],
        segmentation_tiff_bytes=payloads["segmentation.tiff"],
        cell_by_gene_csv_bytes=payloads["cell_by_gene.csv"],
        quality_metrics_json_bytes=payloads["quality_metrics.json"],
        reference_csv_bytes=reference,
        codebook_json_bytes=codebook,
    )
    result = score_merfish.score(**arguments)
    baseline = frozen_scorers["merfish"].score(**arguments)
    assert baseline.score == {"kimi_after": 0.55, "codex_after_repair": 0.46}[selection]
    expected = {"kimi_after": 0.55, "codex_after_repair": 0.535}
    record(
        release,
        f"merfish-{selection}",
        {
            "artifacts": artifacts,
            "baseline": baseline.to_dict(),
            "result": result.to_dict(),
        },
    )
    assert result.score == expected[selection]
    assert result.components["total_count"].score == 0
    assert result.components["matrix_consistency"].score == 0.1


def test_actual_merfish_codebook_full_bit_error_model(release, data_root):
    path = (
        data_root / "life_sciences/merfish_image_decoding_segmentation_1/base/input/codebook.json"
    )
    payload = path.read_bytes()
    mappings = json.loads(payload)["mappings"]
    codes = np.zeros((len(mappings), 8, 2), dtype=np.uint8)
    for index, mapping in enumerate(mappings):
        for bit in mapping["codeword"]:
            codes[index, bit["r"], bit["c"]] = bit["v"]
    full_codes = codes.reshape(len(codes), 16)
    tuples, counts = np.unique(codes.argmax(axis=2), axis=0, return_counts=True)
    distances = cdist(full_codes, full_codes, metric="hamming") * 16
    np.fill_diagonal(distances, np.inf)
    assert len(np.unique(full_codes, axis=0)) == 140
    assert len(tuples) == 62 and counts.max() == 22
    assert distances.min() == 4
    single_bit_count = 0
    two_bit_count = 0
    for target, code in enumerate(full_codes):
        exact = cdist(code[None], full_codes, metric="hamming") * 16
        assert np.flatnonzero(exact[0] <= 1).tolist() == [target]
        for first_bit in range(16):
            changed = code.copy()
            changed[first_bit] ^= 1
            distances = cdist(changed[None], full_codes, metric="hamming") * 16
            assert np.flatnonzero(distances[0] <= 1).tolist() == [target]
            single_bit_count += 1
            for second_bit in range(first_bit + 1, 16):
                double_error = changed.copy()
                double_error[second_bit] ^= 1
                distances = cdist(double_error[None], full_codes, metric="hamming") * 16
                assert not np.any(distances <= 1)
                two_bit_count += 1
    record(
        release,
        "merfish-codebook",
        {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "unique_full_bit_words": 140,
            "checkall_argmax_tuples": len(tuples),
            "largest_collision": int(counts.max()),
            "minimum_hamming_distance": 4,
            "exact_positive_controls": 140,
            "single_bit_positive_controls": single_bit_count,
            "two_bit_rejected_controls": two_bit_count,
        },
    )


@pytest.mark.parametrize("selection", ["final", "old", "astra"])
def test_hst_retained_replayed_products(release, data_root, frozen_scorers, selection):
    audit = release / "validation-checks/hst-acs-r02-20260917T142539Z"
    output = audit / "replayed" / selection
    reference = data_root / "physical_sciences/hst_acs_wfc_visit_reduction/base/reference"
    protected = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (output, reference / "reference_outputs")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    result = score_outputs.evaluate_output_directory(output, reference)
    baseline = frozen_scorers["hst"].evaluate_output_directory(output, reference)
    assert baseline["score"] == {"final": 0.8592, "old": 0.9867, "astra": 1.0}[selection]
    expected = {"final": 0.9592, "old": 0.9867, "astra": 1.0}
    record(
        release,
        f"hst-{selection}",
        {"protected_sha256": protected, "baseline": baseline, "result": result},
    )
    assert result["score"] == pytest.approx(expected[selection])
    if selection == "final":
        assert any("low source completeness" in note for note in result["notes"])
        assert not any("alignment solution" in note for note in result["notes"])


def test_actual_data_inventory(release, data_root):
    inventory = {}
    image_summary = {}
    for task in (
        "life_sciences/merfish_image_decoding_segmentation_1",
        "physical_sciences/hst_acs_wfc_visit_reduction",
    ):
        base = data_root / task / "base"
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            inventory[str(path)] = hashlib.sha256(payload).hexdigest()
            frozen = release / "data-v1.1/files" / task / "base" / path.relative_to(base)
            assert hashlib.sha256(frozen.read_bytes()).hexdigest() == inventory[str(path)]
            if path.suffix == ".tiff":
                with Image.open(io.BytesIO(payload)) as image:
                    values = np.asarray(image)
                    image_summary[str(path)] = {
                        "shape": list(values.shape),
                        "dtype": str(values.dtype),
                        "min": int(values.min()),
                        "max": int(values.max()),
                    }
                    assert values.shape == (2048, 2048)
    before = json.loads(
        (
            release / "post-validation-20260918/merfish-hst-evidence/before-data-inventory.json"
        ).read_text()
    )
    assert inventory == before["sha256"]
    record(release, "data-inventory", {"sha256": inventory, "tiff_summary": image_summary})


@pytest.mark.parametrize(
    "control", ["positive", "wrong_genes", "wrong_locations", "duplicate_spots"]
)
def test_merfish_reference_based_controls(release, data_root, control):
    evidence = json.loads((release / "validation-checks/merfish-r02-evidence.json").read_text())
    mask_path = Path(evidence["runs"]["kimi_after"]["artifacts"]["segmentation.tiff"]["path"])
    mask_bytes = mask_path.read_bytes()
    mask = np.asarray(Image.open(io.BytesIO(mask_bytes)))
    base = data_root / "life_sciences/merfish_image_decoding_segmentation_1/base"
    codebook_bytes = (base / "input/codebook.json").read_bytes()
    genes = score_merfish._real_gene_list(json.loads(codebook_bytes))
    reference = (base / "reference/benchmark_results.csv").read_bytes()
    decoded = pd.read_csv(io.BytesIO(reference))[score_merfish.REQUIRED_DECODED_COLUMNS].copy()
    if control == "wrong_genes":
        ordered = decoded.gene.value_counts().reindex(genes).sort_values().index.tolist()
        decoded["gene"] = decoded.gene.replace(dict(zip(ordered, reversed(ordered))))
    elif control == "wrong_locations":
        decoded["y"] = 2007.0
    elif control == "duplicate_spots":
        decoded = pd.concat([decoded, decoded], ignore_index=True)
    labels = mask[np.rint(decoded.y).astype(int), np.rint(decoded.x).astype(int)]
    real = decoded.gene.isin(genes).to_numpy()
    matrix = pd.crosstab(labels[real], decoded.loc[real, "gene"]).reindex(
        index=np.arange(1, int(mask.max()) + 1), columns=genes, fill_value=0
    )
    matrix.index.name = "cell_id"
    metrics = {
        "total_decoded_transcripts": len(decoded),
        "blank_rate": float(decoded.gene.str.startswith("Blank-").mean()),
        "exact_match_fraction": float(decoded.is_exact.mean()),
        "n_cells": int(mask.max()),
        "assigned_fraction": float((labels > 0).mean()),
        "mean_transcripts_per_cell": float(matrix.to_numpy().sum() / mask.max()),
    }
    result = score_merfish.score(
        decoded_csv_bytes=decoded.to_csv(index=False).encode(),
        segmentation_tiff_bytes=mask_bytes,
        cell_by_gene_csv_bytes=matrix.to_csv().encode(),
        quality_metrics_json_bytes=json.dumps(metrics).encode(),
        reference_csv_bytes=reference,
        codebook_json_bytes=codebook_bytes,
    )
    record(release, f"merfish-control-{control}", result.to_dict())
    assert not result.hard_failed
    assert result.components["matrix_consistency"].score == 0.1
    if control == "positive":
        assert result.score == 1.0
    elif control == "wrong_genes":
        assert result.pearson_r < 0.5
        assert all(
            result.components[name].score == 0 for name in score_merfish.DECODING_CREDIT_COMPONENTS
        )
    else:
        assert result.components["spatial"].score == 0
        if control == "duplicate_spots":
            assert result.spatial_concordance == 0.5
            assert result.components["total_count"].score == 0


@pytest.mark.parametrize(
    "control", ["opposite_sign", "mixed_sign", "wrong_shift", "lost_detection", "wrong_image"]
)
def test_hst_actual_product_controls(release, data_root, tmp_path, control):
    visit = "acs_visit_f606w_lockman"
    source = release / "validation-checks/hst-acs-r02-20260917T142539Z/replayed/final" / visit
    reference = (
        data_root
        / "physical_sciences/hst_acs_wfc_visit_reduction/base/reference/reference_outputs/visible"
        / visit
    )
    output = tmp_path / visit
    shutil.copytree(source, output)
    baseline = score_outputs.score_visit(source, reference)[0]
    if control in {"opposite_sign", "mixed_sign", "wrong_shift"}:
        alignment = pd.read_csv(output / "alignment_solution.csv")
        if control == "opposite_sign":
            alignment[["dx_pix", "dy_pix"]] *= -1
        elif control == "mixed_sign":
            alignment["dx_pix"] *= -1
        else:
            alignment["dx_pix"] += 0.5
        alignment.to_csv(output / "alignment_solution.csv", index=False)
    elif control == "lost_detection":
        catalog = pd.read_csv(output / "source_catalog.csv")
        catalog.iloc[1:].to_csv(output / "source_catalog.csv", index=False)
    else:
        image = np.loadtxt(output / "drizzled_image.csv", delimiter=",")
        np.savetxt(output / "drizzled_image.csv", np.roll(image, 10, axis=1), delimiter=",")
    result, notes = score_outputs.score_visit(output, reference)
    record(
        release, f"hst-control-{control}", {"baseline": baseline, "score": result, "notes": notes}
    )
    if control == "opposite_sign":
        assert result == baseline
    elif control in {"mixed_sign", "wrong_shift"}:
        assert result == baseline - 10
        assert "alignment solution outside tolerance" in notes
    elif control == "lost_detection":
        assert result == pytest.approx(baseline - 16 / 25)
    else:
        assert result == pytest.approx(baseline - 12)
        assert any("RMSE too high" in note for note in notes)


def test_hst_fits_displacements_and_image_registration(release, data_root, monkeypatch):
    audit = release / "validation-checks/hst-acs-r02-20260917T142539Z"
    runtime = json.loads((audit / "runtime-versions.json").read_text())
    monkeypatch.syspath_prepend(str(Path(runtime["packages"]["astropy"]["path"]).parents[1]))
    from astropy.io import fits

    base = data_root / "physical_sciences/hst_acs_wfc_visit_reduction/base"
    measurements = {}
    for split in ("visible", "hidden"):
        for reference in sorted((base / "reference/reference_outputs" / split).iterdir()):
            visit = reference.name
            inputs = (
                base / "input" if split == "visible" else base / "reference/hidden_input"
            ) / visit
            reference_shifts = pd.read_csv(reference / "alignment_solution.csv").set_index(
                "exposure_id"
            )
            actual = pd.read_csv(
                audit / "replayed/final" / visit / "alignment_solution.csv"
            ).set_index("exposure_id")
            for exposure_id, row in reference_shifts.iterrows():
                with fits.open(
                    inputs / "exposures" / f"{exposure_id}_flt.fits", memmap=False
                ) as hdus:
                    assert np.isfinite(hdus[0].data).all()
                    assert row.dx_pix == hdus[0].header["SHIFTX"]
                    assert row.dy_pix == hdus[0].header["SHIFTY"]
            actual = actual.loc[reference_shifts.index]
            interpreted_error = np.abs(
                actual[["dx_pix", "dy_pix"]].to_numpy()
                + reference_shifts[["dx_pix", "dy_pix"]].to_numpy()
            ).max()
            actual_image = np.loadtxt(
                audit / "replayed/final" / visit / "drizzled_image.csv", delimiter=","
            )
            reference_image = np.loadtxt(reference / "drizzled_image.csv", delimiter=",")
            rmse = float(np.sqrt(np.mean((actual_image - reference_image) ** 2)))
            assert interpreted_error < 0.08
            assert rmse < 8
            measurements[visit] = {
                "max_opposite_sign_error_pix": float(interpreted_error),
                "image_rmse": rmse,
            }
    record(release, "hst-input-check", measurements)


def test_staged_public_contracts_match_runtime(release):
    from tasks.life_sciences.merfish_image_decoding_segmentation_1 import main as merfish
    from tasks.physical_sciences.hst_acs_wfc_visit_reduction import main as hst

    overlay = release / "post-validation-20260918/data-overrides"
    merfish_input = overlay / "life_sciences/merfish_image_decoding_segmentation_1/base/input"
    brief = (merfish_input / "task_description.txt").read_text()
    region = json.loads((merfish_input / "output_contract.json").read_text())[
        "reference_comparison_region"
    ]
    assert (region["x_min"], region["y_min"]) == (score_merfish.REFERENCE_MIN_PX,) * 2
    assert (region["x_max"], region["y_max"]) == (score_merfish.REFERENCE_MAX_PX,) * 2
    assert region["bounds"] == "inclusive"
    assert (
        brief.split("3. Extract")[1].split("5. Run Cellpose")[0]
        == merfish.config.task_description.split("3. Extract")[1].split("5. Run Cellpose")[0]
    )
    assert (
        brief.split("Reference-based decoding comparisons")[1]
        == merfish.config.task_description.split("Reference-based decoding comparisons")[1].split(
            "Total-count plausibility"
        )[0]
    )
    hst_prompt = (
        overlay / "physical_sciences/hst_acs_wfc_visit_reduction/base/input/TASK_PROMPT.md"
    ).read_text()
    assert (
        hst_prompt.split("## Alignment Shift Convention")[1].split("Do not modify")[0]
        == hst.config.task_description.split("## Alignment Shift Convention")[1].split(
            "## Photometry QC Contract"
        )[0]
    )
