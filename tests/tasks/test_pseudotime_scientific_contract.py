import asyncio
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.life_sciences.pseudotime_de import main
from tasks.life_sciences.pseudotime_de.scripts.score_outputs import score_submission

TASK = Path(main.__file__).resolve().parent


def test_estimator_preserves_original_workload_and_scientific_parameters():
    contract = json.loads((TASK / "estimator_contract.json").read_text())
    assert contract["estimator_id"] == "palantir-tradeseq-v2-20260910"
    assert contract["estimator_id"] == main.ESTIMATOR_ID
    assert main.config.reference_file.endswith("/degs_palantir_tradeseq_v2_20260910.csv")
    assert main.config.to_metadata()["reference_sha256"] == main.REFERENCE_SHA256
    assert (contract["input"]["cells"], contract["input"]["genes"]) == (4142, 16106)
    assert contract["input"]["additional_filtering"] is False
    preprocessing = contract["preprocessing"]
    assert preprocessing["working_dtype"] == "float64"
    assert preprocessing["normalize_total"] == {
        "target_sum": 10000,
        "exclude_highly_expressed": False,
    }
    assert preprocessing["highly_variable_genes"] == {
        "n_top_genes": 1500,
        "flavor": "cell_ranger",
        "batch_key": None,
    }
    assert preprocessing["pca"] == {
        "n_comps": 50,
        "zero_center": True,
        "svd_solver": "arpack",
        "random_state": 0,
        "dtype": "float64",
        "mask_var": "highly_variable",
    }
    trajectory = contract["trajectory"]
    assert trajectory["run_diffusion_maps"] == {
        "n_components": 5,
        "knn": 30,
        "alpha": 0,
        "seed": 0,
        "kernel_backend": "scanpy",
    }
    assert trajectory["early_cell"] == "Run5_164698952452459"
    assert trajectory["terminal_states"] == [
        {"lineage": "DC", "cell": "Run5_131097901611291"},
        {"lineage": "Mono", "cell": "Run5_134936662236454"},
        {"lineage": "Ery", "cell": "Run4_200562869397916"},
    ]
    assert trajectory["run_palantir"] == {
        "num_waypoints": 500,
        "knn": 30,
        "seed": 20,
        "scale_components": True,
        "use_early_cell_as_start": True,
        "max_iterations": 25,
    }
    differential = contract["differential_expression"]
    assert differential["lineage_order"] == ["DC", "Ery"]
    assert differential["R_seed"] == 27
    assert differential["fitGAM"]["nknots"] == 6
    assert differential["fitGAM"]["family"] == "nb"
    assert differential["patternTest"] == {
        "global": True,
        "pairwise": False,
        "nPoints": 12,
        "l2fc": "log2(1.5)",
        "eigenThresh": 0.01,
    }
    assert differential["p_adjust"] == {
        "method": "BH",
        "n": 16106,
        "scope": "all tested genes together, never per batch",
    }
    assert "effective p=1" in differential["failed_tests"]
    assert "no separate absolute logFC or fcMedian filter" in differential["selection"]
    assert contract["scoring"] == {
        "precision_threshold": 0.80,
        "recall_threshold": 0.80,
        "combine": "both required; compare unrounded",
    }


def test_public_contract_exposes_reproducibility_and_does_not_ship_reference_code():
    card = json.loads((TASK / "task_card.json").read_text())
    assert card["taskPrompt"] == main.config.task_description.replace(main.config.task_dir, "base")
    for text in (main.config.task_description, (TASK / "task_description.txt").read_text()):
        normalized = " ".join(text.split())
        for required in (
            "palantir-tradeseq-v2-20260910",
            "estimator_contract.json",
            "runtime_env/pyproject.toml",
            "runtime_env/R-versions.json",
            "target_sum=10000",
            "use_early_cell_as_start=True",
            "n=16106",
            "effective p=1",
        ):
            assert required in normalized
    assert not (TASK / "trajectory.py").exists()
    assert not (TASK / "fit_reference.R").exists()
    assert not (TASK / "reference_degs.csv").exists()


def test_exact_runtime_pins():
    manifest = tomllib.loads((TASK / "runtime_env/pyproject.toml").read_text())
    assert manifest["project"]["requires-python"] == "==3.11.15"
    dependencies = manifest["project"]["dependencies"]
    assert all(value.count("==") == 1 for value in dependencies)
    assert len(dependencies) == len({value.split("==")[0].lower() for value in dependencies})
    assert {"scanpy==1.11.5", "palantir==1.4.5", "numpy==1.26.4", "scipy==1.17.1"} <= set(
        dependencies
    )
    r_versions = json.loads((TASK / "runtime_env/R-versions.json").read_text())
    assert r_versions["R"] == "4.5.3"
    for package, version in {
        "tradeSeq": "1.24.0",
        "SingleCellExperiment": "1.32.0",
        "edgeR": "4.8.2",
        "mgcv": "1.9-4",
        "Matrix": "1.7-5",
    }.items():
        assert r_versions["packages"][package] == version


@pytest.fixture
def scientific_evidence():
    directory = os.environ.get("PSEUDOTIME_SCIENTIFIC_EVIDENCE")
    if directory is None:
        pytest.skip("Set PSEUDOTIME_SCIENTIFIC_EVIDENCE for full numerical reference receipts")
    root = Path(directory)
    report = json.loads((root / "scientific-validation.json").read_text())
    assert report["pass-a"]["genes"] == report["pass-b"]["genes"] == 16106
    models = json.loads((root / "model-repeat.json").read_text())
    assert models["genes"] == 16106
    assert models["all_coefficients_covariances_convergence_flags_exactly_identical"] is True
    return root, report


def test_genuine_full_reference_repeat_and_independent_global_bh(scientific_evidence):
    root, report = scientific_evidence
    reference = (root / "pass-a/reference_degs.csv").read_bytes()
    candidate = (root / "pass-b/de_genes.csv").read_bytes()
    score = score_submission({"de_genes.csv": candidate}, reference)
    assert score.score == score.precision == score.recall == 1
    assert hashlib.sha256(reference).hexdigest() == report["pass-a"]["reference_sha256"]
    assert hashlib.sha256(reference).hexdigest() == main.REFERENCE_SHA256
    for name in ("pass-a", "pass-b"):
        rows = list(csv.DictReader((root / name / "all_gene_statistics.csv").open()))
        assert len(rows) == len({row["gene"] for row in rows}) == 16106
        assert len(rows) > 1500
        effective = [float(row["effective_p"]) for row in rows]
        ordered = sorted(range(len(rows)), key=effective.__getitem__)
        oracle = [1.0] * len(rows)
        running_minimum = 1.0
        for rank in range(len(rows), 0, -1):
            index = ordered[rank - 1]
            running_minimum = min(running_minimum, len(rows) * effective[index] / rank)
            oracle[index] = running_minimum
        selected = set()
        for row, adjusted in zip(rows, oracle, strict=True):
            assert math.isclose(float(row["padj"]), adjusted, rel_tol=1e-12, abs_tol=1e-14)
            if row["converged"] != "TRUE" or row["finite_fit"] != "TRUE" or row["pvalue"] == "NA":
                assert float(row["effective_p"]) == 1
            if adjusted < 0.05:
                selected.add(row["gene"])
        assert selected == {row["gene"] for row in csv.DictReader(io.StringIO(candidate.decode()))}
        assert report[name]["source_counts_identical"] is True
        assert report[name]["lineage_alignment_validated"] is True


def test_genuine_repeated_pipeline_passes_actual_entrypoint(scientific_evidence):
    root, _ = scientific_evidence
    payloads = {
        main.config.reference_file: (root / "pass-a/reference_degs.csv").read_bytes(),
        main.config.output_file: (root / "pass-b/de_genes.csv").read_bytes(),
    }
    session = SimpleNamespace(
        file_exists=AsyncMock(side_effect=lambda path: path in payloads),
        read_bytes=AsyncMock(side_effect=payloads.__getitem__),
    )
    result = asyncio.run(
        main.evaluate(SimpleNamespace(metadata=main.config.to_metadata()), session)
    )
    assert result == [1.0]


@pytest.mark.parametrize("control", ["all_raw_genes", "hvg_only", "nonselected", "half_selected"])
def test_real_data_negative_selection_controls(scientific_evidence, control):
    root, _ = scientific_evidence
    reference = (root / "pass-a/reference_degs.csv").read_bytes()
    genes = [row["gene"] for row in csv.DictReader((root / "pass-a/genes.csv").open())]
    selected = [row["x"] for row in csv.DictReader(io.StringIO(reference.decode()))]
    candidates = {
        "all_raw_genes": genes,
        "hvg_only": [row["gene"] for row in csv.DictReader((root / "pass-a/hvg.csv").open())],
        "nonselected": sorted(set(genes) - set(selected)),
        "half_selected": selected[::2],
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["gene"])
    writer.writerows([gene] for gene in candidates[control])
    result = score_submission({"de_genes.csv": stream.getvalue().encode()}, reference)
    assert result.score == 0
    assert result.precision < 0.80 or result.recall < 0.80
