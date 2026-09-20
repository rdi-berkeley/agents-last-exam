"""Public numerical package contains no biological I/O or answer artifacts."""

import ast
import os
from pathlib import Path

import pytest

from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.scripts import (
    score_outputs,
)


ROOT = Path(score_outputs.__file__).parents[1]


def test_public_runtime_contains_only_computation():
    package = ROOT / "runtime/deg_inference"
    assert {path.name for path in package.glob("*.py")} == {
        "__init__.py",
        "coefficients.py",
        "inference.py",
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        forbidden = {"pandas", "pathlib", "csv", "json", "pickle", "requests", "argparse"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not forbidden.intersection(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] not in forbidden
            elif isinstance(node, ast.Call):
                name = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", "")
                )
                assert name not in {
                    "open",
                    "read_csv",
                    "read_table",
                    "load",
                    "loadtxt",
                    "read_bytes",
                    "read_text",
                }
    assert not list(package.glob("*.tsv")) and not list(package.glob("*.npz"))


def test_promoted_function_bodies_preserve_reviewed_algorithm():
    if not (ROOT / "scripts/probe_corrected_objective.py").is_file():
        pytest.skip("Historical algorithm comparison requires private audit sources")
    old_probe = ast.parse((ROOT / "scripts/probe_corrected_objective.py").read_text())
    new_probe = ast.parse((ROOT / "runtime/deg_inference/coefficients.py").read_text())
    old_functions = {
        node.name: ast.dump(node) for node in old_probe.body if isinstance(node, ast.FunctionDef)
    }
    for node in new_probe.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == old_functions[node.name]
    old = ast.parse((ROOT / "scripts/experimental_inference.py").read_text())
    new = ast.parse((ROOT / "runtime/deg_inference/inference.py").read_text())
    old_functions = {
        node.name: node for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for node in new.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            original = old_functions[node.name]
            for candidate in (node, original):
                for child in ast.walk(candidate):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        child.value = child.value.replace(
                            "The diagnostic region fallback", "The coefficient optimizer"
                        )
            assert ast.dump(node) == ast.dump(original)


def test_supported_dataset_and_stats_inference_interfaces():
    if os.environ.get("DEG_GUEST_SCIENTIFIC") != "1":
        pytest.skip("Pinned scientific runtime is in the assigned DEG guest")
    import inspect
    import numpy as np
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.runtime.deg_inference import (
        ObjectivePreservingInference,
    )

    assert "inference" in inspect.signature(DeseqDataSet).parameters
    assert "inference" in inspect.signature(DeseqStats).parameters
    inference = ObjectivePreservingInference(n_cpus=1)
    metadata = pd.DataFrame(
        {"batch": ["a", "a", "b", "b"] * 2, "condition": ["normal"] * 4 + ["tumor"] * 4}
    )
    counts = pd.DataFrame(np.random.default_rng(62).negative_binomial(3, 0.15, (8, 32)) + 1)
    counts.columns = counts.columns.astype(str)
    dataset = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=["batch", "condition"],
        ref_level=["condition", "normal"],
        inference=inference,
        refit_cooks=True,
        quiet=True,
    )
    dataset.deseq2()
    stats = DeseqStats(
        dataset,
        contrast=["condition", "tumor", "normal"],
        alpha=0.05,
        inference=inference,
        cooks_filter=True,
        independent_filter=True,
        quiet=True,
    )
    stats.summary()
    assert dataset.inference is inference and stats.inference is inference
    assert len(stats.results_df) == 32
