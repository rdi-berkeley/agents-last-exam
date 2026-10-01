import copy
import hashlib
import json
import os
import re
import shlex
from pathlib import Path

import pytest

from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study import main as task
from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study.scripts import (
    verify_hidden_smoke as verifier,
)


ROOT = Path(task.__file__).parent
PINNED_RSCRIPT = "/opt/R/4.3.2/bin/Rscript"


@pytest.fixture
def guidance_overlay():
    override = os.environ.get("LTMLE_PUBLIC_GUIDANCE_OVERRIDE")
    if not override:
        pytest.skip("set LTMLE_PUBLIC_GUIDANCE_OVERRIDE to the staged task base overlay")
    return Path(override)


@pytest.mark.parametrize("surface", ["main", "task_card", "contract"])
def test_public_launch_guidance_uses_evaluator_runtime(surface):
    assert task.PINNED_RSCRIPT_BINARY == PINNED_RSCRIPT
    if surface == "main":
        text = task.LtmleTargetedBootstrapConfig().task_description
    elif surface == "task_card":
        text = json.loads((ROOT / "task_card.json").read_text())["taskPrompt"]
    else:
        text = (ROOT / "SCIENTIFIC_CONTRACT.md").read_text()
    assert f"{PINNED_RSCRIPT} 05_run_full_simulation_longitudinal.R" in text
    assert "runtime_manifest.json" in text
    assert not re.search(r"(?<![\w/])Rscript\s+(?:-e\b|05_run_)", text)


def test_main_and_card_runtime_instructions_are_identical():
    description = task.LtmleTargetedBootstrapConfig().task_description
    prompt = json.loads((ROOT / "task_card.json").read_text())["taskPrompt"]
    runtime_section = description.split("\n\nRuntime:\n", 1)[1].split("\n\n", 1)[0]
    assert runtime_section == prompt.split("\n\nRuntime:\n", 1)[1].split("\n\n", 1)[0]
    assert "every R session" in runtime_section
    assert "do not install replacement CRAN/GitHub packages" in runtime_section


def test_documented_fresh_session_commands_preserve_analysis_apis():
    text = (ROOT / "SCIENTIFIC_CONTRACT.md").read_text()
    commands = [
        shlex.split(line)
        for block in re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        for line in block.splitlines()
        if line.strip()
    ]
    assert commands == [
        [PINNED_RSCRIPT, "05_run_full_simulation_longitudinal.R"],
        [
            PINNED_RSCRIPT,
            "-e",
            'source("06_analyze_part2_results_longitudinal.R"); '
            'analyze_part2_results_longitudinal(output_dir = ".")',
        ],
        [
            PINNED_RSCRIPT,
            "-e",
            'source("06b_analyze_by_sample_size.R"); '
            'summary_df <- read.csv("summary.csv", stringsAsFactors = FALSE); '
            "grouped <- analyze_results_by_sample_size(summary_df); "
            "stopifnot(length(grouped) >= 1L)",
        ],
    ]


@pytest.mark.parametrize("filename", ["README_output.md", "SCIENTIFIC_CONTRACT.md"])
def test_staged_public_guidance_matches_canonical_document(guidance_overlay, filename):
    document = guidance_overlay / "input/public_benchmark" / filename
    assert document.read_bytes() == (ROOT / "SCIENTIFIC_CONTRACT.md").read_bytes()
    assert {
        str(path.relative_to(guidance_overlay))
        for path in guidance_overlay.rglob("*")
        if path.is_file()
    } == {
        "input/public_benchmark/README_output.md",
        "input/public_benchmark/SCIENTIFIC_CONTRACT.md",
        "reference/evaluation_contract.json",
    }


def test_staged_contract_changes_exactly_two_document_hashes(guidance_overlay):
    original_path = os.environ.get("LTMLE_ORIGINAL_EVALUATION_CONTRACT")
    if not original_path:
        pytest.skip("set LTMLE_ORIGINAL_EVALUATION_CONTRACT to the unchanged frozen contract")
    original_text = Path(original_path).read_text()
    staged_text = (guidance_overlay / "reference/evaluation_contract.json").read_text()
    original = json.loads(original_text)
    staged = json.loads(staged_text)
    expected = copy.deepcopy(original)
    restored_text = staged_text
    for filename in ("README_output.md", "SCIENTIFIC_CONTRACT.md"):
        key = f"public_benchmark/{filename}"
        actual_hash = hashlib.sha256((guidance_overlay / "input" / key).read_bytes()).hexdigest()
        original_hash = original["input_sha256"][key]
        assert actual_hash != original_hash
        expected["input_sha256"][key] = actual_hash
        restored_text = restored_text.replace(
            f'"{key}": "{actual_hash}"', f'"{key}": "{original_hash}"', 1
        )
    assert json.dumps(staged, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert restored_text == original_text


def test_unchanged_verifier_accepts_merged_guidance_contract(guidance_overlay, tmp_path):
    evidence = os.environ.get("LTMLE_REPAIR_EVIDENCE")
    if not evidence:
        pytest.skip("set LTMLE_REPAIR_EVIDENCE to the independently computed v2 bundle")
    baseline = Path(evidence) / "staged/base"
    contract_path = guidance_overlay / "reference/evaluation_contract.json"
    contract = json.loads(contract_path.read_text())
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    input_dir.mkdir()
    reference_dir.mkdir()
    for relative in contract["input_sha256"]:
        source = guidance_overlay / "input" / relative
        if not source.is_file():
            source = baseline / "input" / relative
        assert source.is_file(), source
        destination = input_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve())
    for filename in (
        "expected_summary.csv",
        "public_raw_results.csv",
        "raw_results.csv",
        "summary.csv",
        "fixture_smoke_plan.csv",
    ):
        source = baseline / "reference" / filename
        assert source.is_file(), source
        (reference_dir / filename).symlink_to(source.resolve())
    (reference_dir / "evaluation_contract.json").symlink_to(contract_path.resolve())
    assert verifier._load_evaluator_contract(input_dir, reference_dir, reference_dir) == contract
