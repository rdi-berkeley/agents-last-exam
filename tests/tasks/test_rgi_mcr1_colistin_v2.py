"""Public regression coverage for the RGI MCR-1 task contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "tasks" / "life_sciences" / "rgi_mcr1_colistin_v2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(REPO_ROOT))
SCORER = load_module("public_rgi_mcr1_scorer", TASK_ROOT / "scripts" / "score_output.py")
TASK = load_module("public_rgi_mcr1_task", TASK_ROOT / "main.py")

REFERENCE = json.dumps(
    {
        "score_mapping": {"pass_threshold": 0.75},
        "grading": {
            "gene_name": {
                "reference_value": "ARO-X.7",
                "full_credit_prefix": "ARO",
                "partial_credit_contains": "transferase",
            },
            "percent_identity": {
                "target": 88.5,
                "full_credit_tolerance": 0.5,
                "partial_credit_tolerance": 2.0,
            },
            "drug_class": {
                "reference_value": "synthetic class alpha",
                "required_keyword": "alpha",
            },
            "resistance_mechanism": {
                "reference_value": "synthetic mechanism beta",
                "required_keyword": "beta",
            },
        },
    }
)

EXACT_ANSWER = {
    "best_hit_aro": "ARO-X.7",
    "percent_identity": 88.5,
    "drug_class": "synthetic class alpha",
    "resistance_mechanism": "synthetic mechanism beta",
}


def score(answer: dict[str, object]):
    return SCORER.score_output_payloads(
        output_json_text=json.dumps(answer),
        reference_json_text=REFERENCE,
    )


def test_config_uses_canonical_task_path():
    config = TASK.RgiMcr1ColistinV2Config(REMOTE_ROOT_DIR="/remote")
    expected = "/remote/life_sciences/rgi_mcr1_colistin_v2/base"
    assert config.task_dir == expected
    assert config.data_task_dir == expected
    assert config.remote_output_dir == f"{expected}/output"
    assert "amr_contig_annotation_instance_1" not in config.task_description


def test_exact_and_normalized_answers_pass():
    exact = score(EXACT_ANSWER)
    assert exact.score == 1.0
    assert exact.passed

    normalized = score(
        {
            **EXACT_ANSWER,
            "best_hit_aro": " aro-x.7 ",
            "drug_class": "SYNTHETIC   CLASS ALPHA",
            "resistance_mechanism": "synthetic\tmechanism beta",
        }
    )
    assert normalized.score == 1.0
    assert normalized.passed


def test_critical_fields_cannot_be_bypassed():
    wrong_allele = score({**EXACT_ANSWER, "best_hit_aro": "ARO-X"})
    assert wrong_allele.score == 0.5
    assert not wrong_allele.passed

    old_tolerance_only = score({**EXACT_ANSWER, "percent_identity": 88.56})
    assert old_tolerance_only.identity_tolerance == 0.05
    assert old_tolerance_only.score == 0.5
    assert not old_tolerance_only.passed


def test_substring_only_categorical_fields_receive_no_credit():
    result = score(
        {
            **EXACT_ANSWER,
            "drug_class": "synthetic class alpha plus another class",
            "resistance_mechanism": "putative synthetic mechanism beta",
        }
    )
    assert result.drug_class_score == 0.0
    assert result.resistance_mechanism_score == 0.0
    assert result.score == 0.5
    assert not result.passed


def test_invalid_candidate_is_a_safe_failure():
    result = score({**EXACT_ANSWER, "percent_identity": "88.5"})
    assert result.score == 0.0
    assert not result.passed
    assert not result.valid
