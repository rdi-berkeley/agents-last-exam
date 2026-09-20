import copy
import csv
import hashlib
import json
import shlex
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study import main as task
from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study.scripts import (
    verify_hidden_smoke as verifier,
)


METHODS = [
    "Full-Data Targeting Contrast (EIF)",
    "Standard LMTP Contrast (EIF)",
    "Targeted Bootstrap (Quantile)",
    "Targeted Bootstrap (Wald)",
]
RAW_COLUMNS = [
    "method",
    "scenario",
    "replicate_id",
    "seed",
    "bootstrap_seed",
    "n",
    "tau",
    "bootstrap_B",
    "folds",
    "learners_outcome",
    "learners_trt",
    "treatment_effect",
    "positivity_violation",
    "reference_policy",
    "target_policy",
    "estimate",
    "std_error",
    "conf_low",
    "conf_high",
    "tau_true",
]
SUMMARY_COLUMNS = [
    "method",
    "scenario",
    "bias",
    "empirical_se",
    "estimated_se",
    "se_ratio",
    "coverage",
    "ci_width",
]
PLAN = {
    "scenario": "independent",
    "n": 80,
    "tau": 3,
    "replications": 2,
    "base_seed": 41,
    "bootstrap_B": 8,
    "folds": 2,
    "tau_true_seed": 700,
    "learners_outcome": "SL.glm",
    "learners_trt": "SL.glm",
    "treatment_effect": "moderate",
    "positivity_violation": "moderate",
}


def write_csv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def bundle(tmp_path):
    input_dir = tmp_path / "input"
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    input_root = input_dir / "LTMLE_Targeted_Bootstrap_Task_INPUT"
    (input_root / "lmtp-bootstrap").mkdir(parents=True)
    reference.mkdir()
    candidate.mkdir()
    (input_root / "01_data_generation_longitudinal.R").write_text("identity\n")
    (input_root / "lmtp-bootstrap/DESCRIPTION").write_text("Package: lmtp\n")
    for name in verifier.REQUIRED_SCRIPT_NAMES:
        (candidate / name).write_text("identity\n")
    raw_rows = []
    for replicate_id in (1, 2):
        for method in METHODS:
            raw_rows.append(
                {
                    "method": method,
                    "scenario": PLAN["scenario"],
                    "replicate_id": str(replicate_id),
                    "seed": str(PLAN["base_seed"] + replicate_id - 1),
                    "bootstrap_seed": str(PLAN["base_seed"] + 5000 + replicate_id - 1),
                    **{
                        key: str(PLAN[key])
                        for key in (
                            "n",
                            "tau",
                            "bootstrap_B",
                            "folds",
                            "learners_outcome",
                            "learners_trt",
                            "treatment_effect",
                            "positivity_violation",
                        )
                    },
                    "reference_policy": "never_treat",
                    "target_policy": "always_treat",
                    "estimate": str(0.19 if replicate_id == 1 else 0.21),
                    "std_error": "0.03",
                    "conf_low": str(
                        (0.19 if replicate_id == 1 else 0.21)
                        - statistics.NormalDist().inv_cdf(0.975) * 0.03
                    ),
                    "conf_high": str(
                        (0.19 if replicate_id == 1 else 0.21)
                        + statistics.NormalDist().inv_cdf(0.975) * 0.03
                    ),
                    "tau_true": "0.2",
                }
            )
    summary = verifier._recompute_summary_rows(
        raw_rows=raw_rows, scenario_levels=["independent"], method_levels=METHODS
    )
    write_csv(reference / "raw_results.csv", RAW_COLUMNS, raw_rows)
    write_csv(reference / "public_raw_results.csv", RAW_COLUMNS, raw_rows)
    write_csv(reference / "summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(reference / "expected_summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(reference / verifier.FIXTURE_PLAN_NAME, list(PLAN), [PLAN])
    public = {
        "raw_reference_sha256": hashlib.sha256(
            (reference / "public_raw_results.csv").read_bytes()
        ).hexdigest(),
        "summary_columns": SUMMARY_COLUMNS,
        "row_match_keys": ["method", "scenario"],
        "metric_tolerances": {key: 0.000001 for key in SUMMARY_COLUMNS[2:]},
        "reference_sha256": hashlib.sha256(
            (reference / "expected_summary.csv").read_bytes()
        ).hexdigest(),
        "scenarios": [PLAN],
        "expected_methods": METHODS,
        "tau_true_by_scenario": {"independent": 0.2},
    }
    hidden = {
        "r_version": "4.3.2",
        "raw_metric_tolerances": {
            "estimate": 1e-5,
            "std_error": 1e-5,
            "conf_low": 1e-4,
            "conf_high": 1e-4,
        },
        "raw_result_columns": RAW_COLUMNS,
        "expected_methods": METHODS,
        "scenarios": [PLAN],
        "positive_fixture_reference_policy": "never_treat",
        "tau_true_by_scenario": {"independent": 0.2},
        "summary_metric_tolerances": public["metric_tolerances"],
        "frozen_expected_hashes": {
            "output_test_pos": {
                name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                for name in ("raw_results.csv", "summary.csv")
            }
        },
    }
    contract = {
        "public_benchmark": public,
        "hidden_smoke": hidden,
        "scientific_contract_version": verifier.CONTRACT_VERSION,
        "input_sha256": {
            "LTMLE_Targeted_Bootstrap_Task_INPUT/01_data_generation_longitudinal.R": hashlib.sha256(
                (input_root / "01_data_generation_longitudinal.R").read_bytes()
            ).hexdigest()
        },
    }
    (reference / "evaluation_contract.json").write_text(json.dumps(contract))
    write_csv(candidate / "raw_results.csv", RAW_COLUMNS, raw_rows)
    write_csv(candidate / "summary.csv", SUMMARY_COLUMNS, summary)
    (candidate / "report.pdf").write_bytes(b"%PDF-independent-fixture")
    return SimpleNamespace(
        input=input_dir,
        reference=reference,
        candidate=candidate,
        contract=contract,
        raw=raw_rows,
        scratch=tmp_path / "eval",
    )


def test_valid_configuration_and_independent_raw_rows(bundle):
    assert (
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)
        == bundle.contract
    )
    assert not verifier._validate_raw_results(
        raw_fieldnames=RAW_COLUMNS,
        raw_rows=bundle.raw,
        plan_rows=[PLAN],
        canonical_tau_true_by_scenario={"independent": 0.2},
        hidden_contract=bundle.contract["hidden_smoke"],
    )


@pytest.mark.parametrize(
    "name",
    [
        "evaluation_contract.json",
        "expected_summary.csv",
        "fixture_smoke_plan.csv",
        "raw_results.csv",
        "summary.csv",
    ],
)
def test_missing_or_empty_reference_is_configuration_error(bundle, name):
    (bundle.reference / name).unlink()
    with pytest.raises(verifier.EvaluatorConfigurationError, match="missing_evaluator_files"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)
    (bundle.reference / name).touch()
    with pytest.raises(verifier.EvaluatorConfigurationError, match="missing_evaluator_files"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)


@pytest.mark.parametrize("name", ["raw_results.csv", "summary.csv"])
def test_changed_frozen_reference_rejected(bundle, name):
    with (bundle.reference / name).open("a") as handle:
        handle.write("\n")
    with pytest.raises(verifier.EvaluatorConfigurationError, match="hash_mismatch"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)


def test_changed_plan_rejected(bundle):
    changed = {**PLAN, "n": 160}
    write_csv(bundle.reference / verifier.FIXTURE_PLAN_NAME, list(changed), [changed])
    with pytest.raises(verifier.EvaluatorConfigurationError, match="plan_mismatch"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)


@pytest.mark.parametrize(
    "column", ["replicate_id", "seed", "bootstrap_seed", "n", "tau", "bootstrap_B", "folds"]
)
def test_equivalent_integer_spellings(bundle, column):
    rows = copy.deepcopy(bundle.raw)
    for row in rows:
        row[column] = str(float(row[column]))
    assert not verifier._validate_raw_results(
        raw_fieldnames=RAW_COLUMNS,
        raw_rows=rows,
        plan_rows=[PLAN],
        canonical_tau_true_by_scenario={"independent": 0.2},
        hidden_contract=bundle.contract["hidden_smoke"],
    )


@pytest.mark.parametrize(
    "column", ["replicate_id", "seed", "tau_true", "estimate", "std_error", "conf_low", "conf_high"]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "wrong", ""])
def test_bad_candidate_numeric_is_rejected_without_crash(bundle, column, value):
    rows = copy.deepcopy(bundle.raw)
    rows[0][column] = value
    assert verifier._validate_raw_results(
        raw_fieldnames=RAW_COLUMNS,
        raw_rows=rows,
        plan_rows=[PLAN],
        canonical_tau_true_by_scenario={"independent": 0.2},
        hidden_contract=bundle.contract["hidden_smoke"],
    )


def setup_main(monkeypatch, bundle):
    monkeypatch.setattr(
        verifier,
        "_parse_args",
        lambda: SimpleNamespace(
            candidate_dir=str(bundle.candidate),
            input_dir=str(bundle.input),
            reference_dir=str(bundle.reference),
            positive_fixture_dir=str(bundle.reference),
            eval_data_dir=str(bundle.scratch),
            rscript_binary="unused",
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_materialize_candidate_bundle",
        lambda **kwargs: (bundle.candidate, {}, {"independent": 0.2}),
    )


def test_positive_full_verifier_control_flow(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)
    assert verifier.main() == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_missing_reference_never_becomes_candidate_zero(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)
    (bundle.reference / "raw_results.csv").unlink()
    assert verifier.main() == 2
    assert json.loads(capsys.readouterr().out)["error_type"] == "evaluator"


def test_candidate_r_script_failure_stays_candidate_failure(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)

    def fail(**kwargs):
        raise verifier.CandidateRerunError("bad submitted R")

    monkeypatch.setattr(verifier, "_materialize_candidate_bundle", fail)
    assert verifier.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "candidate_rerun_failed"
    assert "error_type" not in payload


@pytest.mark.parametrize("filename", ["raw_results.csv", "summary.csv"])
def test_short_candidate_row_is_candidate_failure(bundle, monkeypatch, capsys, filename):
    setup_main(monkeypatch, bundle)
    with (bundle.candidate / filename).open("a") as handle:
        handle.write("truncated,row\n")
    assert verifier.main() == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "candidate_csv_row_width_mismatch"


@pytest.mark.asyncio
async def test_host_evaluator_raises_when_reference_is_missing():
    class MissingSession:
        async def file_exists(self, path):
            return not path.endswith("/raw_results.csv")

        async def directory_exists(self, path):
            return False

    metadata = task.LtmleTargetedBootstrapConfig().to_metadata()
    with pytest.raises(RuntimeError, match="LTMLE evaluator files are missing"):
        await task.evaluate(SimpleNamespace(metadata=metadata), MissingSession())


class RuntimeSession:
    def __init__(self, runtimes):
        self.runtimes = runtimes
        self.probed = []

    async def run_command(self, command, *, check=False):
        arguments = shlex.split(command)
        assert arguments[:2] == ["timeout", "30"]
        assert arguments[3:] == ["-e", "cat(as.character(getRversion()))"]
        assert check is False
        binary = arguments[2]
        self.probed.append(binary)
        return_code, version = self.runtimes.get(binary, (127, ""))
        return {"return_code": return_code, "stdout": version, "stderr": ""}


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_result", [(0, "4.5.3"), (126, ""), (124, ""), (127, "")])
async def test_pinned_runtime_cannot_be_shadowed_by_wrapper(wrapper_result):
    metadata = task.LtmleTargetedBootstrapConfig().to_metadata()
    session = RuntimeSession(
        {
            task.PINNED_RSCRIPT_BINARY: (0, "4.3.2\n"),
            metadata["software_rscript"]: wrapper_result,
            task.RSCRIPT_BINARY: (0, "4.5.3"),
        }
    )
    assert (
        await task._select_rscript(
            session, software_rscript=metadata["software_rscript"], required_version="4.3.2"
        )
        == task.PINNED_RSCRIPT_BINARY
    )
    assert session.probed == [task.PINNED_RSCRIPT_BINARY]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["software_rscript", "system"])
async def test_runtime_fallback_must_probe_as_exact_pinned_version(fallback):
    metadata = task.LtmleTargetedBootstrapConfig().to_metadata()
    binary = metadata["software_rscript"] if fallback == "software_rscript" else task.RSCRIPT_BINARY
    session = RuntimeSession({binary: (0, "4.3.2")})
    assert (
        await task._select_rscript(
            session, software_rscript=metadata["software_rscript"], required_version="4.3.2"
        )
        == binary
    )
    assert session.probed[0] == task.PINNED_RSCRIPT_BINARY
    assert session.probed[-1] == binary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [(0, "4.5.3"), (0, "4.3.20"), (1, "4.3.2"), (126, ""), (124, ""), (127, "")],
)
async def test_no_usable_pinned_runtime_is_evaluator_error(invalid_result):
    metadata = task.LtmleTargetedBootstrapConfig().to_metadata()
    binaries = [task.PINNED_RSCRIPT_BINARY, metadata["software_rscript"], task.RSCRIPT_BINARY]
    session = RuntimeSession(dict.fromkeys(binaries, invalid_result))
    with pytest.raises(RuntimeError, match="evaluator pinned R 4.3.2 unavailable"):
        await task._select_rscript(
            session, software_rscript=metadata["software_rscript"], required_version="4.3.2"
        )
    assert session.probed == binaries


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden_passed", [True, False])
async def test_production_entrypoint_passes_verified_pinned_runtime(bundle, hidden_passed):
    metadata = task.LtmleTargetedBootstrapConfig().to_metadata()

    class EvaluationSession(RuntimeSession):
        def __init__(self):
            super().__init__({task.PINNED_RSCRIPT_BINARY: (0, "4.3.2")})
            self.verifier_arguments = None
            self.written = {}

        async def file_exists(self, path):
            return True

        async def directory_exists(self, path):
            return False

        async def read_file(self, path):
            root = (
                bundle.reference
                if path.startswith(metadata["reference_dir"] + "/")
                else bundle.candidate
            )
            return (root / Path(path).name).read_bytes().decode()

        async def write_file(self, path, content):
            self.written[path] = content

        async def run_command(self, command, *, check=False):
            arguments = shlex.split(command)
            if arguments[0] == "timeout":
                return await super().run_command(command, check=check)
            if arguments[0] == "python":
                self.verifier_arguments = arguments
                return {
                    "return_code": 0 if hidden_passed else 1,
                    "stdout": json.dumps({"passed": hidden_passed}),
                    "stderr": "",
                }
            assert arguments[0] in {"mkdir", "bash"}
            return {"return_code": 0, "stdout": "", "stderr": ""}

    session = EvaluationSession()
    assert await task.evaluate(SimpleNamespace(metadata=metadata), session) == [
        float(hidden_passed)
    ]
    arguments = session.verifier_arguments
    assert arguments[arguments.index("--rscript-binary") + 1] == task.PINNED_RSCRIPT_BINARY
    assert arguments[arguments.index("--candidate-dir") + 1] == metadata["remote_output_dir"]
    assert session.probed == [task.PINNED_RSCRIPT_BINARY]
    assert metadata["software_rscript"] != task.PINNED_RSCRIPT_BINARY
