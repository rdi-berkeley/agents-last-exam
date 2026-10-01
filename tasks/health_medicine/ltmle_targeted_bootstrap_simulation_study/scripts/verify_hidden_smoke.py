"""VM-side hidden-smoke verifier for ltmle_targeted_bootstrap_simulation_study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

REQUIRED_SCRIPT_NAMES = [
    "02_variance_methods_longitudinal.R",
    "03_simulation_runner_longitudinal.R",
    "04_analysis_functions_longitudinal.R",
    "05_run_full_simulation_longitudinal.R",
    "06_analyze_part2_results_longitudinal.R",
    "06b_analyze_by_sample_size.R",
]
FIXTURE_PLAN_NAME = "fixture_smoke_plan.csv"
CONTRACT_VERSION = "ltmle-cumulative-v2-20260910"
DERIVATION_TOLERANCE = 1.1e-6


class EvaluatorConfigurationError(RuntimeError):
    pass


class CandidateRerunError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--positive-fixture-dir", required=True)
    parser.add_argument("--eval-data-dir", required=True)
    parser.add_argument("--rscript-binary", required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result(
    *,
    passed: bool,
    reason: str,
    details: dict[str, Any] | None = None,
    infrastructure_error: bool = False,
) -> int:
    payload = {
        "passed": passed,
        "reason": reason,
        "details": details or {},
    }
    if infrastructure_error:
        payload["error_type"] = "evaluator"
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    if infrastructure_error:
        return 2
    return 0 if passed else 1


def _ensure_present(path: Path, errors: list[str], label: str) -> None:
    if not path.exists():
        errors.append(f"missing_{label}:{path}")


def _load_evaluator_contract(
    input_dir: Path, reference_dir: Path, positive_fixture_dir: Path
) -> dict[str, Any]:
    required = [
        reference_dir / "evaluation_contract.json",
        reference_dir / "expected_summary.csv",
        reference_dir / "public_raw_results.csv",
        positive_fixture_dir / FIXTURE_PLAN_NAME,
        positive_fixture_dir / "raw_results.csv",
        positive_fixture_dir / "summary.csv",
        input_dir / "LTMLE_Targeted_Bootstrap_Task_INPUT/01_data_generation_longitudinal.R",
        input_dir / "LTMLE_Targeted_Bootstrap_Task_INPUT/lmtp-bootstrap/DESCRIPTION",
    ]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise EvaluatorConfigurationError(f"missing_evaluator_files:{missing}")
    contract = json.loads((reference_dir / "evaluation_contract.json").read_text(encoding="utf-8"))
    if contract.get("scientific_contract_version") != CONTRACT_VERSION:
        raise EvaluatorConfigurationError("scientific_contract_version_mismatch")
    for relative_path, expected_hash in contract["input_sha256"].items():
        path = input_dir / relative_path
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise EvaluatorConfigurationError(f"input_hash_mismatch:{relative_path}")
    hidden = contract["hidden_smoke"]
    if (
        _sha256_file(reference_dir / "public_raw_results.csv")
        != contract["public_benchmark"]["raw_reference_sha256"]
    ):
        raise EvaluatorConfigurationError("public_raw_reference_hash_mismatch")
    if (
        _sha256_file(reference_dir / "expected_summary.csv")
        != contract["public_benchmark"]["reference_sha256"]
    ):
        raise EvaluatorConfigurationError("public_reference_hash_mismatch")
    for name in ("raw_results.csv", "summary.csv"):
        expected = hidden["frozen_expected_hashes"]["output_test_pos"][name]
        if _sha256_file(positive_fixture_dir / name) != expected:
            raise EvaluatorConfigurationError(f"positive_fixture_hash_mismatch:{name}")
    plan = _load_plan(positive_fixture_dir / FIXTURE_PLAN_NAME)
    expected_plan = hidden["scenarios"]
    if [{key: str(value) for key, value in row.items()} for row in plan] != [
        {key: str(value) for key, value in row.items()} for row in expected_plan
    ]:
        raise EvaluatorConfigurationError("positive_fixture_plan_mismatch")
    for path in (reference_dir / "expected_summary.csv", positive_fixture_dir / "summary.csv"):
        columns, rows = _read_csv(path)
        public = contract["public_benchmark"]
        if columns != public["summary_columns"] or not rows:
            raise EvaluatorConfigurationError(f"reference_summary_schema_mismatch:{path.name}")
        if _compare_summary_maps(candidate_rows=rows, reference_rows=rows, contract_section=public):
            raise EvaluatorConfigurationError(f"invalid_reference_summary:{path.name}")
    for raw_path, summary_path, section in (
        (positive_fixture_dir / "raw_results.csv", positive_fixture_dir / "summary.csv", hidden),
        (reference_dir / "public_raw_results.csv", reference_dir / "expected_summary.csv", public),
    ):
        raw_columns, raw_rows = _read_csv(raw_path)
        raw_errors = _validate_raw_results(
            raw_fieldnames=raw_columns,
            raw_rows=raw_rows,
            plan_rows=section["scenarios"],
            canonical_tau_true_by_scenario=section["tau_true_by_scenario"],
            hidden_contract={**hidden, "scenarios": section["scenarios"]},
        )
        if raw_errors:
            raise EvaluatorConfigurationError(
                f"reference_scientific_invariants:{raw_path.name}:{raw_errors[:5]}"
            )
        derived = _recompute_summary_rows(
            raw_rows=raw_rows,
            scenario_levels=[row["scenario"] for row in section["scenarios"]],
            method_levels=hidden["expected_methods"],
        )
        _, summary = _read_csv(summary_path)
        if _compare_summary_maps(
            candidate_rows=summary,
            reference_rows=derived,
            contract_section={
                "row_match_keys": ["method", "scenario"],
                "metric_tolerances": {
                    key: DERIVATION_TOLERANCE for key in public["metric_tolerances"]
                },
            },
        ):
            raise EvaluatorConfigurationError(
                f"reference_summary_derivation_failed:{summary_path.name}"
            )
    return contract


def _probe_rscript_system_lib(rscript_binary: str) -> str:
    # The pinned Stage 4 runtime ships R 4.3.2 with a pre-installed package
    # closure at `<R_HOME>/lib/R/library/`. The hidden-smoke verifier must
    # reuse those packages (which were compiled at install_software.sh time)
    # instead of rebuilding the closure from source: re-compiling
    # data.table 1.18.2.1 against R 4.3.2 fails because its source calls
    # `base::sort_by`, which was added in R 4.4.0.
    probe = subprocess.run(
        [rscript_binary, "-e", "cat(normalizePath(.Library, mustWork=FALSE))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "rscript_system_lib_probe_failed\n"
            f"stdout:\n{probe.stdout[-2000:]}\n"
            f"stderr:\n{probe.stderr[-2000:]}"
        )
    system_lib = probe.stdout.strip().splitlines()[-1].strip() if probe.stdout.strip() else ""
    if not system_lib or not Path(system_lib).is_dir():
        raise RuntimeError(f"rscript_system_lib_invalid:{system_lib!r}")
    return system_lib


def _prepare_r_library(
    *,
    contract: dict[str, Any],
    copied_input_dir: Path,
    eval_data_dir: Path,
    rscript_binary: str,
) -> tuple[str, str]:
    hidden = contract["hidden_smoke"]
    temp_root = Path(tempfile.mkdtemp(prefix="tmp_", dir=str(eval_data_dir)))
    system_lib = _probe_rscript_system_lib(rscript_binary)
    versions = hidden["declared_runtime_package_versions"]
    fingerprints = hidden["lmtp_function_fingerprints"]
    checks = [
        f"stopifnot(as.character(getRversion()) == {json.dumps(hidden['r_version'])})",
        *(
            f"stopifnot(utils::compareVersion(as.character(packageVersion({json.dumps(name)})), {json.dumps(version)}) == 0)"
            for name, version in versions.items()
        ),
        *(
            "stopifnot(digest::digest(paste(deparse("
            f'getFromNamespace({json.dumps(name)}, "lmtp"), width.cutoff=500L), '
            f'collapse="\\n"), algo="sha256", serialize=FALSE) == {json.dumps(fingerprint)})'
            for name, fingerprint in fingerprints.items()
        ),
    ]
    env = {
        **dict(os.environ),
        "R_LIBS": "",
        "R_LIBS_SITE": "",
        "R_LIBS_USER": system_lib,
        "TMPDIR": str(temp_root),
        "TMP": str(temp_root),
        "TEMP": str(temp_root),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    result = subprocess.run(
        [rscript_binary, "-e", f".libPaths({json.dumps(system_lib)});\n" + "\n".join(checks)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise EvaluatorConfigurationError(
            f"pinned_runtime_mismatch:stdout={result.stdout[-2000:]} stderr={result.stderr[-4000:]}"
        )
    return system_lib, str(temp_root)


def _compute_canonical_tau_true_map(
    *,
    plan_rows: list[dict[str, Any]],
    copied_input_dir: Path,
    env: dict[str, str],
    rscript_binary: str,
) -> dict[str, float]:
    dgp_path = (
        copied_input_dir
        / "LTMLE_Targeted_Bootstrap_Task_INPUT"
        / "01_data_generation_longitudinal.R"
    )
    tau_true_by_scenario: dict[str, float] = {}
    for row in plan_rows:
        compute_r = f"""
source({json.dumps(str(dgp_path))}, local = FALSE)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

with_preserved_rng <- function(seed, code) {{
  code_expr <- substitute(code)
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) {{
    saved_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  }}
  on.exit({{
    if (had_seed) {{
      assign(".Random.seed", saved_seed, envir = .GlobalEnv)
    }} else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {{
      rm(".Random.seed", envir = .GlobalEnv)
    }}
  }}, add = TRUE)
  set.seed(as.integer(seed))
  eval(code_expr, envir = parent.frame())
}}

resolve_beta_psi <- function(treatment_effect) {{
  switch(
    treatment_effect,
    null = 0,
    small = 0.25,
    moderate = 0.5,
    large = 1.0,
    stop(paste("Unknown treatment effect:", treatment_effect))
  )
}}

resolve_beta_p <- function(positivity_violation) {{
  switch(
    positivity_violation,
    none = -2,
    mild = -1.5,
    moderate = -1,
    severe = -0.5,
    stop(paste("Unknown positivity violation:", positivity_violation))
  )
}}

tau_true <- with_preserved_rng(
  {int(row["tau_true_seed"])},
  compute_true_effect_mc(
    n_mc = 100000,
    tau = as.integer({int(row["tau"])}),
    beta_psi = resolve_beta_psi({json.dumps(row["treatment_effect"])}),
    beta_p = resolve_beta_p({json.dumps(row["positivity_violation"])})
  )
)
cat(format(tau_true, digits = 16, scientific = FALSE))
"""
        result = subprocess.run(
            [rscript_binary, "-e", compute_r],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "tau_true_compute_failed\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        tau_true_raw = result.stdout.strip()
        if not tau_true_raw:
            raise RuntimeError(
                "tau_true_compute_failed:no_output\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
        tau_true_value = float(tau_true_raw.splitlines()[-1].strip())
        if not math.isfinite(tau_true_value):
            raise RuntimeError(f"tau_true_compute_failed:non_finite:{tau_true_raw!r}")
        tau_true_by_scenario[row["scenario"]] = tau_true_value
    return tau_true_by_scenario


def _materialize_candidate_bundle(
    *,
    contract: dict[str, Any],
    candidate_dir: Path,
    input_dir: Path,
    positive_fixture_dir: Path,
    eval_data_dir: Path,
    rscript_binary: str,
) -> tuple[Path, dict[str, str], dict[str, float]]:
    scratch_root = Path(tempfile.mkdtemp(prefix="scratch_", dir=str(eval_data_dir)))
    scratch_output = scratch_root / "output"
    scratch_input = scratch_root / "input"
    scratch_output.mkdir(parents=True, exist_ok=True)

    shutil.copytree(input_dir, scratch_input, dirs_exist_ok=True)
    for script_name in REQUIRED_SCRIPT_NAMES:
        shutil.copy2(candidate_dir / script_name, scratch_output / script_name)
    shutil.copy2(positive_fixture_dir / FIXTURE_PLAN_NAME, scratch_output / FIXTURE_PLAN_NAME)

    r_libs_user, temp_root = _prepare_r_library(
        contract=contract,
        copied_input_dir=scratch_input,
        eval_data_dir=eval_data_dir,
        rscript_binary=rscript_binary,
    )

    hidden_contract = contract["hidden_smoke"]
    dgp_env_name = hidden_contract["dgp_source_env_var"]
    dgp_path = (
        scratch_input / "LTMLE_Targeted_Bootstrap_Task_INPUT" / "01_data_generation_longitudinal.R"
    )
    env = {
        **dict(os.environ),
        "R_LIBS": "",
        "R_LIBS_SITE": "",
        "R_LIBS_USER": r_libs_user,
        "TMPDIR": temp_root,
        "TMP": temp_root,
        "TEMP": temp_root,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LC_ALL": "C",
        "TZ": "UTC",
        dgp_env_name: str(dgp_path),
    }
    canonical_tau_true_by_scenario = _compute_canonical_tau_true_map(
        plan_rows=_load_plan(positive_fixture_dir / FIXTURE_PLAN_NAME),
        copied_input_dir=scratch_input,
        env=env,
        rscript_binary=rscript_binary,
    )

    commands = [
        [rscript_binary, "05_run_full_simulation_longitudinal.R"],
        [
            rscript_binary,
            "-e",
            'source("06_analyze_part2_results_longitudinal.R"); '
            'analyze_part2_results_longitudinal(output_dir = ".")',
        ],
        [
            rscript_binary,
            "-e",
            'source("06b_analyze_by_sample_size.R"); '
            'summary_df <- read.csv("summary.csv", stringsAsFactors = FALSE); '
            "grouped <- analyze_results_by_sample_size(summary_df); "
            "stopifnot(length(grouped) >= 1L)",
        ],
    ]
    run_logs: dict[str, str] = {}
    for index, command in enumerate(commands, start=1):
        try:
            result = subprocess.run(
                command,
                cwd=scratch_output,
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            raise CandidateRerunError(f"hidden_smoke_timeout:{index}") from exc
        run_logs[f"command_{index}"] = " ".join(command)
        run_logs[f"stdout_{index}"] = result.stdout[-4000:]
        run_logs[f"stderr_{index}"] = result.stderr[-4000:]
        if result.returncode != 0:
            raise CandidateRerunError(
                f"hidden_smoke_rerun_failed:{index}\n"
                f"stdout:\n{result.stdout[-4000:]}\n"
                f"stderr:\n{result.stderr[-4000:]}"
            )
    return scratch_output, run_logs, canonical_tau_true_by_scenario


def _load_plan(plan_path: Path) -> list[dict[str, Any]]:
    _, rows = _read_csv(plan_path)
    plan_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        for key in (
            "n",
            "tau",
            "replications",
            "base_seed",
            "bootstrap_B",
            "folds",
            "tau_true_seed",
        ):
            normalized[key] = int(row[key])
        plan_rows.append(normalized)
    return plan_rows


def _sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def _round6(value: float) -> float:
    return round(float(value), 6)


def _compare_summary_maps(
    *,
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    contract_section: dict[str, Any],
) -> list[dict[str, Any]]:
    row_keys = list(contract_section["row_match_keys"])
    tolerances = dict(contract_section["metric_tolerances"])

    def build_map(
        rows: list[dict[str, str]],
    ) -> tuple[dict[tuple[str, ...], dict[str, str]], list[tuple[str, ...]]]:
        row_map: dict[tuple[str, ...], dict[str, str]] = {}
        duplicates: list[tuple[str, ...]] = []
        for row in rows:
            key = tuple(row[key_name] for key_name in row_keys)
            if key in row_map:
                duplicates.append(key)
            row_map[key] = row
        return row_map, duplicates

    candidate_map, candidate_dupes = build_map(candidate_rows)
    reference_map, reference_dupes = build_map(reference_rows)
    mismatches: list[dict[str, Any]] = []
    if candidate_dupes:
        mismatches.append(
            {
                "type": "candidate_duplicate_keys",
                "duplicate_keys": [list(key) for key in candidate_dupes],
            }
        )
        return mismatches
    if reference_dupes:
        mismatches.append(
            {
                "type": "reference_duplicate_keys",
                "duplicate_keys": [list(key) for key in reference_dupes],
            }
        )
        return mismatches
    candidate_keys = set(candidate_map)
    reference_keys = set(reference_map)
    if candidate_keys != reference_keys:
        mismatches.append(
            {
                "type": "row_key_mismatch",
                "missing_keys": [list(key) for key in sorted(reference_keys - candidate_keys)],
                "unexpected_keys": [list(key) for key in sorted(candidate_keys - reference_keys)],
            }
        )
        return mismatches

    for row_key in sorted(reference_keys):
        candidate_row = candidate_map[row_key]
        reference_row = reference_map[row_key]
        for metric_name, tolerance in tolerances.items():
            try:
                candidate_value = float(candidate_row[metric_name])
                reference_value = float(reference_row[metric_name])
            except (ValueError, TypeError):
                mismatches.append(
                    {
                        "type": "invalid_numeric",
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_row[metric_name],
                        "reference_value": reference_row[metric_name],
                    }
                )
                continue
            if not math.isfinite(candidate_value) or not math.isfinite(reference_value):
                mismatches.append(
                    {
                        "type": "non_finite_numeric",
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_value,
                        "reference_value": reference_value,
                    }
                )
                continue
            delta = abs(candidate_value - reference_value)
            if delta > float(tolerance):
                mismatches.append(
                    {
                        "type": "metric_mismatch",
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_value,
                        "reference_value": reference_value,
                        "delta": delta,
                        "tolerance": float(tolerance),
                    }
                )
    return mismatches


def _recompute_summary_rows(
    *,
    raw_rows: list[dict[str, str]],
    scenario_levels: list[str],
    method_levels: list[str],
) -> list[dict[str, str]]:
    keyed_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in raw_rows:
        keyed_rows.setdefault((row["method"], row["scenario"]), []).append(row)

    summary_rows: list[dict[str, str]] = []
    for scenario_name in scenario_levels:
        for method_name in method_levels:
            slice_rows = keyed_rows[(method_name, scenario_name)]
            estimates = [float(row["estimate"]) for row in slice_rows]
            std_errors = [float(row["std_error"]) for row in slice_rows]
            conf_low = [float(row["conf_low"]) for row in slice_rows]
            conf_high = [float(row["conf_high"]) for row in slice_rows]
            tau_true_values = {float(row["tau_true"]) for row in slice_rows}
            if len(tau_true_values) != 1:
                raise RuntimeError(f"tau_true_not_unique:{method_name}:{scenario_name}")
            tau_true = next(iter(tau_true_values))
            empirical_se = _sample_sd(estimates)
            estimated_se = statistics.fmean(std_errors)
            coverage = statistics.fmean(
                1.0 if conf_low[i] <= tau_true <= conf_high[i] else 0.0
                for i in range(len(slice_rows))
            )
            ci_width = statistics.fmean(conf_high[i] - conf_low[i] for i in range(len(slice_rows)))
            summary_rows.append(
                {
                    "method": method_name,
                    "scenario": scenario_name,
                    "bias": f"{_round6(statistics.fmean(value - tau_true for value in estimates)):.6f}",
                    "empirical_se": f"{_round6(empirical_se):.6f}",
                    "estimated_se": f"{_round6(estimated_se):.6f}",
                    "se_ratio": f"{_round6(estimated_se / empirical_se if empirical_se > 0 else 0.0):.6f}",
                    "coverage": f"{_round6(coverage):.6f}",
                    "ci_width": f"{_round6(ci_width):.6f}",
                }
            )
    return summary_rows


def _validate_raw_results(
    *,
    raw_fieldnames: list[str],
    raw_rows: list[dict[str, str]],
    plan_rows: list[dict[str, Any]],
    canonical_tau_true_by_scenario: dict[str, float],
    hidden_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    expected_columns = list(hidden_contract["raw_result_columns"])
    expected_methods = list(hidden_contract["expected_methods"])
    expected_scenarios = [row["scenario"] for row in hidden_contract["scenarios"]]
    expected_policy = hidden_contract["positive_fixture_reference_policy"]
    expected_target_policy = "always_treat"

    if raw_fieldnames != expected_columns:
        errors.append(
            {
                "type": "raw_schema_mismatch",
                "candidate_fieldnames": raw_fieldnames,
                "expected_fieldnames": expected_columns,
            }
        )
        return errors
    if not raw_rows:
        errors.append({"type": "raw_empty"})
        return errors

    plan_by_scenario = {row["scenario"]: row for row in plan_rows}
    expected_keys: set[tuple[str, str, int]] = set()
    for plan_row in plan_rows:
        for replicate_id in range(1, int(plan_row["replications"]) + 1):
            for method_name in expected_methods:
                expected_keys.add((method_name, plan_row["scenario"], replicate_id))

    observed_keys: set[tuple[str, str, int]] = set()
    observed_methods = {row["method"] for row in raw_rows}
    observed_scenarios = {row["scenario"] for row in raw_rows}
    if set(expected_methods) != observed_methods:
        errors.append(
            {
                "type": "method_set_mismatch",
                "observed_methods": sorted(observed_methods),
                "expected_methods": expected_methods,
            }
        )
    if set(expected_scenarios) != observed_scenarios:
        errors.append(
            {
                "type": "scenario_set_mismatch",
                "observed_scenarios": sorted(observed_scenarios),
                "expected_scenarios": expected_scenarios,
            }
        )

    for row in raw_rows:
        scenario_name = row["scenario"]
        method_name = row["method"]
        try:
            replicate_value = float(row["replicate_id"])
            if not math.isfinite(replicate_value) or not replicate_value.is_integer():
                raise ValueError("replicate_id is not an integer")
            replicate_id = int(replicate_value)
        except (ValueError, TypeError, OverflowError):
            errors.append({"type": "invalid_replicate_id", "value": row["replicate_id"]})
            continue
        observed_key = (method_name, scenario_name, replicate_id)
        if observed_key in observed_keys:
            errors.append({"type": "duplicate_raw_key", "key": list(observed_key)})
            continue
        observed_keys.add(observed_key)

        if scenario_name not in plan_by_scenario:
            errors.append({"type": "unexpected_scenario", "scenario": scenario_name})
            continue
        plan_row = plan_by_scenario[scenario_name]
        if method_name not in expected_methods:
            errors.append({"type": "unexpected_method", "method": method_name})

        expected_seed = int(plan_row["base_seed"]) + replicate_id - 1
        expected_bootstrap_seed = int(plan_row["base_seed"]) + 5000 + replicate_id - 1
        exact_checks = {
            "seed": expected_seed,
            "bootstrap_seed": expected_bootstrap_seed,
            "n": int(plan_row["n"]),
            "tau": int(plan_row["tau"]),
            "bootstrap_B": int(plan_row["bootstrap_B"]),
            "folds": int(plan_row["folds"]),
            "learners_outcome": plan_row["learners_outcome"],
            "learners_trt": plan_row["learners_trt"],
            "treatment_effect": plan_row["treatment_effect"],
            "positivity_violation": plan_row["positivity_violation"],
            "reference_policy": expected_policy,
            "target_policy": expected_target_policy,
        }
        for key_name, expected_value in exact_checks.items():
            observed_value = row[key_name]
            if isinstance(expected_value, int):
                try:
                    matches = float(observed_value) == expected_value
                except (ValueError, TypeError, OverflowError):
                    matches = False
            else:
                matches = str(observed_value) == str(expected_value)
            if not matches:
                errors.append(
                    {
                        "type": "plan_alignment_mismatch",
                        "key": key_name,
                        "observed_value": observed_value,
                        "expected_value": expected_value,
                        "row_key": [method_name, scenario_name, replicate_id],
                    }
                )

        tau_true_expected = canonical_tau_true_by_scenario[scenario_name]
        try:
            tau_true_observed = float(row["tau_true"])
        except (ValueError, TypeError, OverflowError):
            tau_true_observed = math.nan
        if (
            not math.isfinite(tau_true_observed)
            or abs(tau_true_observed - tau_true_expected) > 1e-6
        ):
            errors.append(
                {
                    "type": "tau_true_mismatch",
                    "row_key": [method_name, scenario_name, replicate_id],
                    "observed_value": tau_true_observed,
                    "expected_value": tau_true_expected,
                }
            )

        for metric_name in ("estimate", "std_error", "conf_low", "conf_high"):
            try:
                metric_value = float(row[metric_name])
            except (ValueError, TypeError, OverflowError):
                metric_value = math.nan
            if not math.isfinite(metric_value):
                errors.append(
                    {
                        "type": "non_finite_raw_value",
                        "metric": metric_name,
                        "row_key": [method_name, scenario_name, replicate_id],
                        "value": row[metric_name],
                    }
                )

    if observed_keys != expected_keys:
        errors.append(
            {
                "type": "raw_key_set_mismatch",
                "missing_keys": [list(key) for key in sorted(expected_keys - observed_keys)],
                "unexpected_keys": [list(key) for key in sorted(observed_keys - expected_keys)],
            }
        )

    if not errors:
        errors.extend(_scientific_raw_invariants(raw_rows))
    return errors


def _scientific_raw_invariants(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    errors = []
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    normal = statistics.NormalDist().inv_cdf(0.975)
    for row in raw_rows:
        key = (row["scenario"], str(int(float(row["replicate_id"]))))
        grouped.setdefault(key, {})[row["method"]] = row
        estimate, error, lower, upper = (
            float(row[name]) for name in ("estimate", "std_error", "conf_low", "conf_high")
        )
        if error < 0 or lower > upper or abs(estimate) > 1 + 1e-8:
            errors.append(
                {"type": "invalid_scientific_range", "key": list(key), "method": row["method"]}
            )
        if row["method"] != "Targeted Bootstrap (Quantile)":
            if (
                max(
                    abs(lower - (estimate - normal * error)),
                    abs(upper - (estimate + normal * error)),
                )
                > 1e-7
            ):
                errors.append(
                    {"type": "wald_interval_mismatch", "key": list(key), "method": row["method"]}
                )
        elif lower < -1 - 1e-8 or upper > 1 + 1e-8:
            errors.append({"type": "quantile_outside_estimand_range", "key": list(key)})
    for key, methods in grouped.items():
        full = methods["Full-Data Targeting Contrast (EIF)"]
        wald = methods["Targeted Bootstrap (Wald)"]
        quantile = methods["Targeted Bootstrap (Quantile)"]
        if (
            max(abs(float(full["estimate"]) - float(row["estimate"])) for row in (wald, quantile))
            > 1e-8
        ):
            errors.append({"type": "shared_full_data_point_mismatch", "key": list(key)})
        if abs(float(wald["std_error"]) - float(quantile["std_error"])) > 1e-8:
            errors.append({"type": "shared_bootstrap_se_mismatch", "key": list(key)})
    return errors


def main() -> int:
    try:
        args = _parse_args()
        candidate_dir = Path(args.candidate_dir)
        input_dir = Path(args.input_dir)
        reference_dir = Path(args.reference_dir)
        positive_fixture_dir = Path(args.positive_fixture_dir)
        eval_data_dir = Path(args.eval_data_dir)

        contract = _load_evaluator_contract(input_dir, reference_dir, positive_fixture_dir)
        errors: list[str] = []
        _ensure_present(candidate_dir, errors, "candidate_dir")
        for script_name in REQUIRED_SCRIPT_NAMES:
            _ensure_present(candidate_dir / script_name, errors, f"candidate_script_{script_name}")
        if errors:
            return _write_result(passed=False, reason="missing_paths", details={"errors": errors})

        eval_data_dir.mkdir(parents=True, exist_ok=True)
        hidden_contract = contract["hidden_smoke"]
        public_contract = contract["public_benchmark"]

        scratch_output_dir, run_logs, canonical_tau_true_by_scenario = (
            _materialize_candidate_bundle(
                contract=contract,
                candidate_dir=candidate_dir,
                input_dir=input_dir,
                positive_fixture_dir=positive_fixture_dir,
                eval_data_dir=eval_data_dir,
                rscript_binary=args.rscript_binary,
            )
        )

        raw_path = scratch_output_dir / "raw_results.csv"
        summary_path = scratch_output_dir / "summary.csv"
        report_path = scratch_output_dir / "report.pdf"
        if not raw_path.exists():
            return _write_result(
                passed=False, reason="hidden_raw_results_missing", details=run_logs
            )
        if not summary_path.exists():
            return _write_result(passed=False, reason="hidden_summary_missing", details=run_logs)
        if not report_path.exists() or report_path.stat().st_size <= 0:
            return _write_result(
                passed=False, reason="hidden_report_missing_or_empty", details=run_logs
            )

        raw_fieldnames, raw_rows = _read_csv(raw_path)
        summary_fieldnames, summary_rows = _read_csv(summary_path)
        if any(
            None in row or any(value is None for value in row.values())
            for row in raw_rows + summary_rows
        ):
            return _write_result(passed=False, reason="candidate_csv_row_width_mismatch")
        plan_rows = _load_plan(positive_fixture_dir / FIXTURE_PLAN_NAME)
        _, positive_reference_raw_rows = _read_csv(positive_fixture_dir / "raw_results.csv")
        positive_summary_fieldnames, positive_reference_summary_rows = _read_csv(
            positive_fixture_dir / "summary.csv"
        )

        if summary_fieldnames != list(public_contract["summary_columns"]):
            return _write_result(
                passed=False,
                reason="hidden_summary_schema_mismatch",
                details={
                    "candidate_fieldnames": summary_fieldnames,
                    "expected_fieldnames": list(public_contract["summary_columns"]),
                },
            )
        if positive_summary_fieldnames != list(public_contract["summary_columns"]):
            return _write_result(
                passed=False,
                reason="positive_fixture_summary_schema_mismatch",
                infrastructure_error=True,
                details={
                    "fixture_fieldnames": positive_summary_fieldnames,
                    "expected_fieldnames": list(public_contract["summary_columns"]),
                },
            )

        raw_errors = _validate_raw_results(
            raw_fieldnames=raw_fieldnames,
            raw_rows=raw_rows,
            plan_rows=plan_rows,
            canonical_tau_true_by_scenario=canonical_tau_true_by_scenario,
            hidden_contract=hidden_contract,
        )
        if raw_errors:
            return _write_result(
                passed=False,
                reason="hidden_raw_contract_failed",
                details={"errors": raw_errors[:20], "error_count": len(raw_errors), **run_logs},
            )

        raw_mismatches = _compare_summary_maps(
            candidate_rows=[
                {**row, "replicate_id": str(int(float(row["replicate_id"])))} for row in raw_rows
            ],
            reference_rows=[
                {**row, "replicate_id": str(int(float(row["replicate_id"])))}
                for row in positive_reference_raw_rows
            ],
            contract_section={
                "row_match_keys": ["method", "scenario", "replicate_id"],
                "metric_tolerances": hidden_contract["raw_metric_tolerances"],
            },
        )
        if raw_mismatches:
            return _write_result(
                passed=False,
                reason="hidden_raw_numerical_mismatch",
                details={
                    "mismatches": raw_mismatches[:20],
                    "mismatch_count": len(raw_mismatches),
                    **run_logs,
                },
            )

        expected_scenarios = [row["scenario"] for row in hidden_contract["scenarios"]]
        expected_methods = list(hidden_contract["expected_methods"])
        recomputed_summary_rows = _recompute_summary_rows(
            raw_rows=raw_rows,
            scenario_levels=expected_scenarios,
            method_levels=expected_methods,
        )
        derivation_mismatches = _compare_summary_maps(
            candidate_rows=summary_rows,
            reference_rows=recomputed_summary_rows,
            contract_section={
                "row_match_keys": public_contract["row_match_keys"],
                "metric_tolerances": {
                    key: DERIVATION_TOLERANCE for key in public_contract["metric_tolerances"]
                },
            },
        )
        if derivation_mismatches:
            return _write_result(
                passed=False,
                reason="hidden_summary_derivation_failed",
                details={
                    "mismatches": derivation_mismatches[:20],
                    "mismatch_count": len(derivation_mismatches),
                    **run_logs,
                },
            )

        hidden_mismatches = _compare_summary_maps(
            candidate_rows=summary_rows,
            reference_rows=positive_reference_summary_rows,
            contract_section={
                "row_match_keys": public_contract["row_match_keys"],
                "metric_tolerances": hidden_contract["summary_metric_tolerances"],
            },
        )
        if hidden_mismatches:
            return _write_result(
                passed=False,
                reason="hidden_summary_tolerance_failed",
                details={
                    "mismatches": hidden_mismatches[:20],
                    "mismatch_count": len(hidden_mismatches),
                    **run_logs,
                },
            )

        return _write_result(
            passed=True,
            reason="hidden_smoke_passed",
            details={
                "raw_row_count": len(raw_rows),
                "summary_row_count": len(summary_rows),
                **run_logs,
            },
        )
    except CandidateRerunError as exc:
        return _write_result(
            passed=False, reason="candidate_rerun_failed", details={"error": str(exc)}
        )
    except Exception as exc:  # pragma: no cover - defensive path for remote execution
        return _write_result(
            passed=False,
            reason="verifier_exception",
            infrastructure_error=True,
            details={"error": str(exc), "traceback": traceback.format_exc()[-12000:]},
        )


if __name__ == "__main__":
    import os

    raise SystemExit(main())
