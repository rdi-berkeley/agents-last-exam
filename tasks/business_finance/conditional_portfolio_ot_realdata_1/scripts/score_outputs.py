"""Local scorer for business_finance/conditional_portfolio_ot_realdata_1."""

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_MANIFEST_KEYS = [
    "repository_path",
    "repo_commit",
    "study_variant",
    "selected_data_files",
    "retained_start_date",
    "retained_end_date",
    "num_retained_dates",
    "asset_columns",
    "predictor_columns",
    "dropped_rows_missing_dates",
    "dropped_rows_missing_predictors",
    "dropped_rows_missing_returns",
    "dropped_rows_duplicate_dates",
]
PORTFOLIO_COLUMNS = ["date", "method", "weights_json", "realized_portfolio_return"]
AGG_COLUMNS = [
    "method",
    "out_of_sample_mean_return",
    "out_of_sample_variance",
    "mean_variance_utility",
    "sharpe_ratio",
    "realized_volatility",
    "runtime_seconds",
]
REQUIRED_METHODS = ["cond_mean_variance", "mean_variance"]
REQUIRED_NOTES_TOKENS = ["88e4ec3", "cond_mean_variance", "mean_variance"]


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failure(reason: str, *, hard_gate: str | None = None, **details: Any) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=hard_gate or reason,
        details=details,
    )


def _success(**details: Any) -> ScoreResult:
    return ScoreResult(score=1.0, passed=True, reason="all checks passed", details=details)


def _load_json_text(text: str | bytes) -> Any:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else str(text)
    return json.loads(payload)


def _read_csv_text(text: str | bytes, *, sep: str = ",") -> pd.DataFrame:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else str(text)
    return pd.read_csv(io.StringIO(payload), sep=sep)


def _compare_json_like(agent_value: Any, reference_value: Any, tolerance: float) -> bool:
    if isinstance(reference_value, dict):
        if not isinstance(agent_value, dict) or set(agent_value) != set(reference_value):
            return False
        return all(
            _compare_json_like(agent_value[key], reference_value[key], tolerance)
            for key in reference_value
        )
    if isinstance(reference_value, list):
        if not isinstance(agent_value, list) or len(agent_value) != len(reference_value):
            return False
        if reference_value and all(isinstance(x, str) for x in reference_value):
            if not all(isinstance(x, str) for x in agent_value):
                return False
            return sorted(agent_value) == sorted(reference_value)
        return all(
            _compare_json_like(agent_item, ref_item, tolerance)
            for agent_item, ref_item in zip(agent_value, reference_value)
        )
    if isinstance(reference_value, float):
        try:
            return math.isclose(float(agent_value), reference_value, rel_tol=tolerance, abs_tol=tolerance)
        except Exception:
            return False
    return agent_value == reference_value


def _validate_policy_run_notes(notes_text: str) -> ScoreResult | None:
    stripped = notes_text.strip()
    if not stripped:
        return _failure("policy_run_notes.md is empty")
    missing_tokens = [token for token in REQUIRED_NOTES_TOKENS if token not in stripped]
    if missing_tokens:
        return _failure(
            "policy_run_notes.md is missing required benchmark details",
            missing_tokens=missing_tokens,
        )
    return None


def _load_and_validate_manifest(
    manifest_text: str,
    reference_manifest: dict[str, Any],
    tolerance: float,
) -> tuple[dict[str, Any] | None, ScoreResult | None]:
    try:
        manifest = _load_json_text(manifest_text)
    except Exception as exc:
        return None, _failure(f"data_manifest.json is not valid JSON: {exc}")

    if not isinstance(manifest, dict):
        return None, _failure("data_manifest.json must decode to an object")
    if list(manifest.keys()) != REQUIRED_MANIFEST_KEYS:
        return None, _failure(
            "data_manifest.json keys must match the required schema exactly",
            keys=list(manifest.keys()),
        )
    if not _compare_json_like(manifest, reference_manifest, tolerance):
        return None, _failure("data_manifest.json does not match the hidden reference manifest")
    return manifest, None


def _parse_weights_json(raw: str, asset_columns: set[str]) -> tuple[dict[str, float] | None, str | None]:
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return None, f"weights_json is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "weights_json must decode to an object"
    converted: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            return None, "weights_json keys must be strings"
        if key not in asset_columns:
            return None, f"weights_json contains unknown asset column {key!r}"
        try:
            number = float(value)
        except Exception:
            return None, f"weights_json value for {key!r} is not numeric"
        if not math.isfinite(number):
            return None, f"weights_json value for {key!r} must be finite"
        if number < -1e-8:
            return None, f"weights_json value for {key!r} must be non-negative"
        converted[key] = number
    total_weight = sum(converted.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        return None, f"weights_json weights must sum to 1.0, got {total_weight}"
    return converted, None


def _load_and_validate_portfolio_results(
    portfolio_text: str,
    reference_portfolio: pd.DataFrame,
    asset_columns: list[str],
    tolerance: float,
) -> tuple[pd.DataFrame | None, ScoreResult | None]:
    try:
        frame = _read_csv_text(portfolio_text)
    except Exception as exc:
        return None, _failure(f"portfolio_results.csv is unreadable: {exc}")

    if list(frame.columns) != PORTFOLIO_COLUMNS:
        return None, _failure(
            "portfolio_results.csv columns must match exactly",
            columns=list(frame.columns),
        )

    normalized = frame.copy()
    normalized["date"] = normalized["date"].astype(str).str.strip()
    if not normalized["date"].str.fullmatch(r"\d{8}").all():
        return None, _failure("portfolio_results.csv dates must use YYYYMMDD")

    if not normalized["method"].isin(REQUIRED_METHODS).all():
        return None, _failure(
            "portfolio_results.csv contains unknown method labels",
            methods=sorted(normalized["method"].astype(str).unique().tolist()),
        )

    normalized["realized_portfolio_return"] = pd.to_numeric(
        normalized["realized_portfolio_return"], errors="coerce"
    )
    if normalized["realized_portfolio_return"].isna().any():
        return None, _failure("portfolio_results.csv contains non-numeric realized_portfolio_return")

    asset_set = set(asset_columns)
    for idx, raw in normalized["weights_json"].items():
        _, error = _parse_weights_json(str(raw), asset_set)
        if error:
            return None, _failure(
                "portfolio_results.csv contains invalid weights_json",
                row=int(idx),
                error=error,
            )

    normalized = normalized.sort_values(["date", "method"]).reset_index(drop=True)
    reference_sorted = reference_portfolio.copy()
    reference_sorted["date"] = reference_sorted["date"].astype(str).str.strip()
    reference_sorted["method"] = reference_sorted["method"].astype(str).str.strip()
    reference_sorted["realized_portfolio_return"] = pd.to_numeric(
        reference_sorted["realized_portfolio_return"], errors="coerce"
    )
    reference_sorted = reference_sorted.sort_values(["date", "method"]).reset_index(drop=True)

    if len(normalized) != len(reference_sorted):
        return None, _failure(
            "portfolio_results.csv row count does not match the hidden reference",
            agent_rows=int(len(normalized)),
            reference_rows=int(len(reference_sorted)),
        )

    agent_pairs = normalized[["date", "method"]]
    reference_pairs = reference_sorted[["date", "method"]]
    if not agent_pairs.equals(reference_pairs):
        return None, _failure("portfolio_results.csv date/method rows do not match the hidden reference")

    if not (
        (normalized["realized_portfolio_return"] - reference_sorted["realized_portfolio_return"])
        .abs()
        .le(tolerance)
        .all()
    ):
        return None, _failure(
            "portfolio_results.csv realized returns differ from the hidden reference beyond tolerance"
        )

    return normalized, None


def _compute_metrics_from_portfolio(
    portfolio_frame: pd.DataFrame, risk_aversion_parameter: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in REQUIRED_METHODS:
        realized = portfolio_frame.loc[
            portfolio_frame["method"] == method, "realized_portfolio_return"
        ].astype(float)
        mean_return = float(realized.mean())
        variance = float(realized.var(ddof=0))
        volatility = float(math.sqrt(variance))
        sharpe = float(mean_return / volatility * math.sqrt(252.0)) if volatility > 0 else 0.0
        rows.append(
            {
                "method": method,
                "out_of_sample_mean_return": mean_return,
                "out_of_sample_variance": variance,
                "mean_variance_utility": mean_return - risk_aversion_parameter * variance,
                "sharpe_ratio": sharpe,
                "realized_volatility": volatility,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _load_and_validate_aggregate_metrics(
    aggregate_text: str,
    candidate_portfolio: pd.DataFrame,
    reference_aggregate: pd.DataFrame,
    risk_aversion_parameter: float,
    tolerance: float,
) -> tuple[pd.DataFrame | None, ScoreResult | None]:
    try:
        frame = _read_csv_text(aggregate_text, sep="\t")
    except Exception as exc:
        return None, _failure(f"aggregate_metrics.tsv is unreadable: {exc}")

    if list(frame.columns) != AGG_COLUMNS:
        return None, _failure(
            "aggregate_metrics.tsv columns must match exactly",
            columns=list(frame.columns),
        )

    normalized = frame.copy()
    if set(normalized["method"].astype(str)) != set(REQUIRED_METHODS):
        return None, _failure(
            "aggregate_metrics.tsv must contain exactly the required methods",
            methods=sorted(normalized["method"].astype(str).tolist()),
        )

    for column in AGG_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[AGG_COLUMNS[1:]].isna().any().any():
        return None, _failure("aggregate_metrics.tsv contains non-numeric metric values")

    if not (normalized["runtime_seconds"] >= 0).all():
        return None, _failure("aggregate_metrics.tsv runtime_seconds must be non-negative")

    normalized = normalized.sort_values("method").reset_index(drop=True)
    reference_sorted = reference_aggregate.copy()
    reference_sorted["method"] = reference_sorted["method"].astype(str).str.strip()
    for column in AGG_COLUMNS[1:]:
        reference_sorted[column] = pd.to_numeric(reference_sorted[column], errors="coerce")
    reference_sorted = reference_sorted.sort_values("method").reset_index(drop=True)
    recomputed = _compute_metrics_from_portfolio(candidate_portfolio, risk_aversion_parameter)

    metric_columns = AGG_COLUMNS[1:-1]
    for column in metric_columns:
        if not (
            (normalized[column] - recomputed[column]).abs().le(tolerance).all()
        ):
            return None, _failure(
                "aggregate_metrics.tsv is inconsistent with portfolio_results.csv",
                column=column,
            )
        if not (
            (normalized[column] - reference_sorted[column]).abs().le(tolerance).all()
        ):
            return None, _failure(
                "aggregate_metrics.tsv differs from the hidden reference beyond tolerance",
                column=column,
            )

    return normalized, None


def score_submission_files(
    *,
    agent_manifest_text: str,
    agent_policy_run_notes_text: str,
    agent_portfolio_results_text: str,
    agent_aggregate_metrics_text: str,
    reference_manifest_text: str,
    reference_portfolio_results_text: str,
    reference_aggregate_metrics_text: str,
    risk_aversion_parameter: float,
    tolerance: float = 1e-6,
) -> ScoreResult:
    try:
        reference_manifest = _load_json_text(reference_manifest_text)
        reference_portfolio = _read_csv_text(reference_portfolio_results_text)
        reference_aggregate = _read_csv_text(reference_aggregate_metrics_text, sep="\t")
    except Exception as exc:
        return _failure(f"hidden reference artifacts are unreadable: {exc}", hard_gate="evaluator reference invalid")

    notes_error = _validate_policy_run_notes(agent_policy_run_notes_text)
    if notes_error is not None:
        return notes_error

    manifest, manifest_error = _load_and_validate_manifest(
        agent_manifest_text,
        reference_manifest=reference_manifest,
        tolerance=tolerance,
    )
    if manifest_error is not None:
        return manifest_error
    assert manifest is not None

    portfolio, portfolio_error = _load_and_validate_portfolio_results(
        agent_portfolio_results_text,
        reference_portfolio=reference_portfolio,
        asset_columns=list(reference_manifest["asset_columns"]),
        tolerance=tolerance,
    )
    if portfolio_error is not None:
        return portfolio_error
    assert portfolio is not None

    aggregate, aggregate_error = _load_and_validate_aggregate_metrics(
        agent_aggregate_metrics_text,
        candidate_portfolio=portfolio,
        reference_aggregate=reference_aggregate,
        risk_aversion_parameter=risk_aversion_parameter,
        tolerance=tolerance,
    )
    if aggregate_error is not None:
        return aggregate_error
    assert aggregate is not None

    return _success(
        checked_files=[
            "data_manifest.json",
            "policy_run_notes.md",
            "portfolio_results.csv",
            "aggregate_metrics.tsv",
        ],
        methods=REQUIRED_METHODS,
        portfolio_rows=int(len(portfolio)),
        aggregate_rows=int(len(aggregate)),
    )


def _read_dir_file(path: Path, name: str) -> str:
    return path.joinpath(name).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--risk-aversion-parameter", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    try:
        result = score_submission_files(
            agent_manifest_text=_read_dir_file(args.agent_dir, "data_manifest.json"),
            agent_policy_run_notes_text=_read_dir_file(args.agent_dir, "policy_run_notes.md"),
            agent_portfolio_results_text=_read_dir_file(args.agent_dir, "portfolio_results.csv"),
            agent_aggregate_metrics_text=_read_dir_file(args.agent_dir, "aggregate_metrics.tsv"),
            reference_manifest_text=_read_dir_file(args.reference_dir, "data_manifest.json"),
            reference_portfolio_results_text=_read_dir_file(
                args.reference_dir, "portfolio_results.csv"
            ),
            reference_aggregate_metrics_text=_read_dir_file(
                args.reference_dir, "aggregate_metrics.tsv"
            ),
            risk_aversion_parameter=args.risk_aversion_parameter,
            tolerance=args.tolerance,
        )
    except FileNotFoundError as exc:
        result = _failure(f"missing required file during local scoring: {exc}")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
