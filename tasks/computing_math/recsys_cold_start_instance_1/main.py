"""AgentHLE task: recsys_cold_start_instance_1."""

import csv
import io
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only
    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "recsys_cold_start_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"

EXPECTED_WARM_COLUMNS = ["user_id", "item_id", "predicted_rating", "rank"]
EXPECTED_COLD_COLUMNS = ["user_id", "item_id", "predicted_score", "rank"]
EXPECTED_COMBINED_COLUMNS = ["user_id", "item_id", "predicted_score", "rank", "item_type"]
REQUIRED_OUTPUT_FILES = [
    "predictions_warm.csv",
    "predictions_cold.csv",
    "predictions_combined.csv",
    "evaluation_report.json",
    "model_config.json",
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
WARM_NDCG_THRESHOLD = 0.03
WARM_HITRATE_THRESHOLD = 0.15
WARM_RMSE_THRESHOLD = 5.0
COLD_NDCG_THRESHOLD = 0.05
COLD_COVERAGE_THRESHOLD = 0.40


@dataclass
class RecsysColdStartConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def interactions_file(self) -> str:
        return f"{self.input_dir}/interactions.csv"

    @property
    def item_metadata_file(self) -> str:
        return f"{self.input_dir}/item_metadata.csv"

    @property
    def item_embeddings_file(self) -> str:
        return f"{self.input_dir}/item_embeddings.json"

    @property
    def user_profiles_file(self) -> str:
        return f"{self.input_dir}/user_profiles.csv"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def setup_runtime_script(self) -> str:
        return f"{self.software_dir}/setup_runtime_env.sh"

    @property
    def run_runtime_script(self) -> str:
        return f"{self.software_dir}/run_recommender_env.sh"

    @property
    def task_description(self) -> str:
        return f"""\
You are building a hybrid recommender system on a Linux VM.

Visible inputs:
- `{self.interactions_file}`
- `{self.item_metadata_file}`
- `{self.item_embeddings_file}`
- `{self.user_profiles_file}`
- `{self.runtime_env_dir}/pyproject.toml`
- `{self.runtime_env_dir}/uv.lock`
- `{self.setup_runtime_script}`
- `{self.run_runtime_script}`

Your task:
1. Build a warm-item recommender using both `rating` and `watch_percentage`.
2. Build a cold-start model from metadata and embeddings without interaction data for cold items.
3. Use the canonical per-user temporal split: 70% train, 10% validation, 20% test.
4. Produce a combined ranking that contains both warm and cold items.
5. Write exactly these files under `{self.remote_output_dir}`:
   - `predictions_warm.csv` — columns: `user_id, item_id, predicted_rating, rank`
   - `predictions_cold.csv` — columns: `user_id, item_id, predicted_score, rank`
   - `predictions_combined.csv` — columns: `user_id, item_id, predicted_score, rank, item_type` (item_type is "warm" or "cold")
   - `evaluation_report.json` — top-level keys: `warm_items`, `cold_items`, `combined`. `warm_items` must include `NDCG@10`, `HitRate@10`, `RMSE`.
   - `model_config.json` — must include keys: `warm_model`, `cold_model`, `hybrid`, `data_split`. `hybrid` must include a key named `method` or `combination` describing the merging strategy. `data_split` must include `method` (containing "temporal"), `train_ratio`, `val_ratio`, `test_ratio`.

Environment notes:
- Use `{self.setup_runtime_script}` to materialize the staged task-local Python environment.
- Use `{self.run_runtime_script}` to run Python inside that environment.
- Do not modify files under `{self.input_dir}`.
- Do not modify hidden evaluator data.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "interactions_file": self.interactions_file,
                "item_metadata_file": self.item_metadata_file,
                "item_embeddings_file": self.item_embeddings_file,
                "user_profiles_file": self.user_profiles_file,
                "runtime_env_dir": self.runtime_env_dir,
                "setup_runtime_script": self.setup_runtime_script,
                "run_runtime_script": self.run_runtime_script,
                "canonical_gcs_root": "gs://ale-data-all/computing_math/recsys_cold_start_instance_1/base/",
            }
        )
        return metadata


config = RecsysColdStartConfig()


@cb.tasks_config(split="train")
def load():
    cfg = RecsysColdStartConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _run_command(session: cb.DesktopSession, command: str, *, check: bool = False) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


async def _read_remote_bytes(session: cb.DesktopSession, path: str) -> bytes:
    try:
        return await session.read_bytes(path)
    except Exception:
        text = await session.read_file(path)
        return text.encode("utf-8")


def _decode_csv_bytes(payload: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return (reader.fieldnames or [], rows)


def _json_loads(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _parse_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return default


def _build_metadata_labels(rows: List[Dict[str, str]]) -> Dict[int, bool]:
    return {_parse_int(row["item_id"]): _parse_bool(row["is_cold_start"]) for row in rows}


def _group_ranked_rows(
    rows: List[Dict[str, str]],
    *,
    score_key: str,
) -> Dict[int, List[Tuple[int, float, int, Dict[str, str]]]]:
    grouped: Dict[int, List[Tuple[int, float, int, Dict[str, str]]]] = defaultdict(list)
    for row in rows:
        user_id = _parse_int(row.get("user_id"))
        item_id = _parse_int(row.get("item_id"))
        rank = _parse_int(row.get("rank"), default=10**9)
        score = _parse_float(row.get(score_key))
        grouped[user_id].append((item_id, score, rank, row))
    for user_id in grouped:
        grouped[user_id].sort(key=lambda item: (item[2], -item[1], item[0]))
    return grouped


def _compute_split_counts(n: int) -> Tuple[int, int, int]:
    test_n = max(1, int(round(n * TEST_RATIO)))
    val_n = max(1, int(round(n * VAL_RATIO)))
    if test_n + val_n >= n:
        test_n = max(1, min(test_n, n - 2))
        val_n = max(1, min(val_n, n - test_n - 1))
    train_n = n - val_n - test_n
    return train_n, val_n, test_n


def _canonical_test_truth(interaction_rows: List[Dict[str, str]]) -> Tuple[Dict[int, Dict[int, float]], Dict[int, set], Dict[Tuple[int, int], float]]:
    grouped: Dict[int, List[Tuple[str, int, float, float]]] = defaultdict(list)
    for row in interaction_rows:
        grouped[_parse_int(row["user_id"])].append(
            (
                row["timestamp"],
                _parse_int(row["item_id"]),
                _parse_float(row["rating"]),
                _parse_float(row["watch_percentage"]),
            )
        )

    gains_by_user: Dict[int, Dict[int, float]] = {}
    relevant_items_by_user: Dict[int, set] = {}
    ratings_by_pair: Dict[Tuple[int, int], float] = {}

    for user_id, entries in grouped.items():
        entries.sort(key=lambda item: (item[0], item[1]))
        train_n, val_n, test_n = _compute_split_counts(len(entries))
        test_entries = entries[train_n + val_n : train_n + val_n + test_n]
        gains = {item_id: rating * watch for _, item_id, rating, watch in test_entries}
        gains_by_user[user_id] = gains
        relevant_items_by_user[user_id] = set(gains.keys())
        for _, item_id, rating, _ in test_entries:
            ratings_by_pair[(user_id, item_id)] = rating

    return gains_by_user, relevant_items_by_user, ratings_by_pair


def _dcg_at_k(items: List[int], gains: Dict[int, float], k: int = 10) -> float:
    total = 0.0
    for idx, item_id in enumerate(items[:k], start=1):
        total += gains.get(item_id, 0.0) / math.log2(idx + 1)
    return total


def _evaluate_warm_metrics(
    interaction_rows: List[Dict[str, str]],
    metadata_labels: Dict[int, bool],
    warm_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    for row in warm_rows:
        if metadata_labels.get(_parse_int(row["item_id"]), False):
            return {
                "passed": False,
                "reason": "warm predictions contain cold-start items",
                "ndcg_at_10": 0.0,
                "hit_rate_at_10": 0.0,
                "rmse": float("inf"),
            }

    gains_by_user, relevant_items_by_user, ratings_by_pair = _canonical_test_truth(interaction_rows)
    ranked = _group_ranked_rows(warm_rows, score_key="predicted_rating")

    ndcgs: List[float] = []
    hits: List[float] = []
    squared_errors: List[float] = []

    for user_id, gains in gains_by_user.items():
        ranked_items = [item_id for item_id, _, _, _ in ranked.get(user_id, [])]
        actual_dcg = _dcg_at_k(ranked_items, gains)
        ideal_items = [item_id for item_id, _ in sorted(gains.items(), key=lambda item: item[1], reverse=True)]
        ideal_dcg = _dcg_at_k(ideal_items, gains)
        ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        hits.append(1.0 if any(item_id in relevant_items_by_user[user_id] for item_id in ranked_items[:10]) else 0.0)

    for user_id, entries in ranked.items():
        for item_id, predicted_rating, _, _ in entries:
            pair = (user_id, item_id)
            if pair in ratings_by_pair:
                squared_errors.append((predicted_rating - ratings_by_pair[pair]) ** 2)

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else float("inf")
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    hit_rate = sum(hits) / len(hits) if hits else 0.0
    passed = (
        ndcg >= WARM_NDCG_THRESHOLD
        and hit_rate >= WARM_HITRATE_THRESHOLD
        and rmse <= WARM_RMSE_THRESHOLD
    )
    return {
        "passed": passed,
        "ndcg_at_10": ndcg,
        "hit_rate_at_10": hit_rate,
        "rmse": rmse,
        "overlap_count": len(squared_errors),
    }


def _evaluate_cold_metrics(
    metadata_labels: Dict[int, bool],
    candidate_rows: List[Dict[str, str]],
    reference_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    candidate_ranked = _group_ranked_rows(candidate_rows, score_key="predicted_score")
    reference_ranked = _group_ranked_rows(reference_rows, score_key="predicted_score")

    ndcgs: List[float] = []
    top_items: set = set()
    total_cold_items = sum(1 for is_cold in metadata_labels.values() if is_cold)

    for user_id, reference_entries in reference_ranked.items():
        gains = {item_id: score for item_id, score, _, _ in reference_entries}
        candidate_items = [item_id for item_id, _, _, _ in candidate_ranked.get(user_id, [])]
        actual_dcg = _dcg_at_k(candidate_items, gains)
        ideal_items = [item_id for item_id, _ in sorted(gains.items(), key=lambda item: item[1], reverse=True)]
        ideal_dcg = _dcg_at_k(ideal_items, gains)
        ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        for item_id in candidate_items[:10]:
            if metadata_labels.get(item_id, False):
                top_items.add(item_id)

    coverage = len(top_items) / total_cold_items if total_cold_items else 0.0
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    passed = ndcg >= COLD_NDCG_THRESHOLD and coverage >= COLD_COVERAGE_THRESHOLD
    return {
        "passed": passed,
        "ndcg_at_10": ndcg,
        "coverage_at_10": coverage,
        "unique_cold_items_in_top10": len(top_items),
    }


def _evaluate_hybrid_integration(
    metadata_labels: Dict[int, bool],
    combined_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    seen_types = set()
    grouped: Dict[int, List[Tuple[int, float, int, str]]] = defaultdict(list)

    for row in combined_rows:
        user_id = _parse_int(row["user_id"])
        item_id = _parse_int(row["item_id"])
        rank = _parse_int(row["rank"], default=10**9)
        score = _parse_float(row["predicted_score"])
        item_type = str(row.get("item_type", "")).strip().lower()
        expected_type = "cold" if metadata_labels.get(item_id, False) else "warm"
        if item_type != expected_type:
            return {"passed": False, "reason": f"item_type mismatch for item_id={item_id}"}
        seen_types.add(item_type)
        grouped[user_id].append((item_id, score, rank, item_type))

    if seen_types != {"warm", "cold"}:
        return {"passed": False, "reason": "combined predictions do not include both warm and cold items"}

    for user_id, entries in grouped.items():
        entries.sort(key=lambda item: item[2])
        previous_score = None
        previous_rank = None
        for _, score, rank, _ in entries:
            if previous_rank is not None and rank < previous_rank:
                return {"passed": False, "reason": f"rank order regressed for user_id={user_id}"}
            if previous_score is not None and score > previous_score + 1e-12:
                return {"passed": False, "reason": f"scores not monotonically non-increasing for user_id={user_id}"}
            previous_rank = rank
            previous_score = score

    return {"passed": True, "reason": "combined rankings are structurally valid"}


def _evaluate_model_config(model_config: Any) -> Dict[str, Any]:
    if not isinstance(model_config, dict):
        return {"passed": False, "reason": "model_config.json is not a JSON object"}

    warm_model = model_config.get("warm_model")
    cold_model = model_config.get("cold_model")
    hybrid = model_config.get("hybrid")
    data_split = model_config.get("data_split")

    if not isinstance(warm_model, dict) or not str(warm_model.get("type", "")).strip():
        return {"passed": False, "reason": "missing warm_model.type"}
    if not isinstance(cold_model, dict) or not str(cold_model.get("type", "")).strip():
        return {"passed": False, "reason": "missing cold_model.type"}
    if not isinstance(hybrid, dict):
        return {"passed": False, "reason": "missing hybrid object"}
    if not isinstance(data_split, dict):
        return {"passed": False, "reason": "missing data_split object"}

    has_hybrid_method = any(str(hybrid.get(key, "")).strip() for key in ["combination", "method", "normalization"])
    if not has_hybrid_method:
        return {"passed": False, "reason": "missing hybrid method/combination"}

    dimensionality_keys = ["n_factors", "latent_factors", "embedding_dim", "n_components", "dim", "rank"]
    has_dimensionality = any(key in warm_model or key in cold_model for key in dimensionality_keys)
    if not has_dimensionality:
        return {"passed": False, "reason": "missing dimensionality / latent-factor parameter"}

    train_ratio = _parse_float(data_split.get("train_ratio"), default=-1.0)
    val_ratio = _parse_float(data_split.get("val_ratio"), default=-1.0)
    test_ratio = _parse_float(data_split.get("test_ratio"), default=-1.0)
    method = str(data_split.get("method", "")).lower()
    if "temporal" not in method:
        return {"passed": False, "reason": "data_split.method is not temporal"}
    if not (
        abs(train_ratio - TRAIN_RATIO) <= 1e-6
        and abs(val_ratio - VAL_RATIO) <= 1e-6
        and abs(test_ratio - TEST_RATIO) <= 1e-6
    ):
        return {"passed": False, "reason": "data_split ratios do not match 0.7 / 0.1 / 0.2"}

    return {"passed": True, "reason": "model_config.json is complete"}


def _evaluate_split_contract(
    model_config: Any,
    metadata_labels: Dict[int, bool],
    warm_rows: List[Dict[str, str]],
    evaluation_report: Any,
    warm_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    model_result = _evaluate_model_config(model_config)
    if not model_result["passed"]:
        return {"passed": False, "reason": model_result["reason"]}

    if any(metadata_labels.get(_parse_int(row["item_id"]), False) for row in warm_rows):
        return {"passed": False, "reason": "warm predictions include cold items"}

    if not isinstance(evaluation_report, dict):
        return {"passed": False, "reason": "evaluation_report.json is not a JSON object"}

    warm_report = evaluation_report.get("warm_items")
    if not isinstance(warm_report, dict):
        return {"passed": False, "reason": "evaluation_report.json missing warm_items"}

    comparisons = [
        ("NDCG@10", _parse_float(warm_report.get("NDCG@10"), default=float("nan")), _parse_float(warm_metrics.get("ndcg_at_10"), default=float("nan")), 0.01),
        ("HitRate@10", _parse_float(warm_report.get("HitRate@10"), default=float("nan")), _parse_float(warm_metrics.get("hit_rate_at_10"), default=float("nan")), 0.05),
        ("RMSE", _parse_float(warm_report.get("RMSE"), default=float("nan")), _parse_float(warm_metrics.get("rmse"), default=float("nan")), 0.1),
    ]

    for metric_name, reported_value, recomputed_value, tolerance in comparisons:
        if not math.isfinite(reported_value):
            return {"passed": False, "reason": f"evaluation_report.json missing warm_items.{metric_name}"}
        if not math.isfinite(recomputed_value):
            return {"passed": False, "reason": f"canonical recomputation for warm_items.{metric_name} is not finite"}
        if abs(reported_value - recomputed_value) > tolerance:
            return {
                "passed": False,
                "reason": f"evaluation_report.json warm_items.{metric_name} is inconsistent with canonical recomputation",
                "reported_value": reported_value,
                "recomputed_value": recomputed_value,
            }

    return {
        "passed": True,
        "reason": "warm-item report metrics align with the documented canonical split",
    }


def _check_output_completeness(
    parsed_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    missing = [name for name in REQUIRED_OUTPUT_FILES if name not in parsed_outputs]
    if missing:
        return {"passed": False, "reason": "missing output files: " + ", ".join(missing)}

    if parsed_outputs["predictions_warm.csv"]["columns"] != EXPECTED_WARM_COLUMNS:
        return {"passed": False, "reason": "predictions_warm.csv columns mismatch"}
    if parsed_outputs["predictions_cold.csv"]["columns"] != EXPECTED_COLD_COLUMNS:
        return {"passed": False, "reason": "predictions_cold.csv columns mismatch"}
    if parsed_outputs["predictions_combined.csv"]["columns"] != EXPECTED_COMBINED_COLUMNS:
        return {"passed": False, "reason": "predictions_combined.csv columns mismatch"}

    report = parsed_outputs["evaluation_report.json"]["data"]
    if not isinstance(report, dict) or set(report.keys()) != {"warm_items", "cold_items", "combined"}:
        return {"passed": False, "reason": "evaluation_report.json sections mismatch"}

    return {"passed": True, "reason": "all required files and schemas are present"}


def _parse_output_payloads(output_payloads: Dict[str, bytes]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for filename, payload in output_payloads.items():
        if filename.endswith(".csv"):
            columns, rows = _decode_csv_bytes(payload)
            parsed[filename] = {"columns": columns, "rows": rows}
        elif filename.endswith(".json"):
            parsed[filename] = {"data": _json_loads(payload)}
    return parsed


def score_submission(
    *,
    interactions_payload: bytes,
    item_metadata_payload: bytes,
    reference_cold_payload: bytes,
    output_payloads: Dict[str, bytes],
) -> Dict[str, Any]:
    parsed_outputs = _parse_output_payloads(output_payloads)
    completeness = _check_output_completeness(parsed_outputs)

    metadata_columns, metadata_rows = _decode_csv_bytes(item_metadata_payload)
    if "item_id" not in metadata_columns or "is_cold_start" not in metadata_columns:
        return {"score": 0.0, "criteria": {"completeness": completeness, "fatal": {"passed": False, "reason": "item_metadata.csv missing required columns"}}}
    metadata_labels = _build_metadata_labels(metadata_rows)

    interaction_columns, interaction_rows = _decode_csv_bytes(interactions_payload)
    if not {"user_id", "item_id", "rating", "watch_percentage", "timestamp"}.issubset(set(interaction_columns)):
        return {"score": 0.0, "criteria": {"completeness": completeness, "fatal": {"passed": False, "reason": "interactions.csv missing required columns"}}}

    warm_rows = parsed_outputs.get("predictions_warm.csv", {}).get("rows", [])
    cold_rows = parsed_outputs.get("predictions_cold.csv", {}).get("rows", [])
    combined_rows = parsed_outputs.get("predictions_combined.csv", {}).get("rows", [])
    evaluation_report = parsed_outputs.get("evaluation_report.json", {}).get("data", {})
    model_config = parsed_outputs.get("model_config.json", {}).get("data", {})
    _, reference_cold_rows = _decode_csv_bytes(reference_cold_payload)

    warm_metrics = _evaluate_warm_metrics(interaction_rows, metadata_labels, warm_rows)
    cold_metrics = _evaluate_cold_metrics(metadata_labels, cold_rows, reference_cold_rows)
    hybrid = _evaluate_hybrid_integration(metadata_labels, combined_rows)
    model_docs = _evaluate_model_config(model_config)
    split_contract = _evaluate_split_contract(
        model_config,
        metadata_labels,
        warm_rows,
        evaluation_report,
        warm_metrics,
    )

    criteria = {
        "output_completeness": completeness,
        "data_split_correctness": split_contract,
        "warm_metrics": warm_metrics,
        "cold_metrics": cold_metrics,
        "hybrid_integration": hybrid,
        "model_documentation": model_docs,
    }
    score = sum(1.0 for item in criteria.values() if item.get("passed")) / 6.0
    return {"score": score, "criteria": criteria}


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        interactions_payload = await _read_remote_bytes(session, meta["interactions_file"])
        item_metadata_payload = await _read_remote_bytes(session, meta["item_metadata_file"])
        reference_cold_payload = await _read_remote_bytes(session, f"{meta['reference_dir']}/predictions_cold.csv")
    except Exception as exc:
        logger.error("failed to read staged evaluator inputs: %s", exc)
        return [0.0]

    output_payloads: Dict[str, bytes] = {}
    for filename in REQUIRED_OUTPUT_FILES:
        remote_path = f"{meta['remote_output_dir']}/{filename}"
        try:
            output_payloads[filename] = await _read_remote_bytes(session, remote_path)
        except Exception as exc:
            logger.warning("missing candidate output %s: %s", remote_path, exc)

    try:
        result = score_submission(
            interactions_payload=interactions_payload,
            item_metadata_payload=item_metadata_payload,
            reference_cold_payload=reference_cold_payload,
            output_payloads=output_payloads,
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary
        logger.exception("recsys evaluator failed: %s", exc)
        return [0.0]

    logger.info("recsys evaluator result: %s", json.dumps(result)[:3000])
    return [float(result["score"])]
