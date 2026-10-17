"""Scoring helpers for merfish_image_decoding_segmentation_1.

Evaluator-only. Runs locally (in the agenthle `uv` env) against agent-produced
output bytes and the hidden reference CSV. No VM-side execution required.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

logger = logging.getLogger(__name__)


REQUIRED_DECODED_COLUMNS = ["gene", "x", "y", "is_exact", "total_magnitude"]
REQUIRED_METRIC_KEYS = [
    "total_decoded_transcripts",
    "blank_rate",
    "exact_match_fraction",
    "n_cells",
    "assigned_fraction",
    "mean_transcripts_per_cell",
]
SEGMENTATION_SHAPE = (2048, 2048)
REFERENCE_MIN_PX = 40.0
REFERENCE_MAX_PX = 2008.0

CORR_FULL = 0.90
CORR_PARTIAL = 0.70
CORR_MIN_FOR_CREDIT = 0.50
TOTAL_COUNT_TOL = 0.20
BLANK_RATE_FULL = 0.05
BLANK_RATE_PARTIAL = 0.07
SPATIAL_MATCH_PX = 3.0
SPATIAL_FULL = 0.80
SPATIAL_PARTIAL = 0.60
NUCLEUS_COUNT_MIN = 10
NUCLEUS_COUNT_MAX = 60
CELL_COUNT_MIN = NUCLEUS_COUNT_MIN
CELL_COUNT_MAX = NUCLEUS_COUNT_MAX
ASSIGNED_MIN = 0.10
MEAN_PER_CELL_MIN = 50
MEAN_PER_CELL_MAX = 1000

WEIGHT_PEARSON = 0.30
WEIGHT_TOTAL = 0.10
WEIGHT_BLANK = 0.10
WEIGHT_SPATIAL = 0.15
WEIGHT_CELLCOUNT = 0.10
WEIGHT_ASSIGNED = 0.10
WEIGHT_MEAN = 0.05
WEIGHT_CONSISTENCY = 0.10

DECODING_CREDIT_COMPONENTS = ("pearson", "total_count", "blank_rate", "spatial")


@dataclass
class ComponentScore:
    score: float = 0.0
    detail: str = ""


@dataclass
class ScoreResult:
    score: float = 0.0
    hard_failed: bool = False
    hard_fail_reason: str = ""
    pearson_r: float | None = None
    spatial_concordance: float | None = None
    components: dict[str, ComponentScore] = field(default_factory=dict)
    metrics_agent: dict[str, Any] = field(default_factory=dict)
    structural: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["components"] = {k: asdict(v) for k, v in self.components.items()}
        return out


def _read_csv(payload: bytes, *, label: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


def _read_segmentation(payload: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(payload))
    img.load()
    return np.array(img)


def _read_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def _integer_values(frame: pd.DataFrame, *, minimum: int) -> np.ndarray:
    values = []
    for token in frame.to_numpy().flat:
        try:
            value = Decimal(token)
        except InvalidOperation as exc:
            raise ValueError("values must be finite integers") from exc
        if (
            not value.is_finite()
            or value < minimum
            or value > np.iinfo(np.int64).max
            or value != value.to_integral_value()
        ):
            raise ValueError(f"values must be integers in [{minimum}, {np.iinfo(np.int64).max}]")
        values.append(int(value))
    return np.array(values, dtype=np.int64).reshape(frame.shape)


def _is_integer_dtype(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def _real_gene_list(codebook: dict[str, Any]) -> list[str]:
    targets = [m["target"] for m in codebook["mappings"]]
    return [t for t in targets if not t.startswith("Blank-")]


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _reference_support_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["x"].between(REFERENCE_MIN_PX, REFERENCE_MAX_PX) & frame["y"].between(
        REFERENCE_MIN_PX, REFERENCE_MAX_PX
    )


def _spatial_concordance(
    agent: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_px: float,
) -> float:
    if agent.empty or reference.empty:
        return 0.0
    matched = 0
    total = 0
    for gene, grp_agent in agent.groupby("gene"):
        ref_grp = reference[reference["gene"] == gene]
        total += len(grp_agent)
        if ref_grp.empty:
            continue
        ref_xy = ref_grp[["x", "y"]].to_numpy(dtype=float)
        ag_xy = grp_agent[["x", "y"]].to_numpy(dtype=float)
        valid_agent = np.isfinite(ag_xy).all(axis=1)
        valid_reference = np.isfinite(ref_xy).all(axis=1)
        if not valid_agent.any() or not valid_reference.any():
            continue
        valid_agent_xy = ag_xy[valid_agent]
        valid_reference_xy = ref_xy[valid_reference]
        tree = cKDTree(valid_reference_xy)
        neighbors = tree.query_ball_point(valid_agent_xy, r=max_px)
        edge_counts = np.fromiter((len(indices) for indices in neighbors), dtype=np.int64)
        if not edge_counts.any():
            continue
        row_indices = np.repeat(np.arange(len(valid_agent_xy)), edge_counts)
        column_indices = np.concatenate(
            [np.asarray(indices, dtype=np.int64) for indices in neighbors if indices]
        )
        adjacency = csr_matrix(
            (np.ones(len(row_indices), dtype=np.int8), (row_indices, column_indices)),
            shape=(len(valid_agent_xy), len(valid_reference_xy)),
        )
        matches = maximum_bipartite_matching(adjacency, perm_type="row")
        matched += int(np.count_nonzero(matches >= 0))
    if total == 0:
        return 0.0
    return matched / total


def score(
    *,
    decoded_csv_bytes: bytes,
    segmentation_tiff_bytes: bytes,
    cell_by_gene_csv_bytes: bytes,
    quality_metrics_json_bytes: bytes,
    reference_csv_bytes: bytes,
    codebook_json_bytes: bytes,
) -> ScoreResult:
    result = ScoreResult()
    codebook = json.loads(codebook_json_bytes.decode("utf-8"))
    real_genes = _real_gene_list(codebook)

    try:
        agent_decoded = _read_csv(decoded_csv_bytes, label="decoded_transcripts.csv")
    except Exception as exc:
        result.hard_failed = True
        result.hard_fail_reason = f"decoded_transcripts.csv unreadable: {exc}"
        return result

    missing_cols = [c for c in REQUIRED_DECODED_COLUMNS if c not in agent_decoded.columns]
    if missing_cols:
        result.hard_failed = True
        result.hard_fail_reason = f"decoded_transcripts.csv missing columns: {missing_cols}"
        return result
    if len(agent_decoded) == 0:
        result.hard_failed = True
        result.hard_fail_reason = "decoded_transcripts.csv has zero rows"
        return result
    try:
        coordinates = agent_decoded[["x", "y"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        result.hard_failed = True
        result.hard_fail_reason = f"decoded_transcripts.csv coordinates are not numeric: {exc}"
        return result
    if not np.isfinite(coordinates).all():
        result.hard_failed = True
        result.hard_fail_reason = "decoded_transcripts.csv coordinates must be finite"
        return result

    try:
        seg = _read_segmentation(segmentation_tiff_bytes)
    except Exception as exc:
        result.hard_failed = True
        result.hard_fail_reason = f"segmentation.tiff unreadable: {exc}"
        return result
    if seg.shape != SEGMENTATION_SHAPE or not _is_integer_dtype(seg):
        result.hard_failed = True
        result.hard_fail_reason = (
            f"segmentation.tiff must be integer {SEGMENTATION_SHAPE}, got "
            f"shape={seg.shape} dtype={seg.dtype}"
        )
        return result

    try:
        cbg = pd.read_csv(io.BytesIO(cell_by_gene_csv_bytes), dtype=str, keep_default_na=False)
    except Exception as exc:
        result.hard_failed = True
        result.hard_fail_reason = f"cell_by_gene.csv unreadable: {exc}"
        return result
    if list(cbg.columns) != ["cell_id", *real_genes]:
        result.hard_failed = True
        result.hard_fail_reason = (
            "cell_by_gene.csv columns must be ['cell_id', *130 real genes in codebook order]"
        )
        return result

    try:
        metrics = _read_json(quality_metrics_json_bytes)
    except Exception as exc:
        result.hard_failed = True
        result.hard_fail_reason = f"quality_metrics.json unreadable: {exc}"
        return result
    missing_keys = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing_keys:
        result.hard_failed = True
        result.hard_fail_reason = f"quality_metrics.json missing keys: {missing_keys}"
        return result

    reference = _read_csv(reference_csv_bytes, label="reference benchmark_results.csv")
    result.metrics_agent = {k: metrics[k] for k in REQUIRED_METRIC_KEYS}

    real_gene_mask = agent_decoded["gene"].isin(real_genes)
    agent_support = _reference_support_mask(agent_decoded)
    agent_real = agent_decoded[real_gene_mask & agent_support]
    ref_real = reference[reference["gene"].isin(real_genes) & _reference_support_mask(reference)]
    reference_total_real = len(ref_real)
    if reference_total_real == 0:
        raise ValueError("reference benchmark_results.csv contains no real-gene transcripts")

    agent_counts = (
        agent_real["gene"].value_counts().reindex(real_genes, fill_value=0).to_numpy(dtype=float)
    )
    ref_counts = (
        ref_real["gene"].value_counts().reindex(real_genes, fill_value=0).to_numpy(dtype=float)
    )
    pearson_r = _pearson_r(agent_counts, ref_counts)
    result.pearson_r = pearson_r

    total_decoded = int(len(agent_decoded))
    blank_mask = agent_decoded["gene"].astype(str).str.startswith("Blank-")
    blank_rate = float(blank_mask.mean()) if total_decoded > 0 else 0.0

    spatial = _spatial_concordance(agent_real, ref_real, max_px=SPATIAL_MATCH_PX)
    result.spatial_concordance = spatial

    n_cells_seg = int(seg.max())
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels > 0]
    n_cells_labels = int(unique_labels.size)
    n_cells = min(n_cells_seg, n_cells_labels) if n_cells_labels > 0 else 0

    xs = agent_decoded["x"].to_numpy(dtype=float)
    ys = agent_decoded["y"].to_numpy(dtype=float)
    finite_coordinates = np.isfinite(xs) & np.isfinite(ys)
    in_bounds = (
        finite_coordinates
        & (xs >= 0)
        & (xs < SEGMENTATION_SHAPE[1])
        & (ys >= 0)
        & (ys < SEGMENTATION_SHAPE[0])
    )
    row_cell_labels = np.zeros(len(agent_decoded), dtype=np.int64)
    if in_bounds.any():
        xi = np.rint(xs[in_bounds]).astype(int)
        yi = np.rint(ys[in_bounds]).astype(int)
        rounded_in_bounds = (
            (xi >= 0) & (xi < SEGMENTATION_SHAPE[1]) & (yi >= 0) & (yi < SEGMENTATION_SHAPE[0])
        )
        valid_indices = np.flatnonzero(in_bounds)[rounded_in_bounds]
        row_cell_labels[valid_indices] = seg[yi[rounded_in_bounds], xi[rounded_in_bounds]].astype(
            np.int64
        )

    real_row_mask = agent_decoded["gene"].isin(real_genes).to_numpy()
    assigned_real = real_row_mask & (row_cell_labels > 0)
    assigned_fraction = float(assigned_real.sum() / max(real_row_mask.sum(), 1))

    try:
        mean_per_cell = (
            float(cbg.iloc[:, 1:].to_numpy(dtype=float).sum() / n_cells) if n_cells > 0 else 0.0
        )
    except (ValueError, OverflowError):
        mean_per_cell = 0.0
    if not np.isfinite(mean_per_cell):
        mean_per_cell = 0.0

    result.structural = {
        "n_cells_segmentation": n_cells,
        "assigned_fraction": assigned_fraction,
        "mean_transcripts_per_cell": mean_per_cell,
        "cbg_rows": int(len(cbg)),
        "invalid_coordinate_rows": int((~in_bounds).sum()),
        "decoded_real_in_reference_support": len(agent_real),
        "decoded_real_outside_reference_support": int((real_gene_mask & ~agent_support).sum()),
        "reference_support_min_px": REFERENCE_MIN_PX,
        "reference_support_max_px": REFERENCE_MAX_PX,
    }

    components: dict[str, ComponentScore] = {}

    decoding_credit_ok = pearson_r >= CORR_MIN_FOR_CREDIT

    if not decoding_credit_ok:
        pearson_component = ComponentScore(
            0.0, f"pearson={pearson_r:.4f} below min_for_credit={CORR_MIN_FOR_CREDIT:.2f}"
        )
    elif pearson_r >= CORR_FULL:
        pearson_component = ComponentScore(
            WEIGHT_PEARSON, f"pearson={pearson_r:.4f} >= {CORR_FULL}"
        )
    elif pearson_r >= CORR_PARTIAL:
        pearson_component = ComponentScore(
            WEIGHT_PEARSON * 0.5, f"pearson={pearson_r:.4f} in [{CORR_PARTIAL},{CORR_FULL})"
        )
    else:
        pearson_component = ComponentScore(
            WEIGHT_PEARSON * 0.2,
            f"pearson={pearson_r:.4f} in [{CORR_MIN_FOR_CREDIT},{CORR_PARTIAL})",
        )
    components["pearson"] = pearson_component

    total_real = len(agent_real)
    tot_rel = abs(total_real - reference_total_real) / reference_total_real
    total_ok = tot_rel <= TOTAL_COUNT_TOL
    components["total_count"] = ComponentScore(
        WEIGHT_TOTAL if total_ok and decoding_credit_ok else 0.0,
        f"total_real={total_real} reference_real={reference_total_real} "
        f"rel_err={tot_rel:.3f} tol={TOTAL_COUNT_TOL} "
        f"reference_support=[{REFERENCE_MIN_PX:g},{REFERENCE_MAX_PX:g}] on both axes",
    )

    if not decoding_credit_ok:
        components["blank_rate"] = ComponentScore(0.0, "skipped: decoding credit gate failed")
    elif blank_rate <= BLANK_RATE_FULL:
        components["blank_rate"] = ComponentScore(
            WEIGHT_BLANK, f"blank_rate={blank_rate:.4f} <= {BLANK_RATE_FULL}"
        )
    elif blank_rate <= BLANK_RATE_PARTIAL:
        components["blank_rate"] = ComponentScore(
            WEIGHT_BLANK * 0.5,
            f"blank_rate={blank_rate:.4f} in ({BLANK_RATE_FULL},{BLANK_RATE_PARTIAL}]",
        )
    else:
        components["blank_rate"] = ComponentScore(
            0.0, f"blank_rate={blank_rate:.4f} > {BLANK_RATE_PARTIAL}"
        )

    if not decoding_credit_ok:
        components["spatial"] = ComponentScore(0.0, "skipped: decoding credit gate failed")
    elif spatial >= SPATIAL_FULL:
        components["spatial"] = ComponentScore(
            WEIGHT_SPATIAL, f"spatial={spatial:.3f} >= {SPATIAL_FULL}"
        )
    elif spatial >= SPATIAL_PARTIAL:
        components["spatial"] = ComponentScore(
            WEIGHT_SPATIAL * 0.5, f"spatial={spatial:.3f} in [{SPATIAL_PARTIAL},{SPATIAL_FULL})"
        )
    else:
        components["spatial"] = ComponentScore(0.0, f"spatial={spatial:.3f} < {SPATIAL_PARTIAL}")

    cell_count_ok = CELL_COUNT_MIN <= n_cells <= CELL_COUNT_MAX
    components["cell_count"] = ComponentScore(
        WEIGHT_CELLCOUNT if cell_count_ok else 0.0,
        f"nuclei_seg={n_cells} range=[{NUCLEUS_COUNT_MIN},{NUCLEUS_COUNT_MAX}]",
    )

    assigned_ok = assigned_fraction >= ASSIGNED_MIN and cell_count_ok
    components["assigned_fraction"] = ComponentScore(
        WEIGHT_ASSIGNED if assigned_ok else 0.0,
        f"assigned_to_nuclei={assigned_fraction:.3f} min={ASSIGNED_MIN}",
    )

    mean_ok = MEAN_PER_CELL_MIN <= mean_per_cell <= MEAN_PER_CELL_MAX and cell_count_ok
    components["mean_per_cell"] = ComponentScore(
        WEIGHT_MEAN if mean_ok else 0.0,
        f"mean_per_cell={mean_per_cell:.2f} range=[{MEAN_PER_CELL_MIN},{MEAN_PER_CELL_MAX}]",
    )

    consistency_ok = False
    consistency_reason = ""
    if len(cbg) != n_cells:
        consistency_reason = f"cell_by_gene.csv rows={len(cbg)} but segmentation n_cells={n_cells}"
    elif n_cells == 0:
        consistency_reason = "n_cells=0; cannot verify matrix consistency"
    else:
        try:
            cell_ids = _integer_values(cbg[["cell_id"]], minimum=1).ravel()
            cbg_values = _integer_values(cbg.iloc[:, 1:], minimum=0)
        except ValueError as exc:
            consistency_reason = f"cell_by_gene.csv invalid cell_id or counts: {exc}"
        else:
            if not np.array_equal(np.sort(cell_ids), unique_labels):
                consistency_reason = (
                    "cell_id must contain each positive segmentation label exactly once"
                )
            else:
                expected = np.zeros((n_cells, len(real_genes)), dtype=np.int64)
                cell_indices = np.searchsorted(unique_labels, row_cell_labels[assigned_real])
                gene_indices = pd.Index(real_genes).get_indexer(
                    agent_decoded.loc[assigned_real, "gene"]
                )
                np.add.at(expected, (cell_indices, gene_indices), 1)
                consistency_ok = np.array_equal(cbg_values[np.argsort(cell_ids)], expected)
                consistency_reason = (
                    "per-cell/per-gene counts tie out to assigned transcripts"
                    if consistency_ok
                    else "per-cell/per-gene counts disagree with assigned transcripts"
                )
    components["matrix_consistency"] = ComponentScore(
        WEIGHT_CONSISTENCY if consistency_ok else 0.0,
        consistency_reason,
    )

    total = sum(c.score for c in components.values())
    result.components = components
    result.score = round(float(total), 6)
    return result


__all__ = ["score", "ScoreResult", "ComponentScore"]
