"""AgentHLE task: vegetation-precipitation spatial vs temporal sensitivity."""

import json as json_mod
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import (
    EvaluationContext,
    llm_multimodal_text,
    resolve_llm_judge_model,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

# ─── Expected conclusions (used by LLM judge) ───────────────────────────────

EXPECTED_CONCLUSIONS = [
    (
        "Vegetation responds more strongly to precipitation across space than "
        "over time, adding uncertainty in ecosystem projections under climate change."
    ),
    (
        "The space-time discrepancy is largely due to differences in seasonal "
        "water variability, rooting depth, and soil properties."
    ),
    (
        "Vegetation sensitivity to soil moisture converges across CONUS drylands, "
        "underscoring soil hydrology's role in ecosystem response."
    ),
]

# Water-variability feature keywords (for Stage 5 top-3 check)
WATER_VARIABILITY_KEYWORDS = [
    "seasonal variation of ssm",
    "seasonal variation of ppt",
    "seasonal variation of precipitation",
    "interannual variation of ssm",
    "interannual variation of soil moisture",
    "seasonal variability of ssm",
    "seasonal variability of ppt",
    "seasonal variability of precipitation",
    "interannual variability of ssm",
    "std of monthly ssm",
    "std of monthly soil moisture",
    "std of monthly ppt",
    "std of monthly precipitation",
    "std of annual ssm",
    "std of annual soil moisture",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  TaskConfig
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskConfig(GeneralTaskConfig):
    """Task configuration for vegetation–precipitation sensitivity analysis."""

    VARIANT_NAME: str = "base"
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "vegetation_precip_spatial_temporal"
    VARIANT_NAME: str = "base"
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")

    # Spatial domain: CONUS bounding box
    CONUS_WEST: float = -125.0
    CONUS_EAST: float = -66.5
    CONUS_SOUTH: float = 24.5
    CONUS_NORTH: float = 49.5

    # Required biome categories in output CSVs (matched flexibly via keywords)
    REQUIRED_BIOME_KEYWORDS: tuple = ("all", "grass", "savanna", "shrub")

    # Evaluation thresholds
    PPT_MIN_RELATIVE_DIFF: float = 0.40
    SSM_MAX_RELATIVE_DIFF: float = 0.20
    MIN_FEATURE_IMPORTANCE_R2: float = 0.40

    # Authentication (same pattern as TerraClimate task)
    GEE_CREDENTIALS_PATH: str = os.environ.get("GEE_CREDENTIALS_PATH", "")
    GEE_SERVICE_ACCOUNT: str = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "")
    GEE_ACCOUNT_EMAIL: str = os.environ.get("GEE_ACCOUNT_EMAIL", "agenthle.sv@gmail.com")

    @property
    def task_dir(self) -> str:
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_test_pos_dir(self) -> str:
        return rf"{self.task_dir}\output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return rf"{self.task_dir}\output_test_neg"

    @property
    def gcs_root(self) -> str:
        return f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def task_description(self) -> str:
        return f"""You are an ecological researcher investigating a well-known puzzle:
why does vegetation respond much more strongly to precipitation across space
than over time?

Research Questions:
  RQ1: Why does vegetation respond much more strongly to precipitation across
       space than over time?
  RQ2: What factors drive the space–time discrepancy in vegetation sensitivity
       to precipitation?
  RQ3: How does the pattern change if we use soil moisture instead of
       precipitation as the water indicator?

Study area: Continental United States (CONUS) drylands — Grasslands, Savannas,
and Shrublands as classified by IGBP land-cover types.
Use the following fixed data products for the benchmark:
  - vegetation variable: MODIS MOD13A1 NDVI
  - precipitation product: PRISM precipitation
  - soil-moisture product: SMAP L4 root-zone soil moisture
  - biome mask: MODIS MCD12Q1 IGBP land-cover classification

Time period: use the fixed overlapping analysis window 2015-2023 inclusive.
Do not extend the analysis into 2024 even if some products expose newer data.

Required supporting covariates for feature importance:
  - TerraClimate PET
  - TerraClimate aridity index
  - root-zone water storage capacity
  - OpenLandMap soil bulk density
  - rainfall characteristics from Bassiouni 2020
  - seasonal/interannual variability metrics derived from the fixed PRISM and
    SMAP L4 products above

Do not substitute alternate primary vegetation, precipitation, or
soil-moisture products.

Canonical Python entry point:
  Run any analysis code through the task-local wrapper at
  `{self.software_dir}\\python.bat`. It forwards to the pinned
  Windows Python 3.12 interpreter provisioned by the benchmark admin.
  Do not rely on a different Python installation or a portable runtime
  under `software/`.

Authentication (if you use Google Earth Engine):
  Prefer benchmark-provided environment variables over browser login:
  - `GEE_SERVICE_ACCOUNT`
  - `GEE_CREDENTIALS_PATH`
  - `GEE_ACCOUNT_EMAIL` (optional)

Required outputs (save all to the output folder):

1. ndvi_ppt_sensitivity.csv
   Spatial and temporal sensitivity of NDVI to precipitation, by biome.
   Columns: biome, spatial_slope, spatial_r2, mean_temporal_slope,
            temporal_min, temporal_max, temporal_r2_mean,
            difference, relative_difference, N
   Rows: one per biome (e.g., "All", "Grasslands", "Savannas", "Shrublands"
         or equivalent IGBP class names)
   Where:
     - spatial_slope: slope of cross-sectional regression (long-term mean NDVI
       vs long-term mean annual precipitation across grid cells)
     - mean_temporal_slope: mean of per-pixel temporal regression slopes
       (annual NDVI anomaly vs annual precipitation anomaly within each pixel)
     - difference = spatial_slope - mean_temporal_slope
     - relative_difference = difference / spatial_slope

2. ndvi_ssm_sensitivity.csv
   Same structure as above, but using soil moisture instead of precipitation.

3. feature_importance.csv
   Train a random forest model to predict D_ppt (= spatial_slope -
   mean_temporal_slope per pixel) using the fixed supporting covariates named
   above: seasonal/interannual variability of precipitation and soil moisture,
   rainfall intensity/frequency, soil properties (bulk density), rooting
   depth, PET, and aridity index.
   Compute SHAP values for feature importance.
   Columns: feature, mean_shap_value, model_r2
   (model_r2 is the same value repeated in every row)

4. conclusions.json
   A JSON list of exactly 3 conclusion strings, one per research question,
   summarizing the key findings from the analysis.

Save all final outputs to: {self.remote_output_dir}
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "reference_gcs_prefix": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/reference"
                ),
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "gcs_root": self.gcs_root,
                "conus_bbox": [
                    self.CONUS_WEST,
                    self.CONUS_SOUTH,
                    self.CONUS_EAST,
                    self.CONUS_NORTH,
                ],
                "required_biome_keywords": list(self.REQUIRED_BIOME_KEYWORDS),
                "ppt_min_relative_diff": self.PPT_MIN_RELATIVE_DIFF,
                "ssm_max_relative_diff": self.SSM_MAX_RELATIVE_DIFF,
                "min_feature_importance_r2": self.MIN_FEATURE_IMPORTANCE_R2,
                "gee_credentials_path": self.GEE_CREDENTIALS_PATH,
                "gee_service_account": self.GEE_SERVICE_ACCOUNT,
                "gee_account_email": self.GEE_ACCOUNT_EMAIL,
                "gcs_bucket": self.GCS_BUCKET,
            }
        )
        return metadata


config = TaskConfig()


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 1: load()
# ═══════════════════════════════════════════════════════════════════════════════


@cb.tasks_config(split="train")
def load():
    """Define the vegetation–precipitation sensitivity task."""
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 2: start()
# ═══════════════════════════════════════════════════════════════════════════════


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 3: evaluate()  — 6-stage, 8-point scoring
# ═══════════════════════════════════════════════════════════════════════════════


def _check_stage1_download(tmpdir: str, file_map: dict[str, str]) -> tuple[float, list[dict]]:
    """Stage 1: Validate raw data download (ndvi_raw.nc, ppt_raw.nc, ssm_raw.nc).

    Returns (score, checks) where score is 1.0 if all 3 files are valid NetCDF.
    """
    import xarray as xr

    raw_files = ["ndvi_raw.nc", "ppt_raw.nc", "ssm_raw.nc"]
    checks = []
    all_valid = True

    for fname in raw_files:
        if fname not in file_map:
            checks.append(
                {
                    "check": f"stage1_{fname}_exists",
                    "passed": False,
                    "message": f"Missing: {fname}",
                }
            )
            all_valid = False
            continue

        try:
            with xr.open_dataset(file_map[fname]) as ds:
                has_vars = len(ds.data_vars) >= 1
                has_time = "time" in ds.dims
                ok = has_vars and has_time
                checks.append(
                    {
                        "check": f"stage1_{fname}_valid",
                        "passed": ok,
                        "message": (
                            f"{fname}: {len(ds.data_vars)} vars, dims={list(ds.dims)}"
                            if ok
                            else f"{fname}: need >=1 var and time dim, got vars={list(ds.data_vars)}, dims={list(ds.dims)}"
                        ),
                    }
                )
                if not ok:
                    all_valid = False
        except Exception as e:
            checks.append(
                {
                    "check": f"stage1_{fname}_valid",
                    "passed": False,
                    "message": f"Cannot open {fname}: {e}",
                }
            )
            all_valid = False

    return (1.0 if all_valid else 0.0, checks)


def _check_stage2_regrid(tmpdir: str, file_map: dict[str, str]) -> tuple[float, list[dict]]:
    """Stage 2: Validate regridded merged file (merged.nc).

    All variables must share the same spatial dimensions.
    """
    import xarray as xr

    checks = []

    if "merged.nc" not in file_map:
        checks.append(
            {
                "check": "stage2_merged_exists",
                "passed": False,
                "message": "Missing: merged.nc",
            }
        )
        return (0.0, checks)

    try:
        with xr.open_dataset(file_map["merged.nc"]) as ds:
            var_names = list(ds.data_vars)
            has_enough_vars = len(var_names) >= 3
            checks.append(
                {
                    "check": "stage2_merged_vars",
                    "passed": has_enough_vars,
                    "message": (
                        f"Variables: {var_names}"
                        if has_enough_vars
                        else f"Need >=3 variables (NDVI, PPT, SSM), got {var_names}"
                    ),
                }
            )
            if not has_enough_vars:
                return (0.0, checks)

            spatial_dims_per_var = []
            for v in var_names:
                dims = set(ds[v].dims) - {"time"}
                spatial_dims_per_var.append(dims)

            all_same_grid = len(set(frozenset(d) for d in spatial_dims_per_var)) == 1
            first_var_shape = {d: ds.sizes[d] for d in spatial_dims_per_var[0]}
            checks.append(
                {
                    "check": "stage2_shared_grid",
                    "passed": all_same_grid,
                    "message": (
                        f"All variables share spatial grid: {first_var_shape}"
                        if all_same_grid
                        else f"Variables have different spatial dims: {spatial_dims_per_var}"
                    ),
                }
            )
            return (1.0 if all_same_grid else 0.0, checks)

    except Exception as e:
        checks.append(
            {
                "check": "stage2_merged_open",
                "passed": False,
                "message": f"Cannot open merged.nc: {e}",
            }
        )
        return (0.0, checks)


NUMERIC_COLS_TO_COMPARE = [
    "spatial_slope",
    "spatial_r2",
    "mean_temporal_slope",
    "temporal_r2_mean",
    "difference",
    "relative_difference",
]

RELATIVE_TOLERANCE = 0.20  # 20% tolerance for numeric comparisons


def _find_biome_row(df, keyword: str):
    """Find the first row whose biome column contains the keyword (case-insensitive).

    Returns the row as a Series, or None if not found.
    """
    mask = df["biome"].str.lower().str.contains(keyword.lower(), na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None


def _compare_against_reference(
    agent_df,
    ref_df,
    stage_prefix: str,
    biome_keywords: list[str],
    rtol: float = RELATIVE_TOLERANCE,
) -> list[dict]:
    """Compare agent CSV values against reference CSV values per biome.

    Uses flexible biome matching and relative tolerance.
    """
    import numpy as np

    checks = []
    all_close = True

    for kw in biome_keywords:
        agent_row = _find_biome_row(agent_df, kw)
        ref_row = _find_biome_row(ref_df, kw)

        if agent_row is None or ref_row is None:
            continue

        mismatches = []
        for col in NUMERIC_COLS_TO_COMPARE:
            if col not in agent_df.columns or col not in ref_df.columns:
                continue
            agent_val = float(agent_row[col])
            ref_val = float(ref_row[col])
            if ref_val == 0:
                close = abs(agent_val) < 0.01
            else:
                close = np.isclose(agent_val, ref_val, rtol=rtol, atol=1e-6)
            if not close:
                mismatches.append(f"{col}: agent={agent_val:.4f} vs ref={ref_val:.4f}")

        ok = len(mismatches) == 0
        if not ok:
            all_close = False
        checks.append(
            {
                "check": f"{stage_prefix}_ref_match_{kw}",
                "passed": ok,
                "message": (
                    f"'{kw}' row matches reference (rtol={rtol})"
                    if ok
                    else f"'{kw}' row mismatches: {'; '.join(mismatches)}"
                ),
            }
        )

    return checks


def _check_sensitivity_csv(
    file_map: dict[str, str],
    fname: str,
    stage_prefix: str,
    required_biome_keywords: list[str],
    relative_diff_check: str,
    relative_diff_threshold: float,
    reference_fname: str | None = None,
    local_reference_dir: Path | None = None,
) -> tuple[float, list[dict]]:
    """Shared validator for ndvi_ppt_sensitivity.csv and ndvi_ssm_sensitivity.csv.

    Biome matching is flexible: each keyword (e.g., "grass") is matched against the
    biome column via case-insensitive substring search, so "Grasslands", "Grassland",
    "grasslands" all match "grass".

    relative_diff_check: "gt" (greater than threshold) or "lt" (less than threshold)
    reference_fname: if provided, also compare against reference CSV in local_reference_dir
    """
    import pandas as pd

    checks = []

    if fname not in file_map:
        checks.append(
            {
                "check": f"{stage_prefix}_exists",
                "passed": False,
                "message": f"Missing: {fname}",
            }
        )
        return (0.0, checks)

    try:
        df = pd.read_csv(file_map[fname])
    except Exception as e:
        checks.append(
            {
                "check": f"{stage_prefix}_readable",
                "passed": False,
                "message": f"Cannot read {fname}: {e}",
            }
        )
        return (0.0, checks)

    expected_cols = {
        "biome",
        "spatial_slope",
        "spatial_r2",
        "mean_temporal_slope",
        "temporal_min",
        "temporal_max",
        "temporal_r2_mean",
        "difference",
        "relative_difference",
        "N",
    }
    actual_cols = set(df.columns)
    has_cols = expected_cols.issubset(actual_cols)
    checks.append(
        {
            "check": f"{stage_prefix}_columns",
            "passed": has_cols,
            "message": (
                f"All 10 required columns present"
                if has_cols
                else f"Missing columns: {expected_cols - actual_cols}"
            ),
        }
    )
    if not has_cols:
        return (0.0, checks)

    biomes_in_csv = df["biome"].str.strip().tolist()
    missing_keywords = []
    for kw in required_biome_keywords:
        if _find_biome_row(df, kw) is None:
            missing_keywords.append(kw)
    has_biomes = len(missing_keywords) == 0
    checks.append(
        {
            "check": f"{stage_prefix}_biome_rows",
            "passed": has_biomes,
            "message": (
                f"Found all required biome categories in: {biomes_in_csv}"
                if has_biomes
                else f"Missing biome keywords {missing_keywords} in rows: {biomes_in_csv}"
            ),
        }
    )
    if not has_biomes:
        return (0.0, checks)

    all_row = _find_biome_row(df, "all")
    spatial_slope_positive = float(all_row["spatial_slope"]) > 0
    r2_valid = 0.0 <= float(all_row["spatial_r2"]) <= 1.0
    n_positive = int(all_row["N"]) > 0
    ranges_ok = spatial_slope_positive and r2_valid and n_positive
    checks.append(
        {
            "check": f"{stage_prefix}_value_ranges",
            "passed": ranges_ok,
            "message": (
                f"slope={all_row['spatial_slope']:.4f}>0, R2={all_row['spatial_r2']:.4f} in [0,1], N={all_row['N']}"
                if ranges_ok
                else f"Bad values: slope={all_row['spatial_slope']}, R2={all_row['spatial_r2']}, N={all_row['N']}"
            ),
        }
    )

    rel_diff = float(all_row["relative_difference"])
    if relative_diff_check == "gt":
        finding_ok = rel_diff > relative_diff_threshold
        finding_msg = (
            f"relative_difference={rel_diff:.4f} > {relative_diff_threshold} (large space-time gap)"
            if finding_ok
            else f"relative_difference={rel_diff:.4f} <= {relative_diff_threshold} (expected large gap)"
        )
    else:
        finding_ok = rel_diff < relative_diff_threshold
        finding_msg = (
            f"relative_difference={rel_diff:.4f} < {relative_diff_threshold} (convergence)"
            if finding_ok
            else f"relative_difference={rel_diff:.4f} >= {relative_diff_threshold} (expected convergence)"
        )
    checks.append(
        {
            "check": f"{stage_prefix}_scientific_finding",
            "passed": finding_ok,
            "message": finding_msg,
        }
    )

    if reference_fname and local_reference_dir is not None:
        ref_path = local_reference_dir / reference_fname
        if ref_path.exists():
            try:
                ref_df = pd.read_csv(ref_path)
                ref_checks = _compare_against_reference(
                    df,
                    ref_df,
                    stage_prefix,
                    required_biome_keywords,
                )
                checks.extend(ref_checks)
            except Exception as e:
                checks.append(
                    {
                        "check": f"{stage_prefix}_ref_load",
                        "passed": False,
                        "message": f"Cannot load reference {reference_fname}: {e}",
                    }
                )
        else:
            logger.warning(f"Reference file not found: {ref_path}")

    all_passed = all(c["passed"] for c in checks)
    return (1.0 if all_passed else 0.0, checks)


def _check_stage5_feature_importance(
    file_map: dict[str, str],
    min_r2: float,
    local_reference_dir: Path,
) -> tuple[float, list[dict]]:
    """Stage 5: Validate feature_importance.csv (SHAP-based RF results)."""
    import pandas as pd

    checks = []
    fname = "feature_importance.csv"

    if fname not in file_map:
        checks.append(
            {
                "check": "stage5_exists",
                "passed": False,
                "message": f"Missing: {fname}",
            }
        )
        return (0.0, checks)

    try:
        df = pd.read_csv(file_map[fname])
    except Exception as e:
        checks.append(
            {
                "check": "stage5_readable",
                "passed": False,
                "message": f"Cannot read {fname}: {e}",
            }
        )
        return (0.0, checks)

    required_cols = {"feature", "mean_shap_value", "model_r2"}
    has_cols = required_cols.issubset(set(df.columns))
    checks.append(
        {
            "check": "stage5_columns",
            "passed": has_cols,
            "message": (
                "Required columns present: feature, mean_shap_value, model_r2"
                if has_cols
                else f"Missing columns: {required_cols - set(df.columns)}"
            ),
        }
    )
    if not has_cols:
        return (0.0, checks)

    enough_features = len(df) >= 5
    checks.append(
        {
            "check": "stage5_feature_count",
            "passed": enough_features,
            "message": (
                f"{len(df)} features listed (>= 5)"
                if enough_features
                else f"Only {len(df)} features (need >= 5)"
            ),
        }
    )
    if not enough_features:
        return (0.0, checks)

    r2_value = float(df["model_r2"].iloc[0])
    r2_ok = r2_value > min_r2
    checks.append(
        {
            "check": "stage5_model_r2",
            "passed": r2_ok,
            "message": (
                f"model_r2={r2_value:.4f} > {min_r2}"
                if r2_ok
                else f"model_r2={r2_value:.4f} <= {min_r2}"
            ),
        }
    )

    df_sorted = df.sort_values("mean_shap_value", ascending=False)
    top3 = df_sorted.head(3)["feature"].str.lower().tolist()
    water_var_count = sum(
        1 for feat in top3 if any(kw in feat for kw in WATER_VARIABILITY_KEYWORDS)
    )
    top3_ok = water_var_count == 3
    checks.append(
        {
            "check": "stage5_top3_water_variability",
            "passed": top3_ok,
            "message": (
                f"Top 3 features are all water-variability-related: {top3}"
                if top3_ok
                else f"Expected top 3 to be water-variability features, got: {top3} ({water_var_count}/3 match)"
            ),
        }
    )

    ref_path = local_reference_dir / "feature_importance.csv"
    if ref_path.exists():
        try:
            ref_df = pd.read_csv(ref_path)
            ref_sorted = ref_df.sort_values("mean_shap_value", ascending=False)
            agent_ranking = df_sorted["feature"].str.lower().tolist()
            ref_ranking = ref_sorted["feature"].str.lower().tolist()
            top5_agent = agent_ranking[:5]
            top5_ref = set(ref_ranking[:5])
            overlap = sum(1 for f in top5_agent if f in top5_ref)
            ranking_ok = overlap >= 3
            checks.append(
                {
                    "check": "stage5_ref_ranking",
                    "passed": ranking_ok,
                    "message": (
                        f"Top-5 overlap with reference: {overlap}/5 (>= 3)"
                        if ranking_ok
                        else f"Top-5 overlap with reference: {overlap}/5 (< 3). Agent: {top5_agent}, Ref: {sorted(top5_ref)}"
                    ),
                }
            )
        except Exception as e:
            checks.append(
                {
                    "check": "stage5_ref_load",
                    "passed": False,
                    "message": f"Cannot load reference feature_importance.csv: {e}",
                }
            )
    else:
        logger.warning(f"Reference file not found: {ref_path}")

    all_passed = all(c["passed"] for c in checks)
    return (1.0 if all_passed else 0.0, checks)


async def _llm_judge_conclusion(
    agent_conclusion: str,
    expected_conclusion: str,
    conclusion_idx: int,
) -> tuple[float, dict]:
    """Use the shared LLM helper to judge semantic conclusion consistency."""
    check_name = f"stage6_conclusion_{conclusion_idx + 1}"

    try:
        prompt = (
            f"You are evaluating whether a scientific conclusion produced by an AI agent "
            f"is semantically consistent with an expected finding.\n\n"
            f'Expected finding:\n"{expected_conclusion}"\n\n'
            f'Agent\'s conclusion:\n"{agent_conclusion}"\n\n'
            f"Does the agent's conclusion capture the same core scientific message as "
            f"the expected finding? Minor wording differences are acceptable as long as "
            f"the scientific meaning is preserved.\n\n"
            f"Answer YES or NO, followed by a brief explanation."
        )

        answer = await llm_multimodal_text(
            content=[{"type": "text", "text": prompt}],
            model=resolve_llm_judge_model(
                env_var="VEGETATION_PRECIP_LLM_JUDGE_MODEL",
            ),
            max_tokens=256,
            temperature=0,
        )
        score = 1.0 if "YES" in answer.upper() else 0.0
        return (
            score,
            {
                "check": check_name,
                "passed": score == 1.0,
                "message": f"LLM judge: {answer[:200]}",
            },
        )

    except Exception as e:
        logger.warning(
            f"LLM judge failed for conclusion {conclusion_idx + 1}, "
            f"falling back to keyword matching: {e}"
        )
        return _keyword_judge_conclusion(agent_conclusion, conclusion_idx)


def _keyword_judge_conclusion(
    agent_conclusion: str,
    conclusion_idx: int,
) -> tuple[float, dict]:
    """Fallback keyword-based check for each conclusion."""
    check_name = f"stage6_conclusion_{conclusion_idx + 1}"
    text = agent_conclusion.lower()

    if conclusion_idx == 0:
        required = [
            any(w in text for w in ["spatial", "across space", "cross-sectional"]),
            any(w in text for w in ["temporal", "over time", "interannual"]),
            any(w in text for w in ["stronger", "more strongly", "larger", "greater"]),
            any(w in text for w in ["precipitation", "rainfall", "ppt"]),
        ]
        label = "space>time for PPT"
    elif conclusion_idx == 1:
        required = [
            any(w in text for w in ["seasonal", "variability", "variation"]),
            any(w in text for w in ["rooting depth", "root", "soil"]),
        ]
        label = "factors: water variability + soil/roots"
    else:
        required = [
            any(w in text for w in ["soil moisture", "ssm"]),
            any(w in text for w in ["converge", "convergence", "smaller", "less"]),
        ]
        label = "SSM convergence"

    passed = all(required)
    return (
        1.0 if passed else 0.0,
        {
            "check": check_name,
            "passed": passed,
            "message": (
                f"Keyword match OK ({label})"
                if passed
                else f"Keyword match FAILED ({label}): matched {sum(required)}/{len(required)} criteria"
            ),
        },
    )


async def _check_stage6_conclusions(
    file_map: dict[str, str],
) -> tuple[float, list[dict]]:
    """Stage 6: Validate conclusions.json (3 conclusions, 1 point each)."""
    checks = []
    total_score = 0.0
    fname = "conclusions.json"

    if fname not in file_map:
        checks.append(
            {
                "check": "stage6_exists",
                "passed": False,
                "message": f"Missing: {fname}",
            }
        )
        return (0.0, checks)

    try:
        with open(file_map[fname]) as f:
            conclusions = json_mod.load(f)
    except Exception as e:
        checks.append(
            {
                "check": "stage6_parse",
                "passed": False,
                "message": f"Cannot parse {fname}: {e}",
            }
        )
        return (0.0, checks)

    if not isinstance(conclusions, list) or len(conclusions) < 3:
        checks.append(
            {
                "check": "stage6_structure",
                "passed": False,
                "message": f"Expected list of 3 strings, got {type(conclusions).__name__} with {len(conclusions) if isinstance(conclusions, list) else 'N/A'} items",
            }
        )
        return (0.0, checks)

    for i in range(3):
        agent_text = str(conclusions[i])
        score, check = await _llm_judge_conclusion(agent_text, EXPECTED_CONCLUSIONS[i], i)
        checks.append(check)
        total_score += score

    return (total_score, checks)


def _print_results(checks: list[dict]):
    """Pretty-print evaluation results."""
    print("\nEvaluation Results:")
    for c in checks:
        status = "\u2713" if c["passed"] else "\u2717"
        print(f"  {status} {c['check']}: {c['message']}")


TOTAL_POINTS = 6  # Active stages: 3 (1pt) + 4 (1pt) + 5 (1pt) + 6 (3pt)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Evaluate agent outputs with staged milestone scoring.

    Active stages (6 points total):
      Stage 3 (1pt): PPT sensitivity CSV — ndvi_ppt_sensitivity.csv
      Stage 4 (1pt): SSM sensitivity CSV — ndvi_ssm_sensitivity.csv
      Stage 5 (1pt): Feature importance CSV — feature_importance.csv
      Stage 6 (3pt): Conclusions — conclusions.json (1pt per conclusion)

    Commented-out stages (re-enable later):
      Stage 1 (1pt): Raw data download — ndvi_raw.nc, ppt_raw.nc, ssm_raw.nc
      Stage 2 (1pt): Regridded merged file — merged.nc

    Returns [total_score / TOTAL_POINTS].
    """
    metadata = task_cfg.metadata
    output_dir = metadata["remote_output_dir"]
    task_tag = metadata["variant_name"]
    required_biome_keywords = metadata.get(
        "required_biome_keywords", ["all", "grass", "savanna", "shrub"]
    )
    ppt_min_rel_diff = metadata.get("ppt_min_relative_diff", 0.40)
    ssm_max_rel_diff = metadata.get("ssm_max_relative_diff", 0.20)
    min_fi_r2 = metadata.get("min_feature_importance_r2", 0.40)

    all_expected = [
        # "ndvi_raw.nc", "ppt_raw.nc", "ssm_raw.nc",  # Stage 1
        # "merged.nc",                                  # Stage 2
        "ndvi_ppt_sensitivity.csv",  # Stage 3
        "ndvi_ssm_sensitivity.csv",  # Stage 4
        "feature_importance.csv",  # Stage 5
        "conclusions.json",  # Stage 6
    ]

    async with EvaluationContext(
        task_tag=task_tag,
        mode="custom",
        split="train",
    ) as ctx:
        all_checks = []
        total_points = 0.0

        try:
            output_files = await session.list_dir(output_dir)
        except Exception as e:
            logger.error(f"Cannot list output dir: {e}")
            ctx.add_score(0.0)
            return [0.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            file_map: dict[str, str] = {}
            local_reference_dir = Path(tmpdir) / "reference"
            local_reference_dir.mkdir(parents=True, exist_ok=True)
            output_set = {f.lower(): f for f in output_files}

            for fname in (
                "ndvi_ppt_sensitivity.csv",
                "ndvi_ssm_sensitivity.csv",
                "feature_importance.csv",
                "conclusions.json",
            ):
                remote_reference_path = rf"{metadata['reference_dir']}\{fname}"
                try:
                    content = await session.read_bytes(remote_reference_path)
                    (local_reference_dir / fname).write_bytes(content)
                except Exception as e:
                    logger.error(f"Failed to download staged reference {fname}: {e}")
                    ctx.add_score(0.0)
                    return [0.0]

            for fname in all_expected:
                actual_name = output_set.get(fname.lower())
                if actual_name:
                    remote_path = rf"{output_dir}\{actual_name}"
                    try:
                        content = await session.read_bytes(remote_path)
                        local_path = os.path.join(tmpdir, fname)
                        with open(local_path, "wb") as fh:
                            fh.write(content)
                        file_map[fname] = local_path
                    except Exception as e:
                        logger.warning(f"Failed to download {fname}: {e}")

            # ── Stage 1: Data Download (1 pt) ── DISABLED ────────────────
            # Uncomment to re-enable:
            # score, checks = _check_stage1_download(tmpdir, file_map)
            # total_points += score
            # all_checks.extend(checks)
            # ctx.add_score(score)

            # ── Stage 2: Regridding (1 pt) ── DISABLED ───────────────────
            # Uncomment to re-enable:
            # score, checks = _check_stage2_regrid(tmpdir, file_map)
            # total_points += score
            # all_checks.extend(checks)
            # ctx.add_score(score)

            # ── Stage 3: PPT Sensitivity CSV (1 pt) ──────────────────────
            score, checks = _check_sensitivity_csv(
                file_map,
                "ndvi_ppt_sensitivity.csv",
                "stage3",
                required_biome_keywords,
                relative_diff_check="gt",
                relative_diff_threshold=ppt_min_rel_diff,
                reference_fname="ndvi_ppt_sensitivity.csv",
                local_reference_dir=local_reference_dir,
            )
            total_points += score
            all_checks.extend(checks)
            ctx.add_score(score)

            # ── Stage 4: SSM Sensitivity CSV (1 pt) ──────────────────────
            score, checks = _check_sensitivity_csv(
                file_map,
                "ndvi_ssm_sensitivity.csv",
                "stage4",
                required_biome_keywords,
                relative_diff_check="lt",
                relative_diff_threshold=ssm_max_rel_diff,
                reference_fname="ndvi_ssm_sensitivity.csv",
                local_reference_dir=local_reference_dir,
            )
            total_points += score
            all_checks.extend(checks)
            ctx.add_score(score)

            # ── Stage 5: Feature Importance (1 pt) ───────────────────────
            score, checks = _check_stage5_feature_importance(
                file_map,
                min_fi_r2,
                local_reference_dir,
            )
            total_points += score
            all_checks.extend(checks)
            ctx.add_score(score)

            # ── Stage 6: Conclusions (3 pt) ──────────────────────────────
            score, checks = await _check_stage6_conclusions(file_map)
            total_points += score
            all_checks.extend(checks)
            ctx.add_score(score)

        _print_results(all_checks)

        final_score = total_points / TOTAL_POINTS
        print(f"\nTotal: {total_points:.1f} / {TOTAL_POINTS} points = {final_score:.4f}")
        return [final_score]


if __name__ == "__main__":
    print("Task Configuration:")
    print(f"  VARIANT_NAME: {config.VARIANT_NAME}")
    print(f"  Task Description: {config.task_description[:200]}...")
    print(f"\nMetadata:")
    print(json_mod.dumps(config.to_metadata(), indent=2))
