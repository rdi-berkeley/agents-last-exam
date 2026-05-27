#!/usr/bin/env python3
"""VM-side verifier for the GLM lake-calibration benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


LAKE_DEPTH = 25
START_DATE = datetime(2009, 1, 1, 12, 0, 0)
RMSE_THRESHOLD = 1.5
EXPECTED_RUNTIME_MD5S = {
    "glm_wrapper": "a47417ab407c3c6487e7169c71bfbab3",
    "run_wrapper": "70a4c1f7ca32988cd67b118028631ec6",
}


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_glm_output(nc_path: Path) -> pd.DataFrame:
    with Dataset(nc_path, "r") as nc:
        time = nc.variables["time"][:]
        z = nc.variables["z"][:]
        temp = nc.variables["temp"][:]
        records: list[dict[str, object]] = []
        for t_idx in range(len(time)):
            date = pd.Timestamp(START_DATE) + pd.Timedelta(hours=float(time[t_idx]))
            heights = z[t_idx, :, 0, 0]
            temps = temp[t_idx, :, 0, 0]
            for d_idx in range(len(heights)):
                h_val = heights[d_idx]
                t_val = temps[d_idx]
                if not np.ma.is_masked(h_val) and not np.ma.is_masked(t_val):
                    depth = LAKE_DEPTH - float(h_val)
                    if 0 <= depth <= LAKE_DEPTH:
                        records.append(
                            {
                                "datetime": date,
                                "depth": round(depth),
                                "temp_sim": float(t_val),
                            }
                        )
    df = pd.DataFrame(records)
    return df.groupby(["datetime", "depth"]).agg({"temp_sim": "mean"}).reset_index()


def read_observations(obs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(obs_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["depth"] = df["depth"].round().astype(int)
    df = df.rename(columns={"temp": "temp_obs"})
    return df[["datetime", "depth", "temp_obs"]]


def calculate_rmse(sim_df: pd.DataFrame, obs_df: pd.DataFrame) -> float:
    merged = pd.merge(obs_df, sim_df, on=["datetime", "depth"], how="inner")
    if merged.empty:
        return 999.0
    return float(np.sqrt(np.mean((merged["temp_sim"] - merged["temp_obs"]) ** 2)))


def output_time_range(nc_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    with Dataset(nc_path, "r") as nc:
        time = nc.variables["time"][:]
    first_date = pd.Timestamp(START_DATE) + pd.Timedelta(hours=float(time[0]))
    last_date = pd.Timestamp(START_DATE) + pd.Timedelta(hours=float(time[-1]))
    return first_date, last_date


def direct_output_checks(output_nc: Path, obs_path: Path) -> tuple[list[dict], float]:
    checks: list[dict[str, object]] = []
    first_date = pd.Timestamp.min
    last_date = pd.Timestamp.min
    rmse = 999.0

    if output_nc.exists():
        sim_df = read_glm_output(output_nc)
        obs_df = read_observations(obs_path)
        first_date, last_date = output_time_range(output_nc)
        rmse = calculate_rmse(sim_df, obs_df)

    checks.append(
        {
            "check": "output_exists",
            "passed": output_nc.exists(),
            "message": str(output_nc),
        }
    )
    checks.append(
        {
            "check": "output_starts_2009_01",
            "passed": output_nc.exists() and first_date.year == 2009 and first_date.month == 1,
            "message": str(first_date) if output_nc.exists() else "missing",
        }
    )
    checks.append(
        {
            "check": "output_reaches_2015_12_29",
            "passed": output_nc.exists() and last_date >= pd.Timestamp("2015-12-29"),
            "message": str(last_date) if output_nc.exists() else "missing",
        }
    )
    checks.append(
        {
            "check": "rmse_below_threshold",
            "passed": output_nc.exists() and rmse < RMSE_THRESHOLD,
            "message": f"{rmse:.6f}" if output_nc.exists() else "missing",
        }
    )
    return checks, rmse


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_fixture_metrics(reference_dir: Path) -> dict[str, object]:
    return json.loads((reference_dir / "fixture_metrics.json").read_text(encoding="utf-8"))


def verify_visible_state(
    *,
    input_dir: Path,
    software_dir: Path,
    metrics: dict[str, object],
) -> list[dict]:
    forcing_md5 = metrics["forcing_md5"]
    expected = {
        "observation_file": (
            input_dir / "field_temp_oxy.csv",
            metrics["observation_md5"],
        ),
        "forcing/meteo.csv": (
            input_dir / "bcs" / "meteo.csv",
            forcing_md5["meteo.csv"],
        ),
        "forcing/outflow.csv": (
            input_dir / "bcs" / "outflow.csv",
            forcing_md5["outflow.csv"],
        ),
        "forcing/pheasant.csv": (
            input_dir / "bcs" / "pheasant.csv",
            forcing_md5["pheasant.csv"],
        ),
        "forcing/yahara.csv": (
            input_dir / "bcs" / "yahara.csv",
            forcing_md5["yahara.csv"],
        ),
        "glm_binary": (
            software_dir / "bin" / "glm.bin",
            metrics["glm_binary_md5"],
        ),
        "glm_wrapper": (
            software_dir / "glm",
            EXPECTED_RUNTIME_MD5S["glm_wrapper"],
        ),
        "run_wrapper": (
            software_dir / "run_glm_from_input.sh",
            EXPECTED_RUNTIME_MD5S["run_wrapper"],
        ),
    }

    checks: list[dict[str, object]] = []
    for key, (path, expected_md5) in expected.items():
        current_md5 = compute_md5(path) if path.exists() else "missing"
        checks.append(
            {
                "check": f"{key}_integrity",
                "passed": current_md5 == expected_md5,
                "message": current_md5,
            }
        )
    return checks


def build_rerun_workspace(
    *,
    candidate_glm_path: Path,
    input_dir: Path,
    software_dir: Path,
) -> Path:
    work_root = Path(tempfile.mkdtemp(prefix="agenthle_glm_eval_"))
    input_root = work_root / "input"
    software_root = work_root / "software"
    output_root = work_root / "output"

    input_root.mkdir(parents=True, exist_ok=True)
    software_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate_glm_path, input_root / "glm3.nml")
    shutil.copy2(input_dir / "field_temp_oxy.csv", input_root / "field_temp_oxy.csv")
    _copy_tree(input_dir / "bcs", input_root / "bcs")

    shutil.copy2(software_dir / "glm", software_root / "glm")
    shutil.copy2(
        software_dir / "run_glm_from_input.sh",
        software_root / "run_glm_from_input.sh",
    )
    shutil.copy2(
        software_dir / "python_glm_env.sh",
        software_root / "python_glm_env.sh",
    )
    (software_root / "glm").chmod(0o755)
    (software_root / "run_glm_from_input.sh").chmod(0o755)
    (software_root / "python_glm_env.sh").chmod(0o755)

    _copy_tree(software_dir / "bin", software_root / "bin")
    _copy_tree(software_dir / "lib", software_root / "lib")
    return work_root


def rerun_checks(
    *,
    candidate_glm_path: Path,
    input_dir: Path,
    software_dir: Path,
) -> tuple[list[dict], float]:
    checks: list[dict[str, object]] = []
    work_root = build_rerun_workspace(
        candidate_glm_path=candidate_glm_path,
        input_dir=input_dir,
        software_dir=software_dir,
    )
    try:
        run_script = work_root / "software" / "run_glm_from_input.sh"
        result = subprocess.run(
            [str(run_script)],
            cwd=str(work_root),
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
        checks.append(
            {
                "check": "glm_runs_successfully",
                "passed": result.returncode == 0,
                "message": result.stderr.strip()[:1000],
            }
        )
        if result.returncode != 0:
            checks.extend(
                [
                    {
                        "check": "output_exists",
                        "passed": False,
                        "message": str(work_root / "output" / "output.nc"),
                    },
                    {
                        "check": "output_starts_2009_01",
                        "passed": False,
                        "message": "glm_run_failed",
                    },
                    {
                        "check": "output_reaches_2015_12_29",
                        "passed": False,
                        "message": "glm_run_failed",
                    },
                    {
                        "check": "rmse_below_threshold",
                        "passed": False,
                        "message": "glm_run_failed",
                    },
                ]
            )
            return checks, 999.0

        output_nc = work_root / "output" / "output.nc"
        output_checks, rmse = direct_output_checks(
            output_nc, work_root / "input" / "field_temp_oxy.csv"
        )
        checks.extend(output_checks)
        return checks, rmse
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def fixture_output_checks(
    *,
    output_dir: Path,
    obs_path: Path,
) -> tuple[list[dict], float]:
    output_nc = output_dir / "output.nc"
    return direct_output_checks(output_nc, obs_path)


def evaluate_mode(
    *,
    mode: str,
    input_dir: Path,
    software_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    metrics = load_fixture_metrics(reference_dir)
    checks = verify_visible_state(
        input_dir=input_dir,
        software_dir=software_dir,
        metrics=metrics,
    )
    details: dict[str, object] = {"mode": mode}

    if mode == "output":
        candidate_glm_path = input_dir / "glm3.nml"
        checks.append(
            {
                "check": "candidate_glm3_present",
                "passed": candidate_glm_path.exists(),
                "message": str(candidate_glm_path),
            }
        )
        if not candidate_glm_path.exists():
            return {
                "score": 0.0,
                "passed": False,
                "checks": checks,
                "details": details,
            }
        details["candidate_source"] = "input/glm3.nml"
        mode_checks, rmse = rerun_checks(
            candidate_glm_path=candidate_glm_path,
            input_dir=input_dir,
            software_dir=software_dir,
        )
    elif mode == "output_test_pos":
        details["candidate_source"] = "output_test_pos/output.nc"
        mode_checks, rmse = fixture_output_checks(
            output_dir=output_dir,
            obs_path=input_dir / "field_temp_oxy.csv",
        )
    elif mode == "output_test_neg":
        details["candidate_source"] = "output_test_neg/output.nc"
        mode_checks, rmse = fixture_output_checks(
            output_dir=output_dir,
            obs_path=input_dir / "field_temp_oxy.csv",
        )
    else:
        checks.append(
            {
                "check": "supported_mode",
                "passed": False,
                "message": mode,
            }
        )
        return {
            "score": 0.0,
            "passed": False,
            "checks": checks,
            "details": details,
        }

    checks.extend(mode_checks)
    details["rmse"] = rmse
    passed = all(bool(item["passed"]) for item in checks)
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "checks": checks,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["output", "output_test_pos", "output_test_neg"],
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--software-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_mode(
        mode=args.mode,
        input_dir=Path(args.input_dir),
        software_dir=Path(args.software_dir),
        reference_dir=Path(args.reference_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
