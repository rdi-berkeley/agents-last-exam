import argparse
import json

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--required-vars", required=True)
    return parser.parse_args()


def list_dir(path):
    import os

    return sorted(os.listdir(path))


def evaluate_netcdf_content(output_path, reference_path, required_vars):
    checks = []

    with xr.open_dataset(reference_path) as ref_ds:
        ref_dims = dict(ref_ds.sizes)
        ref_data_vars = list(ref_ds.data_vars)

    with xr.open_dataset(output_path) as out_ds:
        out_dims = dict(out_ds.sizes)
        out_data_vars = list(out_ds.data_vars)

    has_xy_time = "x" in out_dims and "y" in out_dims and "time" in out_dims
    checks.append(
        {
            "check": "nc_dims_xy_time",
            "passed": has_xy_time,
            "message": (
                f"Has x, y, time dims: {dict(out_dims)}"
                if has_xy_time
                else f"Expected x,y,time dims, got {dict(out_dims)}"
            ),
        }
    )
    if not has_xy_time:
        return checks

    dims_match = True
    mismatches = []
    for dim_name in ["x", "y", "time"]:
        out_len = out_dims.get(dim_name, 0)
        ref_len = ref_dims.get(dim_name, 0)
        if out_len != ref_len:
            dims_match = False
            mismatches.append(f"{dim_name}: {out_len} vs ref {ref_len}")
    checks.append(
        {
            "check": "nc_dim_lengths_match_reference",
            "passed": dims_match,
            "message": (
                "All dim lengths match reference"
                if dims_match
                else "; ".join(mismatches)
            ),
        }
    )
    if not dims_match:
        return checks

    required_set = set(required_vars)
    var_match = required_set.issubset(set(out_data_vars)) and set(out_data_vars) <= required_set
    checks.append(
        {
            "check": "nc_variables_match_required",
            "passed": var_match,
            "message": (
                f"Variables {out_data_vars} match required {required_vars}"
                if var_match
                else f"Expected {required_vars}, got {out_data_vars}"
            ),
        }
    )
    if not var_match:
        return checks

    with xr.open_dataset(output_path) as out_ds, xr.open_dataset(reference_path) as ref_ds:
        max_diff = 0.0
        diff_zero = True
        for var_name in required_vars:
            if var_name not in out_ds.data_vars or var_name not in ref_ds.data_vars:
                diff_zero = False
                break
            diff_da = out_ds[var_name] - ref_ds[var_name]
            diff_vals = np.abs(diff_da.values.astype(float))
            max_diff = max(max_diff, float(np.nanmax(diff_vals)))
            if max_diff > 1e-9:
                diff_zero = False

    checks.append(
        {
            "check": "nc_output_minus_reference_zero",
            "passed": diff_zero,
            "message": (
                "output - reference is zero"
                if diff_zero
                else f"output - reference max diff = {max_diff}"
            ),
        }
    )
    return checks


def main():
    args = parse_args()
    required_vars = [var for var in args.required_vars.split(",") if var]

    output_files = list_dir(args.output_dir)
    reference_files = list_dir(args.reference_dir)

    output_tif = sorted([name for name in output_files if name.lower().endswith(".tif")])
    reference_tif = sorted([name for name in reference_files if name.lower().endswith(".tif")])
    nc_files = sorted([name for name in output_files if name.lower().endswith(".nc")])
    ref_nc_files = sorted([name for name in reference_files if name.lower().endswith(".nc")])

    if not nc_files or not ref_nc_files:
        print(json.dumps({"checks": [], "score": 0.0}))
        return

    output_nc = rf"{args.output_dir}\{nc_files[0]}"
    reference_nc = rf"{args.reference_dir}\{ref_nc_files[0]}"
    checks = evaluate_netcdf_content(output_nc, reference_nc, required_vars)

    payload = {
        "tif_name_match": set(output_tif) == set(reference_tif),
        "output_tif_count": len(output_tif),
        "reference_tif_count": len(reference_tif),
        "checks": checks,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
