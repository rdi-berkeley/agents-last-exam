#!/usr/bin/env python
"""Hidden evaluator for the Allen NWB session translation task.

This script intentionally lives in the repo-side task implementation, not in
agent-visible input. It is uploaded to /tmp during evaluation and reads the
hidden reference bundle only after the agent has finished.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from pynwb import NWBHDF5IO, validate

REQUIRED_FILES = (
    "standardized_session.nwb",
    "qc_summary.json",
    "stimulus_response_metrics.csv",
    "conversion_report.md",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _load_session(path: Path) -> EcephysSession:
    return EcephysSession.from_nwb_path(
        str(path),
        api_kwargs={
            "filter_by_validity": False,
            "filter_out_of_brain_units": False,
        },
    )


def _validate_nwb(path: Path) -> None:
    result = validate(paths=[str(path)])
    errors = result[0] if isinstance(result, tuple) else result
    messages = [str(error) for error in errors]
    if messages:
        raise AssertionError(f"PyNWB validation errors: {messages}")


def _assert_required_files(submission_dir: Path) -> Path:
    missing = [name for name in REQUIRED_FILES if not (submission_dir / name).exists()]
    if missing:
        raise AssertionError(f"missing required output files: {missing}")
    return submission_dir / "standardized_session.nwb"


def _assert_structure(nwb_path: Path, manifest: dict[str, Any]) -> None:
    _validate_nwb(nwb_path)
    session = _load_session(nwb_path)
    if list(session.probes.index) != manifest["expected_probe_ids"]:
        raise AssertionError(f"unexpected probe ids: {list(session.probes.index)}")
    if list(session.channels.index) != manifest["expected_channel_ids"]:
        raise AssertionError("unexpected channel ids")
    if list(session.units.index) != manifest["expected_unit_ids"]:
        raise AssertionError(f"unexpected unit ids: {list(session.units.index)}")
    if list(session.stimulus_presentations.index) != manifest["expected_stimulus_indices"]:
        raise AssertionError("unexpected stimulus presentation ids")
    running_start = session.running_speed["start_time"].to_numpy(dtype=float)
    if not np.allclose(
        running_start,
        manifest["expected_running_start_times"],
        atol=1e-9,
        rtol=0.0,
    ):
        raise AssertionError("running speed timestamps do not match the aligned clock")

    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        if nwbfile.subject.subject_id != manifest["subject_id"]:
            raise AssertionError("wrong subject_id in NWB file")
        pupil = nwbfile.processing["behavior"]["pupil_diameter"]
        if not np.allclose(
            pupil.timestamps[:],
            manifest["expected_pupil_times"],
            atol=1e-9,
            rtol=0.0,
        ):
            raise AssertionError("pupil timestamps do not match the aligned clock")


def _compute_metrics_from_nwb(path: Path) -> pd.DataFrame:
    session = _load_session(path)
    presentations = session.stimulus_presentations[["start_time", "orientation"]]
    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()

    rows: list[dict[str, float | int]] = []
    for unit_id, unit_row in units.iterrows():
        spike_times = np.array(unit_row["spike_times"], dtype=float)
        for orientation in sorted(presentations["orientation"].unique()):
            starts = presentations[presentations["orientation"] == orientation][
                "start_time"
            ].to_numpy(dtype=float)
            window_counts = []
            latencies_ms = []
            for start in starts:
                window = spike_times[(spike_times >= start) & (spike_times < start + 0.30)]
                window_counts.append(int(len(window)))
                latency = float((window[0] - start) * 1000.0) if len(window) else math.nan
                latencies_ms.append(latency)
            valid_latencies = [value for value in latencies_ms if not math.isnan(value)]
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "orientation": float(orientation),
                    "mean_spike_count_0_300ms": float(np.mean(window_counts)),
                    "mean_first_spike_latency_ms": (
                        float(np.mean(valid_latencies)) if valid_latencies else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["unit_id", "orientation"]).reset_index(drop=True)


def _compare_metrics(submitted: pd.DataFrame, reference: pd.DataFrame, tolerance: float = 1e-6) -> None:
    if list(submitted.columns) != list(reference.columns):
        raise AssertionError(f"metric columns differ: {list(submitted.columns)}")
    submitted = submitted.sort_values(["unit_id", "orientation"]).reset_index(drop=True)
    reference = reference.sort_values(["unit_id", "orientation"]).reset_index(drop=True)
    if len(submitted) != len(reference):
        raise AssertionError(f"metric row count differs: {len(submitted)} != {len(reference)}")
    for column in ["unit_id", "orientation"]:
        if not submitted[column].equals(reference[column]):
            raise AssertionError(f"metric identifier column mismatch in {column}")
    for column in ["mean_spike_count_0_300ms", "mean_first_spike_latency_ms"]:
        if not np.allclose(
            submitted[column],
            reference[column],
            atol=tolerance,
            rtol=0.0,
            equal_nan=True,
        ):
            raise AssertionError(f"metric values differ in {column}")


def _extract_nwb_summary_fields(nwb_path: Path) -> dict[str, Any]:
    session = _load_session(nwb_path)
    with NWBHDF5IO(nwb_path, "r", load_namespaces=True) as io:
        nwbfile = io.read()
        subject_metadata = {
            "subject_id": nwbfile.subject.subject_id,
            "species": nwbfile.subject.species,
            "sex": nwbfile.subject.sex,
            "age": nwbfile.subject.age,
            "genotype": nwbfile.subject.genotype,
        }
        electrodes = (
            nwbfile.electrodes.to_dataframe()
            .reset_index()[
                [
                    "id",
                    "probe_id",
                    "probe_channel_number",
                    "probe_vertical_position",
                    "probe_horizontal_position",
                    "location",
                    "valid_data",
                ]
            ]
            .sort_values("id")
        )
        units = (
            nwbfile.units.to_dataframe()
            .reset_index()[
                [
                    "id",
                    "peak_channel_id",
                    "local_index",
                    "cluster_id",
                    "quality",
                    "firing_rate",
                    "isi_violations",
                    "presence_ratio",
                    "amplitude_cutoff",
                ]
            ]
            .sort_values("id")
        )
        stimuli = (
            nwbfile.intervals["drifting_gratings_presentations"]
            .to_dataframe()
            .reset_index()[
                [
                    "start_time",
                    "stop_time",
                    "stimulus_name",
                    "orientation",
                    "contrast",
                    "stimulus_block",
                    "stimulus_index",
                ]
            ]
            .sort_values("stimulus_index")
        )
        invalid_times = nwbfile.invalid_times.to_dataframe().reset_index()[
            ["start_time", "stop_time", "tags"]
        ]

    return _json_ready(
        {
            "session_id": int(session.ecephys_session_id),
            "subject_metadata": subject_metadata,
            "n_probes": int(len(session.probes)),
            "n_channels": int(len(session.channels)),
            "n_units_retained": int(len(session.units)),
            "stimulus_presentations": int(len(session.stimulus_presentations)),
            "running_rows": int(len(session.running_speed)),
            "invalid_intervals": int(len(session.invalid_times)),
            "probe_ids": [int(value) for value in session.probes.index.tolist()],
            "channel_rows": electrodes.to_dict(orient="records"),
            "unit_rows": units.to_dict(orient="records"),
            "stimulus_rows": stimuli.to_dict(orient="records"),
            "invalid_time_rows": [
                {
                    "start_time": float(row["start_time"]),
                    "stop_time": float(row["stop_time"]),
                    "tags": list(row["tags"]) if isinstance(row["tags"], tuple) else row["tags"],
                }
                for _, row in invalid_times.iterrows()
            ],
        }
    )


def _assert_summary(submitted_path: Path, reference_path: Path, nwb_path: Path) -> None:
    submitted = _read_json(submitted_path)
    reference = _read_json(reference_path)
    if submitted != reference:
        raise AssertionError("qc_summary.json does not match the hidden reference")

    extracted = _extract_nwb_summary_fields(nwb_path)
    for key, value in extracted.items():
        if submitted.get(key) != value:
            raise AssertionError(f"qc_summary.json field {key!r} does not match the NWB")


def _assert_report(path: Path, manifest: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in manifest["required_report_markers"] if marker not in text]
    if missing:
        raise AssertionError(f"conversion_report.md is missing required markers: {missing}")


def score_submission(submission_dir: Path, reference_dir: Path) -> dict[str, Any]:
    nwb_path = _assert_required_files(submission_dir)
    manifest = _read_json(reference_dir / "manifest.json")

    _assert_structure(nwb_path, manifest)
    _assert_report(submission_dir / "conversion_report.md", manifest)

    submitted_metrics_csv = pd.read_csv(submission_dir / "stimulus_response_metrics.csv")
    recomputed_metrics = _compute_metrics_from_nwb(nwb_path)
    reference_metrics = pd.read_csv(reference_dir / "stimulus_response_metrics.csv")
    _compare_metrics(submitted_metrics_csv, recomputed_metrics)
    _compare_metrics(recomputed_metrics, reference_metrics)

    _assert_summary(
        submission_dir / "qc_summary.json",
        reference_dir / "qc_summary.json",
        nwb_path,
    )

    return {"score": 1.0, "status": "pass", "message": "submission matches hidden reference"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--strict", action="store_true", help="raise instead of returning score 0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = score_submission(Path(args.submission_dir), Path(args.reference_dir))
    except Exception as exc:  # noqa: BLE001 - failures are scoring outcomes.
        if args.strict:
            raise
        payload = {
            "score": 0.0,
            "status": "fail",
            "message": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
