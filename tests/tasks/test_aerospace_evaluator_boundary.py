import asyncio
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.engineering.aerospace_low_thrust_trajectory.scripts import score_outputs as scorer
from tasks.engineering.aerospace_low_thrust_trajectory.scripts import tier3_contract as contract


def array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array)
    return stream.getvalue()


@pytest.fixture
def reference_bundle():
    """Structurally valid data for boundary tests, not a physical certificate."""
    results = {
        "tier1": dict.fromkeys(("transfer_orbit_sma_km", "dv1_m_s", "dv2_m_s", "dv_total_m_s", "transfer_time_s", "transfer_time_hours", "v_circular_leo_m_s", "v_circular_geo_m_s"), 1.0),
        "tier2": dict.fromkeys(("final_sma_km", "final_eccentricity", "final_inclination_deg", "dv_total_m_s", "transfer_time_s", "final_mass_kg", "fuel_consumed_kg", "edelbaum_dv_m_s"), 1.0),
        "tier3": dict.fromkeys(("final_sma_km", "final_eccentricity", "final_inclination_deg", "dv_total_m_s", "transfer_time_days", "fuel_consumed_kg", "constraint_violation_norm", "hamiltonian_initial", "hamiltonian_final"), 1.0),
    }
    results["tier3"].update(formulation=contract.VERSION, final_mass_kg=1700.0, shooting_converged=True)
    arrays = {}
    for name, columns in (("tier2_trajectory.npy", 8), ("tier3_trajectory.npy", 15), ("tier3_control.npy", 4)):
        array = np.ones((1000, columns))
        array[:, 0] = np.arange(1000)
        if columns == 15:
            array[:, 7] = 1700.0
        arrays[name] = array_bytes(array)
    return {"results.json": json.dumps(results).encode(), **arrays}


def test_missing_reference_is_not_a_candidate_zero():
    with pytest.raises(scorer.EvaluatorReferenceError, match="Missing"):
        scorer.score_submission({}, {})


@pytest.mark.parametrize("bad", ["missing", "json", "version", "mass", "nan", "shape", "archive", "time", "scalar", "bytes"])
def test_reference_faults_precede_candidate_inspection(reference_bundle, bad):
    if bad == "missing":
        del reference_bundle["tier2_trajectory.npy"]
    elif bad in ("json", "bytes"):
        reference_bundle["results.json"] = b"broken" if bad == "json" else None
    elif bad in ("version", "mass", "scalar"):
        result = json.loads(reference_bundle["results.json"])
        if bad == "version":
            result["tier3"]["formulation"] = "legacy"
        elif bad == "mass":
            result["tier3"]["final_mass_kg"] = True
        else:
            del result["tier1"]["dv1_m_s"]
        reference_bundle["results.json"] = json.dumps(result).encode()
    elif bad == "archive":
        stream = io.BytesIO()
        np.savez(stream, values=np.ones(5))
        reference_bundle["tier3_trajectory.npy"] = stream.getvalue()
    else:
        array = np.load(io.BytesIO(reference_bundle["tier3_trajectory.npy"]))
        if bad == "nan":
            array[0, 1] = np.nan
        elif bad == "shape":
            array = array[:, :14]
        else:
            array[1, 0] = array[0, 0]
        reference_bundle["tier3_trajectory.npy"] = array_bytes(array)
    with pytest.raises(scorer.EvaluatorReferenceError):
        scorer.score_submission({}, reference_bundle)


@pytest.mark.parametrize("bad", ["missing", "json", "array", "dtype", "section"])
def test_candidate_faults_remain_zero(reference_bundle, bad):
    candidate = reference_bundle.copy()
    if bad == "missing":
        del candidate["tier3_control.npy"]
    elif bad == "json":
        candidate["results.json"] = b"{"
    elif bad == "array":
        candidate["tier3_trajectory.npy"] = b"broken"
    elif bad == "dtype":
        candidate["tier2_trajectory.npy"] = array_bytes(np.full((1000, 8), "bad"))
    else:
        result = json.loads(candidate["results.json"])
        result["tier2"] = None
        candidate["results.json"] = json.dumps(result).encode()
    assert scorer.score_submission(candidate, reference_bundle).score == 0.0


@pytest.mark.parametrize("error", [RuntimeError, TypeError, IndexError, OSError, MemoryError])
def test_unexpected_checker_fault_propagates(reference_bundle, monkeypatch, error):
    scorer.validate_reference(reference_bundle)

    def broken(*args):
        raise error("internal fault")

    monkeypatch.setattr(scorer, "_check_tier2", broken)
    with pytest.raises(error, match="internal fault"):
        scorer.score_submission(reference_bundle, reference_bundle)


@pytest.mark.parametrize("failure", ["missing_metadata", "missing_reference", "reference_race", "missing_candidate", "candidate_race", "malformed_candidate", "session_fault"])
def test_installed_metadata_entrypoint_boundary(reference_bundle, failure):
    from tasks.engineering.aerospace_low_thrust_trajectory import main

    metadata = main.TaskConfig().to_metadata()
    reference_paths = metadata["reference_files"]
    candidate_paths = metadata["candidate_files"]
    assert all("/reference/" in path for path in reference_paths.values())
    assert all("/output/" in path for path in candidate_paths.values())
    files = {reference_paths[name]: payload for name, payload in reference_bundle.items()}
    files.update({candidate_paths[name]: payload for name, payload in reference_bundle.items()})
    if failure == "missing_metadata":
        del metadata["reference_files"]
    elif failure == "missing_reference":
        del files[reference_paths["results.json"]]
    elif failure == "missing_candidate":
        del files[candidate_paths["results.json"]]
    elif failure == "malformed_candidate":
        files[candidate_paths["results.json"]] = b"{"

    class Session:
        async def file_exists(self, path):
            if failure == "session_fault":
                raise RuntimeError("session unavailable")
            return path in files

        async def read_bytes(self, path):
            if failure == "reference_race" and path == reference_paths["results.json"]:
                raise FileNotFoundError(path)
            if failure == "candidate_race" and path == candidate_paths["results.json"]:
                raise FileNotFoundError(path)
            return files[path]

    evaluation = main.evaluate(SimpleNamespace(metadata=metadata), Session())
    if failure in ("missing_metadata", "missing_reference", "reference_race"):
        with pytest.raises(scorer.EvaluatorReferenceError):
            asyncio.run(evaluation)
    elif failure == "session_fault":
        with pytest.raises(RuntimeError, match="session unavailable"):
            asyncio.run(evaluation)
    else:
        assert asyncio.run(evaluation) == [0.0]
