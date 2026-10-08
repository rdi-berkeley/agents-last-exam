"""Freeze submitted inputs and check the explicit model contract without model services."""

import asyncio
import hashlib
import json
import math
import shlex
import shutil
import tempfile
from pathlib import Path

from .scripts.native_model_contract import CONTRACT, validate_model
from .remote_commands import run_command
from .observation_evidence import build_controller_evidence
from .verification import CASES, EvaluationUnavailableError, verify_source_bundle

ASSETS = Path(__file__).with_name("assets")


class EvidenceCoverageError(EvaluationUnavailableError):
    pass


def required_evidence_coverage(diagnostics, diagnostics_hash, input_hashes):
    coverage = diagnostics.get("coverage")
    keys = {
        "scope",
        "baseline_input_hashes",
        "diagnostic_input_hashes",
        "evidence_sha256",
        "responses_mm",
    }
    if not isinstance(coverage, dict) or set(coverage) != keys:
        raise EvidenceCoverageError("Native diagnostics lack explicit evidence coverage facts")
    if coverage["baseline_input_hashes"] != input_hashes:
        raise EvidenceCoverageError("Native coverage concerns a different baseline model")
    for key in ("baseline_input_hashes", "diagnostic_input_hashes", "evidence_sha256"):
        hashes = coverage[key]
        if (
            not isinstance(hashes, dict)
            or not hashes
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for name, value in hashes.items()
            )
        ):
            raise EvidenceCoverageError("Native coverage lacks valid hashes: " + key)
    if set(coverage["diagnostic_input_hashes"]) != set(input_hashes):
        raise EvidenceCoverageError("Native coverage lacks diagnostic model input hashes")
    scope = coverage["scope"]
    if not isinstance(scope, dict) or set(scope) != {
        "cases",
        "bottom_domain_compared",
        "local_mesh_refinement_compared",
    }:
        raise EvidenceCoverageError("Native coverage lacks the tested comparison scope")
    cases = scope["cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, str) or case not in CASES for case in cases)
        or len(set(cases)) != len(cases)
        or any(
            type(scope[key]) is not bool
            for key in ("bottom_domain_compared", "local_mesh_refinement_compared")
        )
    ):
        raise EvidenceCoverageError("Native coverage has invalid comparison scope")
    responses = coverage["responses_mm"]
    if not isinstance(responses, dict) or set(responses) != {"max_settlement_mm", "max_disp_mm"}:
        raise EvidenceCoverageError("Native coverage requires both displacement maxima")
    for values in responses.values():
        if (
            not isinstance(values, dict)
            or set(values) != {"baseline", "diagnostic"}
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values.values()
            )
        ):
            raise EvidenceCoverageError("Native coverage has invalid displacement maxima")
    expected = {"diagnostics_sha256": diagnostics_hash, **coverage}
    if "controller_evidence" in diagnostics:
        evidence = diagnostics["controller_evidence"]
        try:
            rebuilt = build_controller_evidence(evidence["records"], input_hashes)
            if rebuilt != evidence:
                raise ValueError("Aggregation differs from retained native observations")
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise EvidenceCoverageError("Invalid controller evidence: " + str(error)) from error
        expected["controller_evidence"] = {
            "aggregation_sha256": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest(),
            "baseline_cases": sorted(
                row["case"] for row in evidence["cases"].values() if row["kind"] == "baseline"
            ),
            "diagnostic_cases": {
                label: row["case"]
                for label, row in evidence["cases"].items()
                if row["kind"] == "diagnostic"
            },
            "observation_scope": "discrete_DBC_stations_and_full_registered_CX_profiles",
        }
    return expected


def validate_evidence_coverage(response, expected):
    coverage = response.get("evidence_coverage")
    if not isinstance(coverage, dict) or set(coverage) != set(expected):
        raise EvidenceCoverageError("Diagnostic receipt omitted native evidence coverage")
    if (
        "controller_evidence" in expected
        and coverage["controller_evidence"] != expected["controller_evidence"]
    ):
        raise EvidenceCoverageError(
            "Diagnostic receipt omitted or misstated controller evidence scope/hash"
        )
    for key in (
        "diagnostics_sha256",
        "baseline_input_hashes",
        "diagnostic_input_hashes",
        "evidence_sha256",
    ):
        if coverage[key] != expected[key]:
            raise EvidenceCoverageError(
                "Diagnostic receipt cited different native evidence hashes: " + key
            )
    scope = coverage["scope"]
    if (
        not isinstance(scope, dict)
        or set(scope) != set(expected["scope"])
        or any(
            type(scope[key]) is not type(value) or scope[key] != value
            for key, value in expected["scope"].items()
        )
    ):
        raise EvidenceCoverageError("Diagnostic receipt misstated bottom/local-mesh evidence scope")
    responses = coverage["responses_mm"]
    if not isinstance(responses, dict) or set(responses) != set(expected["responses_mm"]):
        raise EvidenceCoverageError("Diagnostic receipt omitted native displacement maxima")
    for quantity, expected_values in expected["responses_mm"].items():
        values = responses[quantity]
        if (
            not isinstance(values, dict)
            or set(values) != set(expected_values)
            or any(
                type(values[key]) not in (int, float)
                or not math.isfinite(values[key])
                or not math.isclose(values[key], value, rel_tol=0, abs_tol=1e-6)
                for key, value in expected_values.items()
            )
        ):
            raise EvidenceCoverageError(
                "Diagnostic receipt misstated native displacement maxima: " + quantity
            )


async def review_native_family(family_path, original_sources, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = json.loads((ASSETS / "source_manifest.json").read_text())["sha256"]
    verify_source_bundle(
        {name: (Path(original_sources) / name).read_bytes() for name in source_hashes}
    )
    record = {
        "format": CONTRACT,
        "full_task_acceptance": False,
        "native_physics_or_responses_certified": False,
        "engineering_caveats": [],
        "evidence_directory": str(output),
    }
    try:
        native = validate_model(family_path)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        finding = {
            "requirement": "model_definition.md: Deterministic admissibility",
            "submitted_feature": "family.json / model.mdpa / Materials.json",
            "reason": str(error),
        }
        record.update(
            outcome="source_violation", replay_eligible=False, blocking_findings=[finding]
        )
    else:
        record.update(
            outcome="source_consistent",
            replay_eligible=True,
            blocking_findings=[],
            input_hashes=native["input_hashes"],
        )
        (output / "native-review.json").write_text(json.dumps(native, indent=2))
    (output / "source-review-result.json").write_text(json.dumps(record, indent=2))
    return record


async def review_submission(metadata, session):
    if session is None or not {"input_dir", "remote_output_dir"}.issubset(metadata):
        raise EvaluationUnavailableError(
            "Source review needs task input/output paths; not a solver score"
        )
    sources = json.loads((ASSETS / "source_manifest.json").read_text())["sha256"]
    model_root = Path(metadata["remote_output_dir"]) / "model_family_support_position"
    paths = {f"sources/{name}": str(Path(metadata["input_dir"]) / name) for name in sources}
    paths.update(
        {
            f"family/{name}": str(model_root / name)
            for name in ("family.json", "model.mdpa", "Materials.json")
        }
    )
    check = """import hashlib,json,sys
from pathlib import Path
result={}
for key,name in json.loads(sys.argv[1]).items():
    path=Path(name)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Missing or unsafe supported native input: '+name)
    if path.stat().st_size>100000000:
        raise ValueError('Supported input size exceeded: '+name)
    result[key]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(result))
"""
    stat = await run_command(
        session, shlex.join(["python3", "-c", check, json.dumps(paths)]), timeout=30, check=False
    )
    if stat.get("return_code", 1):
        raise EvaluationUnavailableError(
            "Submitted native input adapter could not prepare source review: "
            + stat.get("stderr", "")
        )
    inventory = json.loads(stat["stdout"])
    root = ASSETS.parent / "scripts/evidence/task-source-reviews"
    root.mkdir(parents=True, exist_ok=True)
    required = sum(row["bytes"] for row in inventory.values())
    if shutil.disk_usage(root).free < 8 * 1024**3 + required + 32 * 1024**2:
        raise EvaluationUnavailableError(
            "Source review snapshot would cross evaluator disk reserve"
        )
    snapshot = Path(tempfile.mkdtemp(prefix="review-", dir=root))
    for key, remote in paths.items():
        content = await session.read_bytes(remote)
        if hashlib.sha256(content).hexdigest() != inventory[key]["sha256"]:
            raise EvaluationUnavailableError(
                "Submitted inputs changed during source-review snapshot"
            )
        target = snapshot / key
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
    record = await review_native_family(
        snapshot / "family", snapshot / "sources", snapshot / "review"
    )
    record["evidence_directory"] = str(snapshot / "review")
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("original_sources")
    parser.add_argument("output")
    arguments = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                review_native_family(arguments.family, arguments.original_sources, arguments.output)
            ),
            indent=2,
        )
    )
