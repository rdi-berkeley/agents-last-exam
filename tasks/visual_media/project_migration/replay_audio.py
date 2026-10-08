"""Recompute production scoring from immutable recorded audio responses, without requests."""

import argparse
import asyncio
import base64
import copy
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil

from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_judge


async def replay_evaluation(source: Path, destination: Path) -> dict:
    original = json.loads(source.read_text())
    if "evaluations" in original:
        original = next(
            entry["result"]
            for entry in original["evaluations"]
            if entry["identifier"] == "audio_semantic_rubric"
        )
    destination.mkdir(parents=True, exist_ok=False)
    source_hashes = {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()}
    expected = {}
    for stem_index, stem in enumerate(original["delivery"]["stems"]):
        for index, (passage, judgment) in enumerate(
            zip(stem["passages"], stem["judgments"], strict=True)
        ):
            expected[f"judge-{stem_index:03d}-{index}"] = (passage, judgment)
    calls = []

    async def recorded_comparison(
        reference, delivered, native, receipt, provenance, *, transport=None
    ):
        passage, old = expected[receipt.stem]
        original_receipt = Path(passage["reference_path"]).parent / receipt.name
        request_path = original_receipt.with_suffix(".request.json")
        request = json.loads(request_path.read_text())
        if json.loads(original_receipt.read_text()) != old or any(
            old.get(key) != value or request.get(key) != value for key, value in provenance.items()
        ):
            raise JudgeInfrastructureError("Recorded judgment provenance differs from inputs")
        messages = [{"role": "user", "content": audio_judge.RUBRIC}]
        normalized_hashes, gains = [], []
        for label, path in zip(
            ("Reference", "Candidate A", "Candidate B"),
            (reference, delivered, native),
            strict=True,
        ):
            payload, gain = audio_judge.normalized_payload(path, allow_silence=label != "Reference")
            gains.append(gain)
            normalized_hashes.append(hashlib.sha256(base64.b64decode(payload)).hexdigest())
            source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": label},
                        {"type": "input_audio", "input_audio": {"data": payload, "format": "wav"}},
                    ],
                }
            )
        body = {**request["parameters"], "messages": messages}
        request_hash = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
        if (
            request_hash != request["request_sha256"]
            or request_hash != old["request_sha256"]
            or normalized_hashes != request["normalized_audio_sha256"]
            or gains != request["normalization_gains"]
            or gains != old["normalization_gains"]
            or request["rubric"] != audio_judge.RUBRIC
            or request["parameters"]["model"] != audio_judge.MODEL
        ):
            raise JudgeInfrastructureError("Recorded request does not bind the supplied audio")
        responses = sorted(original_receipt.parent.glob(original_receipt.stem + ".response-*.json"))
        matched = False
        for path in responses:
            response = json.loads(path.read_text())
            if response["request_sha256"] != request_hash or any(
                response.get(key) != value for key, value in provenance.items()
            ):
                raise JudgeInfrastructureError("Recorded response provenance does not match")
            if response["http_status"] == 200:
                matched = matched or json.loads(response["body"]) == old["raw"]
        if not matched:
            raise JudgeInfrastructureError("Selected raw response is missing")
        for path in [original_receipt, request_path, *responses]:
            source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path != original_receipt:
                shutil.copyfile(path, destination / path.name)
        calls.append(receipt.stem)
        return {
            key: copy.deepcopy(old[key]) for key in ("raw", "normalization_gains", "request_sha256")
        }

    scored = await audio_judge.score_delivery(
        copy.deepcopy(original["delivery"]),
        destination,
        window_judge=partial(audio_judge.judge_window, comparison_request=recorded_comparison),
    )
    changes, means = [], []
    for stem in scored["delivery"]["stems"]:
        if not stem["reference_audio"]["audible"]:
            continue
        windows = []
        for judgment in stem["judgments"]:
            score = min(judgment["delivery"]["score"], judgment["native"]["score"])
            if score != judgment["score"]:
                raise JudgeInfrastructureError("Window minimum does not match")
            windows.append(Fraction(score))
        mean = sum(windows, Fraction()) / len(windows) if windows else Fraction()
        if float(mean) != stem["timbre_score"]:
            raise JudgeInfrastructureError("Stem arithmetic does not match")
        means.append(mean)
    timbre = sum(means, Fraction()) / len(means)
    weighted = (
        Fraction(1, 5) * Fraction(scored["delivery"]["delivery_score"]) + Fraction(4, 5) * timbre
    )
    if abs(float(weighted) - scored["weighted_score"]) > 1e-12:
        raise JudgeInfrastructureError("Weighted arithmetic does not match")
    for label, (_, old) in expected.items():
        updated = json.loads((destination / f"{label}.json").read_text())
        if updated.get("raw") != old.get("raw"):
            raise JudgeInfrastructureError("Replay changed the recorded raw judgment")
        if updated["score"] != old["score"]:
            changes.append({"receipt": label, "before": old["score"], "after": updated["score"]})
    if any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash
        for path, expected_hash in source_hashes.items()
    ):
        raise JudgeInfrastructureError("Original evidence changed during replay")
    (destination / "scored.json").write_text(json.dumps(scored, indent=2) + "\n")
    report = {
        "kind": "deterministic_evaluator_only_correction",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "original_score": original["weighted_score"],
        "weighted_score": scored["weighted_score"],
        "exact_weighted_score": str(weighted),
        "timbre_score": scored["timbre_score"],
        "exact_timbre_score": str(timbre),
        "delivery_score": scored["delivery"]["delivery_score"],
        "weights": scored["weights"],
        "recorded_responses_used": len(calls),
        "api_calls": 0,
        "native_rerenders": 0,
        "new_solver_runs": 0,
        "changed_windows": changes,
        "original_evidence_unchanged": True,
        "source_sha256": source_hashes,
        "production_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[3])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (Path(audio_judge.__file__).resolve(), Path(__file__).resolve())
        },
    }
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    result = asyncio.run(replay_evaluation(arguments.source, arguments.destination))
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}))
