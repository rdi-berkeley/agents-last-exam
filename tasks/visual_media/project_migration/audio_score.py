"""Score delivered and independently rendered stems without audio model calls."""

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from tasks.utils.evaluation import JudgeInfrastructureError
    from tasks.visual_media.project_migration.scripts import audio_metrics
    from tasks.visual_media.project_migration.scripts.equivalence import compare_pcm
else:
    import audio_metrics
    from equivalence import compare_pcm

    JudgeInfrastructureError = RuntimeError


RUBRIC = Path(__file__).with_name("rubric.txt").read_text()


def score_window(reference: Path, delivered: Path, native: Path, receipt: Path) -> dict:
    try:
        paths = {"reference": reference, "delivery": delivered, "native": native}
        decoded = {label: audio_metrics.read_audio(path) for label, path in paths.items()}
        before, before_rate = decoded["reference"]
        if np.all(before == before[0]):
            raise ValueError("Reference contains only silence or constant DC")
        provenance = {
            "protocol": audio_metrics.VERSION,
            "algorithm_sha256": hashlib.sha256(
                Path(audio_metrics.__file__).read_bytes()
            ).hexdigest(),
            "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
            **{
                label + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                for label, path in paths.items()
            },
        }
        components = {}
        for label in ("delivery", "native"):
            after, after_rate = decoded[label]
            if np.all(after == after[0]):
                components[label] = {"score": 0.0, "route": "silent_candidate"}
                continue
            if before_rate == after_rate and np.array_equal(before, after):
                components[label] = {"score": 1.0, "route": "exact_decoded_pcm"}
                continue
            proof = compare_pcm(reference, paths[label], allow_float_quantization=True)
            if proof["equivalent"]:
                components[label] = {"score": 1.0, "route": "analytic_quantization", "guard": proof}
            else:
                components[label] = audio_metrics.compare_audio(
                    before, before_rate, after, after_rate
                )
        delivery_samples, delivery_rate = decoded["delivery"]
        native_samples, native_rate = decoded["native"]
        if delivery_rate == native_rate and np.array_equal(delivery_samples, native_samples):
            components["native"] = {**components["delivery"], "reused_identical_audio": True}
        elif components["native"]["route"] != "silent_candidate":
            proof = compare_pcm(delivered, native, allow_float_quantization=True)
            if proof["equivalent"]:
                components["native"] = {
                    **components["delivery"],
                    "reused_analytic_equivalence": True,
                    "delivery_native_equivalence": proof,
                }
        result = {
            **provenance,
            **components,
            "score": min(component["score"] for component in components.values()),
        }
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        raise JudgeInfrastructureError(f"Deterministic audio evaluation failed: {exc}") from exc


async def score_delivery(delivery: dict, evidence: Path) -> dict:
    scores = []
    for stem_index, stem in enumerate(delivery["stems"]):
        if not stem["reference_audio"]["audible"]:
            continue
        windows = []
        if stem["valid"]:
            if not stem["passages"]:
                raise JudgeInfrastructureError("Active reference has no fixed passages")
            for index, passage in enumerate(stem["passages"]):
                if not passage.get("native_path"):
                    raise JudgeInfrastructureError("Evaluator-native audio passage is missing")
                windows.append(
                    await asyncio.to_thread(
                        score_window,
                        Path(passage["reference_path"]),
                        Path(passage["candidate_path"]),
                        Path(passage["native_path"]),
                        evidence / f"audio-{stem_index:03d}-{index}.json",
                    )
                )
        score = sum(window["score"] for window in windows) / len(windows) if windows else 0.0
        scores.append(score)
        stem["timbre_score"] = score
        stem["judgments"] = windows
    if not scores:
        raise JudgeInfrastructureError("No active reference stems")
    timbre = sum(scores) / len(scores)
    return {
        "protocol": audio_metrics.VERSION,
        "delivery": delivery,
        "timbre_score": timbre,
        "weighted_score": 0.2 * delivery["delivery_score"] + 0.8 * timbre,
        "weights": [0.2, 0.8],
    }
