"""Host-side joint audio evaluation; credentials never enter agent inputs."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import httpx
import numpy as np
import soundfile as sf

from tasks.utils.evaluation import JudgeInfrastructureError, load_eval_env
from tasks.visual_media.project_migration.scripts.equivalence import compare_pcm


RUBRIC = Path(__file__).with_name("rubric.txt").read_text()
MODEL = "gpt-audio-1.5"
PROTOCOL = "reference-candidate-a-candidate-b-v2"


def validate_judgment(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "reference_character",
        "candidate_a",
        "candidate_b",
    }:
        raise JudgeInfrastructureError("Audio judge did not return the joint JSON object")
    if (
        not isinstance(value["reference_character"], str)
        or not value["reference_character"].strip()
    ):
        raise JudgeInfrastructureError("Audio judge omitted reference_character")
    scores = {}
    for label, field in (("delivery", "candidate_a"), ("native", "candidate_b")):
        component = value[field]
        if not isinstance(component, dict) or component.get("assessable") is not True:
            raise JudgeInfrastructureError("Audio judge could not assess the supplied recordings")
        rating = component.get("rating")
        if type(rating) is not int or not 0 <= rating <= 4:
            raise JudgeInfrastructureError("Audio judge returned an invalid ordinal rating")
        if (
            not isinstance(component.get("candidate_character"), str)
            or not component["candidate_character"].strip()
        ):
            raise JudgeInfrastructureError("Audio judge omitted candidate_character")
        for field in ("agreements", "differences", "limitations"):
            if not isinstance(component.get(field), list) or not all(
                isinstance(item, str) for item in component[field]
            ):
                raise JudgeInfrastructureError(f"Audio judge returned invalid {field}")
        if "reference_character" in component:
            raise JudgeInfrastructureError("Audio judge must describe the reference only once")
        scores[label] = rating / 4
    return scores


def silence_reason(samples: np.ndarray) -> str | None:
    if not len(samples) or not np.isfinite(samples).all():
        raise JudgeInfrastructureError("Invalid audio passage")
    if not np.any(samples):
        return "zero_samples"
    if np.all(samples == samples[0]):
        return "constant_per_channel_dc"
    return None


def normalized_payload(path: Path, *, allow_silence=False) -> tuple[str, float]:
    samples, rate = sf.read(path, always_2d=True)
    silence = silence_reason(samples)
    if silence and not allow_silence:
        raise JudgeInfrastructureError("Audio passage contains only zero samples or constant DC")
    peak = float(np.abs(samples).max())
    rms = float(np.sqrt(np.square(samples).mean()))
    gain = min(0.1 / rms, 0.95 / peak) if not silence else 1
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros_like(samples) if silence else samples * gain,
        rate,
        format="WAV",
        subtype="PCM_16",
    )
    return base64.b64encode(buffer.getvalue()).decode(), gain


def deterministic_score(reference: Path, candidate: Path) -> dict:
    before, before_rate = sf.read(reference, always_2d=True)
    after, after_rate = sf.read(candidate, always_2d=True)
    silence = silence_reason(after)
    if silence:
        return {"score": 0.0, "route": "silent_candidate", "reason": silence}
    if before_rate == after_rate and before.shape == after.shape and np.array_equal(before, after):
        return {"score": 1.0, "route": "exact_decoded_pcm"}
    guard = compare_pcm(reference, candidate)
    if guard["equivalent"]:
        return {"score": 1.0, "route": "analytic_quantization", "guard": guard}
    return {"guard": guard}


async def request_comparison(reference, delivered, native, receipt, provenance, *, transport=None):
    load_eval_env(override=True)
    key = os.environ.get("OPENAI_API_KEY")
    if not key and transport is None:
        raise JudgeInfrastructureError("Official audio judge requires evaluator OPENAI_API_KEY")
    messages = [{"role": "user", "content": RUBRIC}]
    gains = []
    payload_hashes = []
    for label, path in (
        ("Reference", reference),
        ("Candidate A", delivered),
        ("Candidate B", native),
    ):
        data, gain = normalized_payload(path, allow_silence=label != "Reference")
        gains.append(gain)
        payload_hashes.append(hashlib.sha256(base64.b64decode(data)).hexdigest())
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": label},
                    {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                ],
            }
        )
    body = {
        "model": MODEL,
        "modalities": ["text"],
        "temperature": 0,
        "max_completion_tokens": 1600,
        "store": False,
        "messages": messages,
    }
    request_bytes = json.dumps(body, separators=(",", ":")).encode()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    request_record = {
        **provenance,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "normalization_gains": gains,
        "normalized_audio_sha256": payload_hashes,
        "parameters": {key: value for key, value in body.items() if key != "messages"},
        "rubric": RUBRIC,
        "labels": ["Reference", "Candidate A", "Candidate B"],
    }
    receipt.with_suffix(".request.json").write_text(json.dumps(request_record, indent=2) + "\n")
    try:
        for attempt in range(2):
            async with httpx.AsyncClient(timeout=120, transport=transport) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    content=request_bytes,
                )
            response_receipt = receipt.with_name(
                receipt.stem + ".response-" + uuid.uuid4().hex + ".json"
            )
            response_receipt.write_text(
                json.dumps(
                    {
                        **provenance,
                        "attempt": attempt + 1,
                        "http_status": response.status_code,
                        "request_sha256": request_record["request_sha256"],
                        "body": response.text,
                    },
                    indent=2,
                )
                + "\n"
            )
            response.raise_for_status()
            try:
                raw = response.json()
                judgment = json.loads(raw["choices"][0]["message"]["content"])
            except json.JSONDecodeError:
                if attempt == 0:
                    continue
                raise
            validate_judgment(judgment)
            return {
                "raw": raw,
                "normalization_gains": gains,
                "request_sha256": request_record["request_sha256"],
            }
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        JudgeInfrastructureError,
    ) as exc:
        receipt.write_text(
            json.dumps({**provenance, "infrastructure_error": type(exc).__name__}, indent=2) + "\n"
        )
        raise JudgeInfrastructureError(
            f"Official audio judge failed: {type(exc).__name__}"
        ) from exc


async def judge_window(
    reference: Path,
    delivered: Path,
    native: Path,
    receipt: Path,
    *,
    transport=None,
    comparison_request=request_comparison,
) -> dict:
    provenance = {
        "protocol": PROTOCOL,
        "model": MODEL,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "delivery_sha256": hashlib.sha256(delivered.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    before, _ = sf.read(reference, always_2d=True)
    if silence_reason(before):
        raise JudgeInfrastructureError(
            "Reference passage contains only zero samples or constant DC"
        )
    cached = None
    if receipt.is_file():
        cached = json.loads(receipt.read_text())
        if any(cached.get(key) != value for key, value in provenance.items()):
            raise JudgeInfrastructureError("Audio receipt does not match inputs/model/rubric")
    components = {
        "delivery": deterministic_score(reference, delivered),
        "native": deterministic_score(reference, native),
    }
    identical = delivered.read_bytes() == native.read_bytes()
    decoded_identity = False
    if not identical:
        delivery_samples, delivery_rate = sf.read(delivered, always_2d=True)
        native_samples, native_rate = sf.read(native, always_2d=True)
        decoded_identity = bool(
            delivery_rate == native_rate
            and delivery_samples.shape == native_samples.shape
            and np.isfinite(delivery_samples).all()
            and np.isfinite(native_samples).all()
            and np.array_equal(delivery_samples, native_samples)
        )
        identical = decoded_identity
    equivalence = (
        None if identical else compare_pcm(delivered, native, allow_float_quantization=True)
    )
    reuse = identical or equivalence["equivalent"]
    needs_judge = "score" not in components["delivery"] or (
        not reuse and "score" not in components["native"]
    )
    response = {}
    if needs_judge:
        if cached and "raw" in cached:
            response = {
                key: cached[key] for key in ("raw", "normalization_gains", "request_sha256")
            }
        else:
            response = await comparison_request(
                reference,
                delivered,
                native,
                receipt,
                provenance,
                transport=transport,
            )
        judgment = json.loads(response["raw"]["choices"][0]["message"]["content"])
        scores = validate_judgment(judgment)
        for label, component in components.items():
            if "score" not in component:
                component.update(score=scores[label], route="audio_semantic_judge")
    if reuse:
        components["native"] = {**components["delivery"]}
        if identical:
            components["native"]["reused_identical_audio"] = True
            if decoded_identity:
                components["native"]["reused_exact_decoded_pcm"] = True
        else:
            components["native"].update(
                reused_analytic_equivalence=True, delivery_native_equivalence=equivalence
            )
    result = {
        **provenance,
        **response,
        **components,
        "score": min(component["score"] for component in components.values()),
    }
    if cached and "raw" in cached:
        if any(cached.get(label) != result[label] for label in ("score", "delivery", "native")):
            raise JudgeInfrastructureError(
                "Audio receipt score differs from raw judgment or deterministic proof"
            )
        return {**result, "cached_evaluator_judgment": True}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return result


async def score_delivery(delivery: dict, evidence: Path, *, window_judge=judge_window) -> dict:
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
                    await window_judge(
                        Path(passage["reference_path"]),
                        Path(passage["candidate_path"]),
                        Path(passage["native_path"]),
                        evidence / f"judge-{stem_index:03d}-{index}.json",
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
        "delivery": delivery,
        "timbre_score": timbre,
        "weighted_score": 0.2 * delivery["delivery_score"] + 0.8 * timbre,
        "weights": [0.2, 0.8],
    }
