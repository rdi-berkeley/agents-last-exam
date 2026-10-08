import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_judge


@pytest.fixture
def pair(tmp_path):
    paths = [tmp_path / name for name in ("reference.wav", "delivery.wav", "native.wav")]
    for path, frequency in zip(paths, (440, 660, 880), strict=True):
        sf.write(path, 0.2 * np.sin(2 * np.pi * frequency * np.arange(4410) / 44100), 44100)
    return paths


def reply():
    component = {
        "assessable": True,
        "rating": 0,
        "candidate_character": "different",
        "agreements": [],
        "differences": ["different instrument"],
        "limitations": [],
    }
    return {
        "reference_character": "sustained",
        "candidate_a": component,
        "candidate_b": dict(component),
    }


@pytest.mark.asyncio
async def test_invalid_json_retries_once_and_retains_both_bodies(pair, tmp_path):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        content = "not JSON" if len(requests) == 1 else json.dumps(reply())
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    result = await audio_judge.judge_window(
        *pair, tmp_path / "judge.json", transport=httpx.MockTransport(handle)
    )
    assert result["score"] == 0
    assert len(requests) == 2 and requests[0] == requests[1]
    responses = [json.loads(path.read_text()) for path in tmp_path.glob("judge.response-*.json")]
    assert len(responses) == 2
    assert any("not JSON" in response["body"] for response in responses)


@pytest.mark.asyncio
async def test_repeated_invalid_json_is_infrastructure_not_zero(pair, tmp_path):
    with pytest.raises(JudgeInfrastructureError, match="JSONDecodeError"):
        await audio_judge.judge_window(
            *pair,
            tmp_path / "judge.json",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": "not JSON"}}]}
                )
            ),
        )
    assert len(list(tmp_path.glob("judge.response-*.json"))) == 2
    assert "score" not in json.loads((tmp_path / "judge.json").read_text())


@pytest.mark.asyncio
async def test_successful_negative_is_reused_without_request(pair, tmp_path):
    receipt = tmp_path / "judge.json"
    first = await audio_judge.judge_window(
        *pair,
        receipt,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(reply())}}]}
            )
        ),
    )
    retained = receipt.read_bytes()

    def forbidden(request):
        pytest.fail("An evaluator-owned identical judgment must not be requested again")

    result = await audio_judge.judge_window(
        *pair, receipt, transport=httpx.MockTransport(forbidden)
    )
    assert result["score"] == first["score"] == 0
    assert result["cached_evaluator_judgment"]
    assert receipt.read_bytes() == retained
    changed = json.loads(retained)
    changed["score"] = 1
    receipt.write_text(json.dumps(changed))
    with pytest.raises(JudgeInfrastructureError, match="differs from raw"):
        await audio_judge.judge_window(*pair, receipt, transport=httpx.MockTransport(forbidden))


@pytest.mark.asyncio
async def test_mismatched_evaluator_receipt_is_rejected(pair, tmp_path):
    receipt = tmp_path / "judge.json"
    receipt.write_text(json.dumps({"reference_sha256": "different"}))
    with pytest.raises(JudgeInfrastructureError, match="does not match"):
        await audio_judge.judge_window(*pair, receipt)


@pytest.mark.asyncio
async def test_joint_request_contains_three_inputs_and_minimum(pair, tmp_path):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        judgment = reply()
        judgment["candidate_a"]["rating"] = 3
        judgment["candidate_b"]["rating"] = 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(judgment)}}]}
        )

    result = await audio_judge.judge_window(
        *pair, tmp_path / "joint.json", transport=httpx.MockTransport(handle)
    )
    assert len(requests) == 1
    content = [part for message in requests[0]["messages"][1:] for part in message["content"]]
    assert [part["text"] for part in content if part["type"] == "text"] == [
        "Reference",
        "Candidate A",
        "Candidate B",
    ]
    assert sum(part["type"] == "input_audio" for part in content) == 3
    assert result["delivery"]["score"] == 0.75
    assert result["native"]["score"] == result["score"] == 0.25
    assert (
        json.loads((tmp_path / "joint.request.json").read_text())["request_sha256"]
        == result["request_sha256"]
    )


@pytest.mark.asyncio
async def test_same_candidate_reuses_fixed_delivery_rating_not_best(pair, tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        judgment = reply()
        judgment["candidate_a"]["rating"] = 1
        judgment["candidate_b"]["rating"] = 4
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(judgment)}}]}
        )

    result = await audio_judge.judge_window(
        pair[0], pair[1], pair[1], tmp_path / "same.json", transport=httpx.MockTransport(handle)
    )
    assert len(calls) == 1
    assert result["score"] == result["native"]["score"] == 0.25
    assert result["native"]["reused_identical_audio"]
    assert (
        json.loads(result["raw"]["choices"][0]["message"]["content"])["candidate_b"]["rating"] == 4
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "unassessable", "duplicate_reference"])
async def test_joint_failures_are_unscored(pair, tmp_path, failure):
    requests = []

    def handle(request):
        requests.append(request)
        if failure == "http":
            return httpx.Response(503, text="unavailable")
        judgment = reply()
        if failure == "unassessable":
            judgment["candidate_b"]["assessable"] = False
            judgment["candidate_b"]["rating"] = None
        else:
            judgment["candidate_b"]["reference_character"] = "a conflicting reference"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(judgment)}}]}
        )

    with pytest.raises(JudgeInfrastructureError):
        await audio_judge.judge_window(
            *pair, tmp_path / "failed.json", transport=httpx.MockTransport(handle)
        )
    assert len(requests) == 1
    assert "score" not in json.loads((tmp_path / "failed.json").read_text())
