import asyncio
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI, BadRequestError, OpenAI
import pytest

from tasks.utils import evaluation


def completion(content='{"score":1}', *, finish="stop", refusal=None):
    return {
        "id": "test-reply",
        "created": 1,
        "model": "test-judge",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


@pytest.fixture
def judge_endpoint(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.test/v1")
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("OPENAI_API_KEY", "solver-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://solver.test/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://other-solver.test/v1")


def install_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def sync_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=transport))

    monkeypatch.setattr(evaluation, "OpenAI", sync_client)

    def async_client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=transport)
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(evaluation, "AsyncOpenAI", async_client)


def test_private_configuration_is_lazy_and_never_replaces_solver_environment(monkeypatch, tmp_path):
    directory = Path(os.environ["AGENTHLE_EVAL_CREDENTIALS_DIR"])
    directory.mkdir()
    path = directory / "judge.env"
    path.write_text(
        "JUDGE_API_KEY=private-key\nJUDGE_BASE_URL=https://private.test/v1\nJUDGE_MODEL=private-model\nOPENAI_API_KEY=must-not-export\n"
    )
    (directory / "ignored.env.example").write_text("JUDGE_API_KEY=wrong\n")
    monkeypatch.setenv("OPENAI_API_KEY", "solver-key")
    before = dict(os.environ)
    spec = importlib.util.spec_from_file_location("private_judge_import", evaluation.__file__)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    assert dict(os.environ) == before
    assert evaluation.judge_config()["api_key"] == "private-key"
    assert evaluation.resolve_llm_judge_model() == "private-model"
    path.write_text(path.read_text().replace("private-model", "new-model"))
    assert evaluation.resolve_llm_judge_model() == "new-model"
    assert dict(os.environ) == before
    from ale_run.orchestration.lifecycle import _collect_env_passthrough

    forwarded = _collect_env_passthrough()
    assert forwarded["OPENAI_API_KEY"] == "solver-key"
    assert not any("JUDGE" in name or "EVAL_CREDENTIALS" in name for name in forwarded)
    assert "private-key" not in forwarded.values()


def test_missing_judge_key_never_uses_solver_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "solver-key")
    with pytest.raises(evaluation.JudgeInfrastructureError, match="JUDGE_API_KEY"):
        evaluation.llm_multimodal_text_sync(content="Score this")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_routing_images_roles_and_parameters(monkeypatch, judge_endpoint, asynchronous):
    calls = []
    monkeypatch.setenv("JUDGE_REASONING_EFFORT", "high")

    def handle(request):
        assert str(request.url) == "https://judge.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer judge-key"
        assert (
            "openai-organization" not in request.headers
            or not request.headers["openai-organization"]
        )
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=completion("YES"))

    install_transport(monkeypatch, handle)
    content = evaluation.build_vision_content("Exact rubric", [b"image-one", b"image-two"])
    kwargs = dict(content=content, system_prompt="Exact system rubric", max_tokens=5)
    result = (
        asyncio.run(evaluation.llm_multimodal_text(**kwargs))
        if asynchronous
        else evaluation.llm_multimodal_text_sync(**kwargs)
    )
    assert result == "YES"
    assert calls[0]["messages"] == [
        {"role": "system", "content": "Exact system rubric"},
        {"role": "user", "content": content},
    ]
    assert calls[0]["model"] == "test-judge"
    assert calls[0]["max_completion_tokens"] >= 4096
    assert calls[0]["reasoning_effort"] == "high"
    assert "max_tokens" not in calls[0]
    assert os.environ["OPENAI_API_KEY"] == "solver-key"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_supported_parameter_adaptation_is_bounded(monkeypatch, judge_endpoint, asynchronous):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        for parameter in ("temperature", "max_completion_tokens", "response_format"):
            if parameter in body:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": f"Unsupported parameter: '{parameter}'",
                            "param": parameter,
                            "code": "unsupported_parameter",
                        }
                    },
                )
        return httpx.Response(200, json=completion())

    install_transport(monkeypatch, handle)
    kwargs = dict(content=[{"type": "text", "text": "Return JSON score"}])
    result = (
        asyncio.run(evaluation.llm_multimodal_json(**kwargs))
        if asynchronous
        else evaluation.llm_multimodal_json_sync(**kwargs)
    )
    assert result == {"score": 1}
    assert len(calls) == 4
    assert calls[-1]["max_tokens"] == 2048
    assert all(body["messages"] == calls[0]["messages"] for body in calls)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "replies,success", [(["bad", '{"score":1}'], True), (["[]", "bad"], False)]
)
def test_json_retry_preserves_rubric_and_never_returns_zero(
    monkeypatch, judge_endpoint, asynchronous, replies, success
):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion(replies[len(calls) - 1]))

    install_transport(monkeypatch, handle)
    kwargs = dict(content=[{"type": "text", "text": "Exact JSON scoring rubric"}])

    def run():
        return (
            asyncio.run(evaluation.llm_multimodal_json(**kwargs))
            if asynchronous
            else evaluation.llm_multimodal_json_sync(**kwargs)
        )

    if success:
        assert run() == {"score": 1}
    else:
        with pytest.raises(evaluation.JudgeInfrastructureError, match="invalid JSON twice"):
            run()
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize(
    "finish,content,refusal",
    [("length", '{"score":1}', None), ("stop", "", None), ("stop", '{"score":1}', "refused")],
)
def test_unusable_completions_are_infrastructure(
    monkeypatch, judge_endpoint, finish, content, refusal
):
    install_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, json=completion(content, finish=finish, refusal=refusal)
        ),
    )
    with pytest.raises(evaluation.JudgeInfrastructureError):
        evaluation.llm_multimodal_json_sync(content="JSON score")


def test_unrelated_bad_request_is_not_retried(monkeypatch, judge_endpoint):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            400, json={"error": {"message": "Invalid image data", "param": "messages"}}
        )

    install_transport(monkeypatch, handle)
    with pytest.raises(BadRequestError):
        evaluation.llm_multimodal_json_sync(content="Return JSON")
    assert len(calls) == 1


def test_audio_configuration_shares_gateway_but_keeps_audio_model(judge_endpoint, monkeypatch):
    monkeypatch.setenv("JUDGE_REASONING_EFFORT", "high")
    config = evaluation.judge_config(capability="audio")
    assert config["model"] == evaluation.DEFAULT_AUDIO_JUDGE_MODEL
    assert config["base_url"] == "https://judge.test/v1"
    assert config["api_key"] == "judge-key"
    assert not config["reasoning_effort"]
    monkeypatch.setenv("AUDIO_JUDGE_MODEL", "custom-audio-model")
    assert evaluation.judge_config(capability="audio")["model"] == "custom-audio-model"


@pytest.mark.asyncio
async def test_unsupported_audio_never_switches_provider_or_model(judge_endpoint):
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == "https://judge.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer judge-key"
        assert json.loads(request.content)["model"] == evaluation.DEFAULT_AUDIO_JUDGE_MODEL
        return httpx.Response(400, json={"error": {"message": "Audio model is not supported"}})

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "wav"}},
            ],
        }
    ]
    with pytest.raises(BadRequestError):
        await evaluation.llm_judge_completion(
            messages=messages,
            capability="audio",
            transport=httpx.MockTransport(handle),
        )
    assert len(calls) == 1


def test_robotics_rubric_and_score_are_preserved(monkeypatch, judge_endpoint):
    from tasks.engineering.robotics_blender_tabletop_reconstruction import main

    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=completion(
                json.dumps(
                    dict(
                        object_count=1,
                        spatial_layout=0.5,
                        camera_perspective=1,
                        material_fidelity=0.5,
                        lighting_quality=1,
                        table_surface=0.5,
                    )
                )
            ),
        )

    install_transport(monkeypatch, handle)
    assert main._call_vlm_judge("reference", "candidate") == 0.75
    content = calls[0]["messages"][0]["content"]
    assert content[0]["text"] == main._VLM_PROMPT
    assert content[2]["image_url"]["url"].endswith("reference")
    assert content[4]["image_url"]["url"].endswith("candidate")


@pytest.mark.asyncio
async def test_gemini_video_reads_private_key_and_model_without_export(monkeypatch):
    genai = pytest.importorskip("google.genai")
    directory = Path(os.environ["AGENTHLE_EVAL_CREDENTIALS_DIR"])
    directory.mkdir()
    (directory / "gemini.env").write_text(
        "GEMINI_API_KEY=private-video-key\nGEMINI_EVAL_MODEL=private-video-model\n"
        "GEMINI_AUTH_MODE=developer_api\n"
    )
    monkeypatch.setenv("GEMINI_API_KEY", "solver-video-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    before = dict(os.environ)
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"score":1}')

    def client(**kwargs):
        assert kwargs == {"api_key": "private-video-key"}
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", client)
    result = await evaluation.gemini_video_json_judge(
        prompt="Exact video rubric", video_bytes=b"video"
    )
    assert result["ok"] is True
    assert result["model_used"] == "private-video-model"
    assert result["parsed"] == {"score": 1}
    assert calls[0]["contents"][0].text == "Exact video rubric"
    assert calls[0]["contents"][1].inline_data.data == b"video"
    assert dict(os.environ) == before


def test_vertex_private_adc_is_passed_directly(monkeypatch):
    import google.auth

    directory = Path(os.environ["AGENTHLE_EVAL_CREDENTIALS_DIR"])
    directory.mkdir()
    (directory / "gemini.env").write_text(
        "GEMINI_AUTH_MODE=vertex_ai\nGOOGLE_CLOUD_PROJECT=private-project\n"
        "GOOGLE_CLOUD_LOCATION=us-central1\nGOOGLE_APPLICATION_CREDENTIALS=/private/adc.json\n"
    )
    credential = object()
    calls = []

    def load_credentials(path, *, scopes):
        calls.append((path, scopes))
        return credential, "private-project"

    monkeypatch.setattr(google.auth, "load_credentials_from_file", load_credentials)
    before = dict(os.environ)
    auth = evaluation._resolve_gemini_auth()
    assert auth["client_kwargs"]["credentials"] is credential
    assert auth["client_kwargs"]["project"] == "private-project"
    assert calls == [("/private/adc.json", ["https://www.googleapis.com/auth/cloud-platform"])]
    assert dict(os.environ) == before


def test_lowpoly_report_records_shared_model(monkeypatch, tmp_path, judge_endpoint):
    from tasks.visual_media.high_to_low_modeling.scripts.run_lowpoly_evaluation import (
        LowpolyBenchmark,
    )

    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    calls = []

    def score(bundle, *, model, temperature):
        calls.append((model, temperature))
        return {"judge_score": 75, "verdict": "pass"}

    monkeypatch.setattr(LowpolyBenchmark, "openai_judge", score)
    report = LowpolyBenchmark.run_judge(
        judge_bundle_path=bundle,
        output_dir=tmp_path / "judgment",
        backend="openai",
        model=None,
        temperature=0,
    )
    assert calls == [("test-judge", 0)]
    assert report["summary"]["model"] == "test-judge"
    assert report["summary"]["judge_score"] == 0.75
