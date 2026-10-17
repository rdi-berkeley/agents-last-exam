import json
import tempfile
from functools import partial
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zipfile import ZipFile

import httpx
import pytest
from openai import AuthenticationError

from tasks.utils import evaluation
from tasks.visual_media.video_storyboard_001 import main as storyboard
from tasks.visual_media.video_storyboard_001.scripts import evaluate_storyboard
from tasks.visual_media.skeletal_animation_reproduction import main as skeletal
from tasks.visual_media.skeletal_animation_reproduction.scripts import local_soft_eval


def docx(text):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{evaluate_storyboard.WORD_NS}">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return stream.getvalue()


@pytest.mark.asyncio
async def test_storyboard_authentication_failure_propagates(monkeypatch):
    error = AuthenticationError(
        "IP not authorized",
        response=httpx.Response(401, request=httpx.Request("POST", "https://judge.test/v1")),
        body={"error": "IP not authorized"},
    )
    monkeypatch.setattr(evaluate_storyboard, "llm_multimodal_json", AsyncMock(side_effect=error))
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        read_bytes=AsyncMock(return_value=docx("Candidate or question")),
    )
    config = SimpleNamespace(metadata={"output_file": "candidate.docx", "question_file": "questions.docx"})
    with pytest.raises(AuthenticationError):
        await storyboard.evaluate(config, session)


@pytest.mark.asyncio
async def test_missing_official_brief_is_not_agent_zero():
    session = SimpleNamespace(file_exists=AsyncMock(return_value=False))
    config = SimpleNamespace(metadata={"output_file": "candidate.docx", "question_file": "questions.docx"})
    with pytest.raises(FileNotFoundError, match="official question"):
        await storyboard.evaluate(config, session)


@pytest.mark.asyncio
async def test_invalid_brief_errors_but_invalid_submission_scores_zero():
    with pytest.raises(RuntimeError, match="official question"):
        await evaluate_storyboard.score_storyboard_docx(storyboard_docx=b"bad", question_docx=b"bad")
    result = await evaluate_storyboard.score_storyboard_docx(
        storyboard_docx=b"bad", question_docx=docx("Official questions")
    )
    assert result["score"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("answers,score", [(evaluate_storyboard.ANSWER_KEY, 1.0),
    ({key: "UNKNOWN" for key in evaluate_storyboard.QUESTION_IDS}, 0.0)])
async def test_storyboard_valid_judgments_keep_scores(monkeypatch, answers, score):
    judge = AsyncMock(return_value={"answers": answers})
    monkeypatch.setattr(evaluate_storyboard, "llm_multimodal_json", judge)
    result = await evaluate_storyboard.score_storyboard_docx(
        storyboard_docx=docx("Shots"), question_docx=docx("Questions")
    )
    assert result["score"] == score
    judge.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}, {"answers": {}}, {"answers": {"Q1": "A"}},
    {"answers": {key: "A or B" for key in evaluate_storyboard.QUESTION_IDS}}])
async def test_storyboard_incomplete_or_malformed_answers_error(monkeypatch, payload):
    monkeypatch.setattr(evaluate_storyboard, "llm_multimodal_json", AsyncMock(return_value=payload))
    with pytest.raises(evaluation.JudgeInfrastructureError):
        await evaluate_storyboard.score_storyboard_docx(
            storyboard_docx=docx("Shots"), question_docx=docx("Questions")
        )


@pytest.mark.parametrize("response", ["", "I cannot judge this image.", "YES or NO"])
def test_invalid_binary_reply_is_an_error(monkeypatch, response):
    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", Mock(return_value=response))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="non-binary"):
        evaluation.llm_multimodal_binary_questions_sync(
            prompt_context="Compare", questions=["Matches?"], content=[]
        )


def test_valid_binary_answers_are_not_retried(monkeypatch):
    judge = Mock(side_effect=["NO", "YES", "NO"])
    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", judge)
    result = evaluation.llm_multimodal_binary_questions_sync(
        prompt_context="Compare", questions=["One?", "Two?", "Three?"], content=[]
    )
    assert result["final_score"] == pytest.approx(1 / 3)
    assert judge.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [0.5, -1, float("nan"), float("inf")])
async def test_checklist_nonbinary_numbers_do_not_become_pass_or_fail(monkeypatch, answer):
    monkeypatch.setattr(evaluation, "llm_vision_json_judge", AsyncMock(return_value={"answers": {"item": answer}}))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="non-binary"):
        await evaluation.llm_vision_binary_checklist_judge(
            prompt_intro="Compare", checklist_items=[("item", "Matches?")], image_bytes=b"image"
        )


def test_judge_error_log_has_null_score(monkeypatch):
    context = evaluation.EvaluationContext("test", auto_save=False)
    context.log_error("image", RuntimeError("unavailable"))
    assert context.evaluation_details["evaluations"][0]["score"] is None


def test_skeletal_missing_key_does_not_produce_a_score(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    image = tmp_path / "sheet.png"
    image.write_bytes(b"image")
    with pytest.raises(evaluation.JudgeInfrastructureError, match="JUDGE_API_KEY"):
        local_soft_eval.run_local_soft_eval(
            reference_sheet=image, preview_sheet=image, replay_sheet=image, pose_state_sheet=image
        )
    assert not (tmp_path / "soft_eval_report.json").exists()


@pytest.fixture
def skeletal_episode(monkeypatch, tmp_path):
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        read_file=AsyncMock(side_effect=lambda path: json.dumps(
            {"sample_count": 1} if path.endswith("evaluation_config.json")
            else {"validity_gate_passed": True, "sample_positions": []}
        )),
        read_bytes=AsyncMock(return_value=b"video"),
        write_file=AsyncMock(),
        interface=SimpleNamespace(create_dir=AsyncMock()),
    )
    monkeypatch.setattr(skeletal, "TemporaryDirectory", partial(tempfile.TemporaryDirectory, dir=tmp_path))
    monkeypatch.setattr(skeletal, "_download_reference_package", AsyncMock(return_value=(tmp_path, tmp_path / "ref.mp4")))
    monkeypatch.setattr(skeletal, "extract_reference_frames", Mock(return_value=[]))
    monkeypatch.setattr(skeletal, "reference_framing_targets", Mock(return_value={}))
    monkeypatch.setattr(skeletal, "_upload_scripts", AsyncMock())
    monkeypatch.setattr(skeletal, "_launch_remote_job", AsyncMock())
    monkeypatch.setattr(skeletal, "_wait_for_file", AsyncMock(return_value=True))
    monkeypatch.setattr(skeletal, "compute_local_metrics", Mock(return_value={
        "validity_gate_passed": True, "video_match_score": 1.0,
        "replay_consistency_score": 1.0, "minimal_skeleton_score": 1.0,
    }))
    monkeypatch.setattr(skeletal, "write_eval_bundle", Mock(return_value={
        key: tmp_path / "sheet.png" for key in
        ("reference_sheet", "preview_sheet", "replay_sheet", "pose_state_sheet")
    }))
    config = SimpleNamespace(metadata=skeletal._build_config(skeletal.VARIANTS[0]).to_metadata())
    return config, session


@pytest.mark.asyncio
async def test_skeletal_judge_failure_never_uses_machine_score(monkeypatch, skeletal_episode):
    monkeypatch.setattr(skeletal, "run_local_soft_eval", Mock(side_effect=evaluation.JudgeInfrastructureError("offline")))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="offline"):
        await skeletal.evaluate(*skeletal_episode)


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_score,expected", [(0.0, 0.85), (1.0, 1.0)])
async def test_skeletal_successful_judge_keeps_weights(monkeypatch, skeletal_episode, judge_score, expected):
    monkeypatch.setattr(skeletal, "run_local_soft_eval", Mock(return_value=judge_score))
    assert await skeletal.evaluate(*skeletal_episode) == [pytest.approx(expected)]
