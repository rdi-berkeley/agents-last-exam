from __future__ import annotations

import pytest

from tasks.business_finance.legal_ma_consistency_audit_01.scripts import score_audit_report
from tasks.engineering.robotics_blender_tabletop_reconstruction.main import _vlm_render_score
from tasks.utils import evaluation
from tasks.visual_media.chroma_key_from_reference.scripts import local_soft_eval as chroma_soft
from tasks.visual_media.uv_reproduction.scripts import local_soft_eval as uv_soft


@pytest.mark.asyncio
async def test_shared_vision_judge_does_not_turn_infrastructure_failure_into_zero(monkeypatch):
    async def fail(**_kwargs):
        raise OSError("judge unavailable")

    monkeypatch.setattr(evaluation, "llm_multimodal_text", fail)

    with pytest.raises(evaluation.JudgeInfrastructureError, match="judge unavailable"):
        await evaluation.llm_vision_judge(
            prompt="judge this",
            image_bytes=b"image",
            return_binary_score=True,
        )


@pytest.mark.asyncio
async def test_shared_vision_judge_rejects_non_binary_response(monkeypatch):
    async def respond(**_kwargs):
        return "maybe"

    monkeypatch.setattr(evaluation, "llm_multimodal_text", respond)

    with pytest.raises(evaluation.JudgeInfrastructureError, match="non-binary"):
        await evaluation.llm_vision_judge(
            prompt="judge this",
            image_bytes=b"image",
            return_binary_score=True,
        )


@pytest.mark.asyncio
async def test_checklist_judge_does_not_turn_infrastructure_failure_into_zero(monkeypatch):
    async def fail(**_kwargs):
        raise RuntimeError("invalid judge response")

    monkeypatch.setattr(evaluation, "llm_vision_json_judge", fail)

    with pytest.raises(evaluation.JudgeInfrastructureError, match="invalid judge response"):
        await evaluation.llm_vision_binary_checklist_judge(
            prompt_intro="judge this",
            checklist_items=[("complete", "Is it complete?")],
            image_bytes=b"image",
        )


@pytest.mark.asyncio
async def test_checklist_judge_rejects_missing_answer(monkeypatch):
    async def respond(**_kwargs):
        return {"answers": {}, "summary": "incomplete"}

    monkeypatch.setattr(evaluation, "llm_vision_json_judge", respond)

    with pytest.raises(evaluation.JudgeInfrastructureError, match="missing checklist answer"):
        await evaluation.llm_vision_binary_checklist_judge(
            prompt_intro="judge this",
            checklist_items=[("complete", "Is it complete?")],
            image_bytes=b"image",
        )


class BrokenRenderSession:
    async def file_exists(self, _path: str) -> bool:
        return True

    async def directory_exists(self, _path: str) -> bool:
        return False

    async def read_bytes(self, _path: str) -> bytes:
        raise OSError("render storage unavailable")


class MilestoneSession:
    async def file_exists(self, _path: str) -> bool:
        return True

    async def directory_exists(self, _path: str) -> bool:
        return False

    async def list_dir(self, _path: str) -> list[str]:
        return ["frame.png"]

    async def read_bytes(self, _path: str) -> bytes:
        return b"image"


@pytest.mark.asyncio
async def test_milestone_mode_does_not_swallow_judge_failure(tmp_path):
    async def fail(*_args):
        raise evaluation.JudgeInfrastructureError("judge unavailable")

    with pytest.raises(evaluation.JudgeInfrastructureError, match="judge unavailable"):
        await evaluation.evaluate_milestone_mode(
            session=MilestoneSession(),
            target_path="output",
            reference_path="reference",
            task_tag="milestone",
            comparison_fn=fail,
            output_dir=str(tmp_path),
        )


@pytest.mark.asyncio
async def test_robotics_vlm_failure_does_not_receive_half_credit():
    with pytest.raises(RuntimeError, match="VLM render comparison failed"):
        await _vlm_render_score(BrokenRenderSession(), "reference.png", "candidate.png")


@pytest.mark.asyncio
async def test_chroma_soft_eval_propagates_judge_failure(monkeypatch):
    async def fail(**_kwargs):
        raise evaluation.JudgeInfrastructureError("judge unavailable")

    monkeypatch.setattr(chroma_soft, "llm_vision_binary_checklist_judge", fail)
    frame = {
        "identifier": "frame-1",
        "index": 1,
        "time_sec": 0.5,
        "input_bytes": b"input",
        "output_bytes": b"output",
    }

    with pytest.raises(evaluation.JudgeInfrastructureError, match="judge unavailable"):
        await chroma_soft.run_soft_eval_local(
            task_tag="chroma",
            frame_items=[frame],
            model="test-model",
        )


def test_uv_soft_eval_distinguishes_missing_frames_from_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(uv_soft, "_load_api_key", lambda: None)

    assert uv_soft.run_local_soft_eval([], tmp_path) == 0.0
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        uv_soft.run_local_soft_eval([{"view": "front"}], tmp_path)


def test_audit_summary_and_negative_checks_are_not_false_positive_claims(monkeypatch):
    monkeypatch.setattr(
        score_audit_report,
        "TARGET_RULES",
        {
            "required": {
                "match": lambda text, _normalized: "required issue" in text.lower(),
                "evidence": lambda _text: True,
                "location": lambda _text: True,
            }
        },
    )
    report = """\
## Finding: required issue
Evidence: confirmed in the source documents.

## Summary
| # | Finding | Category |
| 1 | Required issue | clerical |

## Checks That Passed
- No numerical mismatch found in the registered-capital figures.
- No logical contradiction in the payment schedule.
- Reconciliation complete; arithmetic error not found.
"""

    result = score_audit_report.score_report_text(
        report_text=report,
        reference_payload={"required_findings": [{"id": "required"}]},
    )

    assert result["score"] == 1.0
    assert result["false_positive_count"] == 0


def test_audit_still_penalizes_an_affirmative_unsupported_finding(monkeypatch):
    monkeypatch.setattr(
        score_audit_report,
        "TARGET_RULES",
        {
            "required": {
                "match": lambda text, _normalized: "required issue" in text.lower(),
                "evidence": lambda _text: True,
                "location": lambda _text: True,
            }
        },
    )
    report = """\
## Finding: required issue
Evidence: confirmed in the source documents.

## Finding: unsupported numerical mismatch
Evidence: this additional assertion is not a gold finding.
"""

    result = score_audit_report.score_report_text(
        report_text=report,
        reference_payload={"required_findings": [{"id": "required"}]},
    )

    assert result["score"] == 0.8
    assert result["false_positive_count"] == 1
