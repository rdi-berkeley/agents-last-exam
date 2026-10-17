import base64
import io

from PIL import Image, ImageChops, ImageDraw
import pytest

from tasks.business_finance.saas_onepager_brand_refresh_instance_1.scripts import (
    score_onepager_v2 as scorer,
)
from tasks.utils import evaluation


@pytest.fixture
def page_pair(tmp_path):
    submission = Image.new("RGB", (300, 600), "white")
    reference = Image.new("RGB", (300, 600), "ivory")
    for page in [submission, reference]:
        drawing = ImageDraw.Draw(page)
        drawing.rectangle((20, 60, 230, 75), fill="navy")
        drawing.rectangle((20, 95, 280, 105), fill="teal")
        drawing.rectangle((20, 140, 280, 260), fill="silver")
    regions = {label: (10, 50, 160, 65) for label in scorer.REGION_RUBRIC}
    return submission, reference, regions, tmp_path


@pytest.mark.parametrize("size", [(300, 600), (200, 400)])
def test_headline_sees_complete_page_and_other_checks_keep_exact_crops(page_pair, size):
    submission, reference, regions, folder = page_pair
    submission.resize(size, Image.Resampling.LANCZOS).save(folder / "submission.png")
    reference.save(folder / "reference.png")
    normalized = Image.open(folder / "submission.png").resize(
        reference.size, Image.Resampling.LANCZOS
    )
    calls = []

    class InspectingJudge:
        def judge(self, candidate, target, question):
            label = next(
                name for name, rubric in scorer.REGION_RUBRIC.items() if rubric == question
            )
            calls.append(label)
            if label == "headline_subheadline":
                expected_candidate, expected_target = normalized, reference
                assert "not truncated or overlapping" in question
                assert "most spacing relationships" in question
                assert "whole-page images" in question
            else:
                left, top, width, height = regions[label]
                bounds = (left, top, left + width, top + height)
                expected_candidate = normalized.crop(bounds)
                expected_target = reference.crop(bounds)
            assert candidate.size == expected_candidate.size
            assert target.size == expected_target.size
            assert ImageChops.difference(candidate, expected_candidate).getbbox() is None
            assert ImageChops.difference(target, expected_target).getbbox() is None
            return True

    score, details = scorer.vlm_region_score(
        InspectingJudge(), folder / "submission.png", folder / "reference.png", regions
    )
    assert score == 1
    assert calls == list(scorer.REGION_RUBRIC)
    assert all(details.values())


@pytest.mark.parametrize("defect", ["clipped", "overlap"])
def test_actual_text_damage_reaches_shared_judge_and_retains_negative_credit(
    page_pair, monkeypatch, defect
):
    submission, reference, regions, folder = page_pair
    drawing = ImageDraw.Draw(submission)
    if defect == "clipped":
        drawing.rectangle((170, 95, 280, 105), fill="white")
    else:
        drawing.rectangle((20, 95, 280, 105), fill="white")
        drawing.rectangle((20, 68, 280, 78), fill="teal")
    submission.save(folder / "submission.png")
    reference.save(folder / "reference.png")
    headline_calls = []

    def judge_reply(**kwargs):
        content = kwargs["content"]
        if scorer.REGION_RUBRIC["headline_subheadline"] not in content[0]["text"]:
            return "YES"
        headline_calls.append(content)
        images = [item for item in content if item["type"] == "image_url"]
        assert len(images) == 2
        payload = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
        delivered = Image.open(io.BytesIO(payload)).convert("RGB")
        assert delivered.size == submission.size
        assert ImageChops.difference(delivered, submission).getbbox() is None
        assert "most spacing relationships" in content[0]["text"]
        return "NO"

    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", judge_reply)
    score, details = scorer.vlm_region_score(
        scorer.SharedVlmJudge(), folder / "submission.png", folder / "reference.png", regions
    )
    assert len(headline_calls) == 1
    assert score == pytest.approx(0.9)
    assert details["headline_subheadline"] is False
    assert (1 - score) * scorer.WEIGHTS["region_visual"] == pytest.approx(0.025)


def test_headline_context_still_requires_reference_region_metadata(page_pair):
    submission, reference, regions, folder = page_pair
    submission.save(folder / "submission.png")
    reference.save(folder / "reference.png")
    regions.pop("headline_subheadline")
    with pytest.raises(evaluation.JudgeInfrastructureError, match="headline_subheadline"):
        scorer.vlm_region_score(
            scorer.ConstantJudge(), folder / "submission.png", folder / "reference.png", regions
        )
