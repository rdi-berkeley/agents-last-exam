from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import zipfile

import httpx
from openai import OpenAI
from PIL import Image, ImageDraw
import pytest

from tasks.business_finance.saas_onepager_brand_refresh_instance_1 import main
from tasks.business_finance.saas_onepager_brand_refresh_instance_1.scripts import (
    score_onepager_v2 as scorer,
)
from tasks.utils import evaluation

score_output_v2 = main.score_output_v2
ConstantJudge = scorer.ConstantJudge
SharedVlmJudge = main.SharedVlmJudge
NS = scorer.NS

NUMERIC = {
    "kpis": [
        {"label": "Task automation rate", "value": "97.4%"},
        {"label": "Manual work saved", "value": "6.5 hrs"},
        {"label": "Annualized margin lift", "value": "$412k"},
    ],
    "pricing": [
        {"plan": "Starter", "price": "$99 / store / month"},
        {"plan": "Growth", "price": "$249 / store / month"},
        {"plan": "Enterprise", "price": "Custom"},
    ],
}
TEXT = {
    "title": "Operational clarity for every retail team.",
    "subtitle": "NorthstarOS connects labor, inventory, and store execution into one "
    "operating layer for multi-site retail teams.",
    "feature_titles": [
        "Workforce orchestration",
        "Demand-aware automation",
        "Audit-ready execution",
    ],
    "testimonial_quote": "We cut cross-store coordination time in half within six weeks.",
    "testimonial_attribution": "Maya Chen, VP of Store Operations, Alder & Ash",
    "cta_button": "Book the 20-min walkthrough",
    "section_label": "Growth snapshot",
    "chart_title": "Store throughput trend",
}


def element(parent, qualified, **attributes):
    prefix, name = qualified.split(":")
    return ET.SubElement(parent, f"{{{NS[prefix]}}}{name}", attributes)


def write_pptx(
    path,
    *,
    renamed=False,
    wrong_chart=False,
    raster=False,
    slides=1,
    chart_title="Workflow completion trend",
):
    presentation = ET.Element(f"{{{NS['p']}}}presentation")
    element(presentation, "p:sldSz", cx="1000", cy="1000")
    slide = ET.Element(f"{{{NS['p']}}}sld")
    tree = element(element(slide, "p:cSld"), "p:spTree")

    def transform(parent, tag, box):
        position = element(parent, tag)
        left, top, width, height = map(str, box)
        element(position, "a:off", x=left, y=top)
        element(position, "a:ext", cx=width, cy=height)

    if raster:
        picture = element(tree, "p:pic")
        transform(element(picture, "p:spPr"), "a:xfrm", (0, 0, 1000, 1000))
    else:
        phrases = scorer.phrase_list_from_expected_text({**TEXT, "chart_title": chart_title})
        phrases += [value for row in NUMERIC["kpis"] + NUMERIC["pricing"] for value in row.values()]
        if renamed:
            phrases[phrases.index("Annualized margin lift")] = "Margin lift"
            phrases.append("annualized estimate")
        for index, phrase in enumerate(phrases):
            shape = element(tree, "p:sp")
            transform(element(shape, "p:spPr"), "a:xfrm", (10, 10 + index * 30, 400, 20))
            paragraph = element(element(shape, "p:txBody"), "a:p")
            element(element(paragraph, "a:r"), "a:t").text = phrase
        for top in (100, 220, 340, 460):
            picture = element(tree, "p:pic")
            transform(element(picture, "p:spPr"), "a:xfrm", (500, top, 200, 100))
        table = element(tree, "p:graphicFrame")
        transform(table, "p:xfrm", (500, 800, 400, 100))
        element(element(table, "a:graphic"), "a:graphicData", uri="fixture/table")
        chart = element(tree, "p:graphicFrame")
        transform(chart, "p:xfrm", (500, 600, 250, 100))
        data = element(element(chart, "a:graphic"), "a:graphicData", uri="fixture/chart")
        element(data, "c:chart", **{f"{{{NS['r']}}}id": "chart1"})

    relationships = ET.Element(f"{{{NS['rel']}}}Relationships")
    element(relationships, "rel:Relationship", Id="chart1", Target="../charts/chart1.xml")
    chart_root = ET.Element(f"{{{NS['c']}}}chartSpace")
    series = element(chart_root, "c:ser")
    element(element(series, "c:tx"), "c:v").text = "throughput_index"
    categories = element(series, "c:cat")
    values = element(series, "c:val")
    for month, value in [("Jan", 61), ("Feb", 70 if wrong_chart else 68)]:
        element(categories, "c:v").text = month
        element(values, "c:v").text = str(value)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", ET.tostring(presentation))
        for index in range(slides):
            archive.writestr(f"ppt/slides/slide{index + 1}.xml", ET.tostring(slide))
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", ET.tostring(relationships))
        archive.writestr("ppt/charts/chart1.xml", ET.tostring(chart_root))


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "output"
    reference = tmp_path / "reference"
    inputs = tmp_path / "input"
    for folder in (output, reference, inputs):
        folder.mkdir()
    image = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 400, 900), fill="navy")
    for folder in (output, reference):
        image.save(folder / "edited_onepager.png")
        write_pptx(folder / "edited_onepager.pptx")
    original = Image.new("RGB", (1000, 1000), "black")
    ImageDraw.Draw(original).rectangle((0, 0, 400, 900), fill="white")
    original.save(inputs / "original_onepager.png")
    regions = {label: (10, 10, 200, 100) for label in scorer.REGION_RUBRIC}
    regions.update(
        hero_render=(500, 100, 200, 100),
        chart_panel=(500, 600, 250, 100),
        pricing_table=(500, 800, 400, 100),
    )
    data = {
        "expected_text_fields.json": TEXT,
        "expected_numeric_fields.json": NUMERIC,
        "expected_chart_data.json": {
            "x": ["Jan", "Feb"],
            "y": [61, 68],
            "series_name": "Throughput",
        },
        "structural_constraints.json": {
            "min_slide_count": 1,
            "max_slide_count": 1,
            "min_text_shapes": 14,
            "min_image_shapes": 4,
            "min_table_shapes": 1,
            "min_chart_shapes": 1,
            "max_single_image_coverage_ratio": 0.5,
            "max_total_image_coverage_ratio": 0.5,
            "min_image_box_area_ratio": 0.0005,
            "min_table_box_area_ratio": 0.02,
            "min_chart_box_area_ratio": 0.01,
        },
        "evaluation_thresholds.json": {
            "hard_gates": {"min_resolution_ratio_per_dimension": 0.5, "min_png_stddev": 0.01}
        },
        "edited_regions.json": {
            "regions": [
                dict(label=label, x=box[0], y=box[1], w=box[2], h=box[3])
                for label, box in regions.items()
            ]
        },
    }
    for name, value in data.items():
        (reference / name).write_text(json.dumps(value))
    return output, reference, inputs


@pytest.mark.parametrize("renamed,expected", [(False, 1.0), (True, 0.9875)])
def test_exact_kpi_label_receives_credit_but_kimi_card_loses_only_its_share(
    bundle, renamed, expected
):
    output, reference, inputs = bundle
    write_pptx(output / "edited_onepager.pptx", renamed=renamed)
    result = score_output_v2(output, reference, ConstantJudge(), inputs)
    assert result["scores"]["numeric"] == pytest.approx(round(11 / 12, 4) if renamed else 1)
    assert result["final_score"] == pytest.approx(expected)
    assert result["vlm_used"] is True
    assert result["pass"] is True


def test_visually_wrong_delivery_does_not_receive_full_credit(bundle):
    output, reference, inputs = bundle
    result = score_output_v2(output, reference, ConstantJudge(False), inputs)
    assert result["scores"]["region_visual"] == 0
    assert result["final_score"] == pytest.approx(0.6)
    assert result["pass"] is False


def test_incorrect_chart_keeps_its_weighted_penalty(bundle):
    output, reference, inputs = bundle
    write_pptx(output / "edited_onepager.pptx", wrong_chart=True)
    result = score_output_v2(output, reference, ConstantJudge(), inputs)
    assert result["scores"]["chart"] == 0
    assert result["final_score"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "chart_title,expected_text,expected_score",
    [
        ("Workflow completion trend", 1, 1),
        ("Store throughput trend", 0.9, 0.982),
        ("Quarterly revenue", 0.9, 0.982),
        ("", 0.9, 0.982),
    ],
)
def test_original_chart_title_overrides_hidden_replacement_without_free_wrong_title_credit(
    bundle, chart_title, expected_text, expected_score
):
    output, reference, inputs = bundle
    assert json.loads((reference / "expected_text_fields.json").read_text())["chart_title"] == (
        "Store throughput trend"
    )
    write_pptx(output / "edited_onepager.pptx", chart_title=chart_title)
    result = score_output_v2(output, reference, ConstantJudge(), inputs)
    assert result["scores"]["text"] == expected_text
    assert result["scores"]["chart"] == result["scores"]["numeric"] == 1
    assert result["final_score"] == pytest.approx(expected_score)


@pytest.mark.parametrize("chart_reply,expected_score", [("YES", 1), ("NO", 0.975)])
def test_chart_visual_rubric_preserves_original_title_and_honors_negative_answer(
    bundle, monkeypatch, chart_reply, expected_score
):
    prompts = []

    def judge_reply(**kwargs):
        prompt = kwargs["content"][0]["text"]
        prompts.append(prompt)
        if scorer.REGION_RUBRIC["chart_panel"] in prompt:
            assert "title 'Workflow completion trend'" in prompt
            assert "preserving the original one-pager title" in prompt
            assert "Do not require the reference title 'Store throughput trend'" in prompt
            assert len([item for item in kwargs["content"] if item["type"] == "image_url"]) == 2
            return chart_reply
        return "YES"

    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", judge_reply)
    output, reference, inputs = bundle
    result = score_output_v2(output, reference, SharedVlmJudge(), inputs)
    assert len(prompts) == 12
    assert result["vlm_detail"]["region"]["chart_panel"] is (chart_reply == "YES")
    assert result["scores"]["text"] == 1
    assert result["final_score"] == pytest.approx(expected_score)


@pytest.mark.parametrize("extra_regions,expected_score", [(False, 0.73), (True, 0.78)])
def test_previous_rerun_visual_answers_keep_their_penalties_after_title_fix(
    bundle, extra_regions, expected_score
):
    accepted = {"header_brand", "footer_band"}
    if extra_regions:
        accepted.update({"hero_render", "cta_button_and_qr"})

    class RecordedAnswersJudge:
        def judge(self, submission, reference, question):
            return question == scorer.GLOBAL_RUBRIC["polish"] or any(
                question == scorer.REGION_RUBRIC[label] for label in accepted
            )

    output, reference, inputs = bundle
    result = score_output_v2(output, reference, RecordedAnswersJudge(), inputs)
    assert result["scores"]["text"] == result["scores"]["numeric"] == 1
    assert result["scores"]["region_visual"] == (0.4 if extra_regions else 0.2)
    assert result["scores"]["tone"] == 0
    assert result["final_score"] == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "corrupt_png",
        "blank",
        "small",
        "corrupt_pptx",
        "raster",
        "zero_slide",
        "multi_slide",
        "unchanged",
    ],
)
def test_required_delivery_gates_still_zero_invalid_submissions(bundle, defect):
    output, reference, inputs = bundle
    png = output / "edited_onepager.png"
    pptx = output / "edited_onepager.pptx"
    if defect == "missing":
        pptx.unlink()
    elif defect == "corrupt_png":
        png.write_bytes(b"not a PNG")
    elif defect == "blank":
        Image.new("RGB", (1000, 1000), "white").save(png)
    elif defect == "small":
        Image.new("RGB", (100, 100), "white").save(png)
    elif defect == "corrupt_pptx":
        pptx.write_bytes(b"not a PPTX")
    elif defect == "raster":
        write_pptx(pptx, raster=True)
    elif defect == "zero_slide":
        write_pptx(pptx, slides=0)
    elif defect == "multi_slide":
        write_pptx(pptx, slides=2)
    else:
        png.write_bytes((inputs / "original_onepager.png").read_bytes())
    result = score_output_v2(output, reference, ConstantJudge(), inputs)
    assert result["final_score"] == 0
    assert result["pass"] is False


def test_absent_visual_judge_never_renormalizes_deterministic_credit(bundle):
    output, reference, inputs = bundle
    with pytest.raises(evaluation.JudgeInfrastructureError, match="requires a visual judge"):
        score_output_v2(output, reference, None, inputs)


@pytest.mark.parametrize("reply", ["maybe", "", "{broken", {"final_score": 0}, None])
def test_malformed_text_judge_reply_is_not_converted_to_no(monkeypatch, reply):
    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", lambda **kwargs: reply)
    judge = SharedVlmJudge()
    image = Image.new("RGB", (2, 2))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="non-binary"):
        judge.judge(image, image, "Is the requested edit correct?")


@pytest.mark.parametrize("failure", [OSError("offline"), ValueError("parse error")])
def test_text_judge_request_failures_propagate(monkeypatch, failure):
    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", fail)
    image = Image.new("RGB", (2, 2))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="request failed"):
        SharedVlmJudge().judge(image, image, "Is the requested edit correct?")


def test_failed_vote_is_not_discarded_after_a_valid_vote(monkeypatch):
    replies = iter(["YES", "uncertain", "YES"])
    monkeypatch.setattr(evaluation, "llm_multimodal_text_sync", lambda **kwargs: next(replies))
    image = Image.new("RGB", (2, 2))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="non-binary"):
        SharedVlmJudge(votes=3).judge(image, image, "Is the requested edit correct?")


def test_missing_judge_credentials_do_not_fall_back_to_solver_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "solver-only-key")
    image = Image.new("RGB", (2, 2))
    with pytest.raises(evaluation.JudgeInfrastructureError, match="request failed") as failure:
        SharedVlmJudge().judge(image, image, "Is the requested edit correct?")
    assert "JUDGE_API_KEY" in str(failure.value.__cause__)


class LocalSession:
    async def file_exists(self, path):
        return Path(path).is_file()

    async def directory_exists(self, path):
        return Path(path).is_dir()

    async def read_bytes(self, path):
        return Path(path).read_bytes()


@pytest.fixture
def task_cfg(bundle):
    output, reference, inputs = bundle
    metadata = {
        "variant_name": "test",
        "output_png": str(output / "edited_onepager.png"),
        "output_pptx": str(output / "edited_onepager.pptx"),
        "original_png": str(inputs / "original_onepager.png"),
    }
    for key, filename in {
        "reference_png": "edited_onepager.png",
        "reference_pptx": "edited_onepager.pptx",
        "reference_regions": "edited_regions.json",
        "reference_text": "expected_text_fields.json",
        "reference_numeric": "expected_numeric_fields.json",
        "reference_chart": "expected_chart_data.json",
        "reference_constraints": "structural_constraints.json",
        "reference_thresholds": "evaluation_thresholds.json",
    }.items():
        metadata[key] = str(reference / filename)
    return SimpleNamespace(metadata=metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "renamed,visual,expected", [(False, True, 1), (True, True, 0.9875), (False, False, 0.6)]
)
async def test_main_returns_fractional_score_instead_of_pass_flag(
    bundle, task_cfg, monkeypatch, renamed, visual, expected
):
    write_pptx(bundle[0] / "edited_onepager.pptx", renamed=renamed)
    monkeypatch.setattr(main, "SharedVlmJudge", lambda: ConstantJudge(visual))
    assert await main.evaluate(task_cfg, LocalSession()) == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["constructor", "region", "global"])
async def test_image_judge_failure_reaches_main_without_fallback(task_cfg, monkeypatch, stage):
    class FailingJudge:
        def judge(self, submission, reference, question):
            if stage == "region" or question in scorer.GLOBAL_RUBRIC.values():
                raise evaluation.JudgeInfrastructureError("image judge failed")
            return True

    def factory():
        if stage == "constructor":
            raise OSError("judge unavailable")
        return FailingJudge()

    monkeypatch.setattr(main, "SharedVlmJudge", factory)
    with pytest.raises(evaluation.JudgeInfrastructureError):
        await main.evaluate(task_cfg, LocalSession())


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["region", "global"])
async def test_missing_boolean_judge_answer_reaches_main(task_cfg, monkeypatch, stage):
    class InvalidJudge:
        def judge(self, submission, reference, question):
            return None if stage == "region" or question in scorer.GLOBAL_RUBRIC.values() else True

    monkeypatch.setattr(main, "SharedVlmJudge", InvalidJudge)
    with pytest.raises(evaluation.JudgeInfrastructureError, match="boolean"):
        await main.evaluate(task_cfg, LocalSession())


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["reference_numeric", "reference_png", "original_png"])
async def test_missing_evaluator_inputs_are_not_agent_zeros(task_cfg, monkeypatch, key):
    Path(task_cfg.metadata[key]).unlink()
    monkeypatch.setattr(main, "SharedVlmJudge", lambda: ConstantJudge())
    with pytest.raises(evaluation.JudgeInfrastructureError):
        await main.evaluate(task_cfg, LocalSession())


@pytest.mark.asyncio
async def test_unreadable_reference_is_not_agent_zero(task_cfg, monkeypatch):
    Path(task_cfg.metadata["reference_png"]).write_bytes(b"broken evaluator reference")
    monkeypatch.setattr(main, "SharedVlmJudge", lambda: ConstantJudge())
    with pytest.raises(evaluation.JudgeInfrastructureError):
        await main.evaluate(task_cfg, LocalSession())


@pytest.mark.asyncio
async def test_missing_reference_region_does_not_receive_implicit_full_credit(
    task_cfg, monkeypatch
):
    Path(task_cfg.metadata["reference_regions"]).write_text('{"regions": []}')
    monkeypatch.setattr(main, "SharedVlmJudge", lambda: ConstantJudge())
    with pytest.raises(evaluation.JudgeInfrastructureError, match="reference region"):
        await main.evaluate(task_cfg, LocalSession())


@pytest.mark.asyncio
async def test_artifact_transport_failure_is_not_an_agent_zero(task_cfg):
    class UnavailableSession(LocalSession):
        async def read_bytes(self, path):
            raise OSError("transport unavailable")

    with pytest.raises(evaluation.JudgeInfrastructureError, match="evaluation artifacts"):
        await main.evaluate(task_cfg, UnavailableSession())


def test_shared_judge_uses_canonical_configuration_and_preserves_image_rubric(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "test-judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.test/v1")
    monkeypatch.setenv("JUDGE_MODEL", "test-judge-model")
    monkeypatch.setenv("OPENAI_API_KEY", "solver-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://solver.test/v1")
    calls = []

    def handle(request):
        assert str(request.url) == "https://judge.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-judge-key"
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-judge-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "NO"},
                    }
                ],
            },
        )

    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(evaluation, "OpenAI", client)
    image = Image.new("RGB", (2, 2))
    question = scorer.REGION_RUBRIC["header_brand"]
    assert SharedVlmJudge().judge(image, image, question) is False
    assert calls[0]["model"] == "test-judge-model"
    content = calls[0]["messages"][0]["content"]
    assert scorer.VLM_PROMPT_CONTEXT in content[0]["text"]
    assert question in content[0]["text"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
