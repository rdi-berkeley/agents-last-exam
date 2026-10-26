from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.orchestration import judge_preflight
from ale_run.orchestration.experiment_spec import EnvironmentSpec, ExperimentSpec, OutputSpec
from ale_run.orchestration.runner import Runner
from tasks.utils import evaluation


def test_selected_task_requirements_are_deduplicated():
    text = "engineering/humanoid_wbc_policy_evaluation"
    vision = "business_finance/saas_onepager_brand_refresh_instance_1"
    ordinary = "business_finance/sec_10k_financial_parsing"
    assert judge_preflight.judge_requirements([text, text, ordinary]) == {"text": [text]}
    assert judge_preflight.judge_requirements([ordinary]) == {}
    assert judge_preflight.judge_requirements([vision, vision]) == {
        "text": [vision], "vision": [vision],
    }


def test_digital_marketing_report_only_needs_text_judge():
    task = "business_finance/digital_marketing_ab_test_analysis_1"
    assert judge_preflight.judge_requirements([task]) == {"text": [task]}


@pytest.mark.parametrize("task", [
    "business_finance/saas_onepager_brand_refresh_instance_1",
    "engineering/2d_drawings_to_3d_building_model",
    "engineering/2d_drawings_to_3d_bridge_model",
    "engineering/humanoid_wbc_policy_evaluation",
    "engineering/robotics_blender_tabletop_reconstruction",
    "life_sciences/spatial_transcriptomics_spatial_domain_identification",
    "other/mota_exploration",
    "visual_media/chroma_key_from_reference",
    "visual_media/high_to_low_modeling",
    "visual_media/mota_reproduction",
    "visual_media/music_transcription",
    "visual_media/object_generation",
    "visual_media/project_migration",
    "visual_media/scene_restoration",
    "visual_media/skeletal_animation_reproduction",
    "visual_media/uv_reproduction",
    "visual_media/video_storyboard_001",
])
def test_existing_judge_tasks_are_discovered(task):
    assert task in judge_preflight.judge_requirements([task])["text"]


@pytest.mark.asyncio
async def test_preflight_probes_text_and_actual_image(monkeypatch):
    requests = []

    async def reply(*, content, max_tokens):
        requests.append(content)
        if len(content) > 1:
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
            assert '"red"' not in content[0]["text"]
            return {"color": "Red", "explanation": "uniform square"}
        return {"sum": 15, "explanation": "arithmetic"}

    monkeypatch.setattr(evaluation, "llm_multimodal_json", reply)
    unit = SimpleNamespace(task_path="business_finance/saas_onepager_brand_refresh_instance_1")
    await judge_preflight.check_judges([unit, unit])
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_no_judge_task_needs_no_api(monkeypatch):
    probe = AsyncMock(side_effect=AssertionError("must not call API"))
    monkeypatch.setattr(evaluation, "llm_multimodal_json", probe)
    await judge_preflight.check_judges([SimpleNamespace(task_path="demo/hello")])
    probe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [RuntimeError("secret-key must not leak"), {"sum": 0}, {}])
async def test_failed_probe_prevents_entire_batch_from_starting(monkeypatch, tmp_path, reply):
    if isinstance(reply, Exception):
        probe = AsyncMock(side_effect=reply)
    else:
        probe = AsyncMock(return_value=reply)
    monkeypatch.setattr(evaluation, "llm_multimodal_json", probe)
    launch = AsyncMock()
    monkeypatch.setattr("ale_run.orchestration.lifecycle.run_one_unit", launch)
    monkeypatch.setattr("ale_run.orchestration.host_preflight.check_host_dependencies", lambda _: None)
    spec = ExperimentSpec(
        name="preflight-test", output=OutputSpec(root=str(tmp_path)),
        environment=EnvironmentSpec(), agents=[], tasks=[],
    )
    units = [SimpleNamespace(task_path="engineering/humanoid_wbc_policy_evaluation")]
    with pytest.raises(evaluation.JudgeInfrastructureError, match="no sandboxes") as caught:
        await Runner(spec).run(units)
    assert "secret-key" not in str(caught.value)
    launch.assert_not_called()
    assert not (tmp_path / spec.name).exists()


@pytest.mark.asyncio
async def test_vision_failure_is_not_replaced_by_text_success(monkeypatch):
    async def reply(*, content, max_tokens):
        if len(content) > 1:
            raise RuntimeError("image API unsupported")
        return {"sum": 15}

    monkeypatch.setattr(evaluation, "llm_multimodal_json", reply)
    unit = SimpleNamespace(task_path="business_finance/saas_onepager_brand_refresh_instance_1")
    with pytest.raises(evaluation.JudgeInfrastructureError, match="vision preflight failed"):
        await judge_preflight.check_judges([unit])
