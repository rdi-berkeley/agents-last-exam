"""VLM judge payload schema enforcement for robotics_blender.

The judge scores agent-produced renders at 45% of final_score. Before this
change the score averaged whatever keys the judge returned with no
required-key or range check: a single {"overall": 1.0} scored 1.0, and a 7.0
value propagated to final_score 3.15.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tasks.engineering.robotics_blender_tabletop_reconstruction import main as task_main
from tasks.utils.evaluation import JudgeInfrastructureError


def _full_valid() -> dict:
    return {k: 1.0 for k in task_main._VLM_DIMENSIONS}


def test_full_valid_payload_scores_one():
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=_full_valid()):
        assert task_main._call_vlm_judge("r", "c") == 1.0


def test_partial_payload_is_rejected():
    payload = {"object_count": 1.0}
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=payload):
        with pytest.raises(JudgeInfrastructureError, match="missing required keys"):
            task_main._call_vlm_judge("r", "c")


def test_wrong_single_key_payload_is_rejected():
    payload = {"overall": 1.0}
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=payload):
        with pytest.raises(JudgeInfrastructureError):
            task_main._call_vlm_judge("r", "c")


def test_out_of_range_value_is_rejected():
    payload = _full_valid()
    payload["object_count"] = 7.0
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=payload):
        with pytest.raises(JudgeInfrastructureError, match="out of range"):
            task_main._call_vlm_judge("r", "c")


def test_negative_value_is_rejected():
    payload = _full_valid()
    payload["lighting_quality"] = -0.5
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=payload):
        with pytest.raises(JudgeInfrastructureError, match="out of range"):
            task_main._call_vlm_judge("r", "c")


def test_extra_key_payload_is_rejected():
    payload = _full_valid()
    payload["overall"] = 1.0
    with patch.object(task_main, "llm_multimodal_json_sync", return_value=payload):
        with pytest.raises(JudgeInfrastructureError, match="unexpected keys"):
            task_main._call_vlm_judge("r", "c")
