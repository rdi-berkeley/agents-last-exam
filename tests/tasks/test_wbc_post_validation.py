from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import pytest

from tasks.engineering.humanoid_wbc_policy_evaluation import main as wbc_task
from tasks.engineering.humanoid_wbc_policy_evaluation.scripts import score_outputs as scorer


RELEASE = Path("/home/allennie/ale-overall/release-staging/ale-task-fixes-20260913")
DATA = RELEASE / "data-v1.1/files/engineering/humanoid_wbc_policy_evaluation/base"
OVERLAY = (
    RELEASE
    / "post-validation-20260918/data-overrides/engineering"
    / "humanoid_wbc_policy_evaluation/base"
)
CASE05 = "case_05_lowstairs_0dk7n238"
CASE07 = "case_07_lowstairs_cyyp71b5"
CASE08 = "case_08_fall_getup_flat_os6wfu5w"


def test_non_exhaustive_evidence_summary_preserves_material_outcome_rules(monkeypatch):
    calls = []

    def capture_judge(**kwargs):
        calls.append(kwargs)
        return {"results": [{"result": "YES"}]}

    monkeypatch.setattr(scorer, "llm_multimodal_binary_questions_sync", capture_judge)
    monkeypatch.setenv("WBC_EVIDENCE_JUDGE_REPEATS", "3")
    case = "case_03_sitting_chair_7zlbjqi4"
    item = {"case_id": case, "evidence": {"observation": "Tips across the chair then falls."}}
    score, _ = scorer._semantic_evidence_score([item], {case: "Failed sitting attempt."})
    assert score == 1 and len(calls) == 3
    context = calls[0]["prompt_context"]
    assert "concise, non-exhaustive summary" in context
    assert "unless it reverses the decisive outcome" in context
    assert "stable sitting still contradicts a failed sitting attempt" in context
    assert all(call["content"] == [] for call in calls)
    assert scorer.LABEL_WEIGHT == 0.7 and scorer.EVIDENCE_WEIGHT == 0.3
    probe = RELEASE / "post-validation-20260918/task-reviews-r06/wbc/judge-candidate/call-007.json"
    if probe.exists():
        recorded = json.loads(probe.read_text())
        assert recorded["scorer_api_kwargs"]["prompt_context"] == context


@pytest.fixture
def reference_path():
    path = OVERLAY / "reference/expected_verdicts.json"
    return path if path.exists() else DATA / "reference/expected_verdicts.json"


def test_public_categories_disambiguate_final_step_delay():
    path = OVERLAY / "input/task_brief.md"
    brief = " ".join((path if path.exists() else DATA / "input/task_brief.md").read_text().split())
    prompt = " ".join(wbc_task.HumanoidWbcPolicyEvaluationConfig().task_description.split())
    card = json.loads(Path(wbc_task.__file__).with_name("task_card.json").read_text())
    for text in (brief, prompt, " ".join(card["taskPrompt"].split())):
        assert "A slight delay or offset confined to the last few steps" in text
        assert "does not by itself make a rollout nearly_successful" in text
        assert "persistent tracking deficit beyond this final-step exception" in text
        assert "An unrecovered fall that prevents completion is failed" in text


def test_case07_reference_uses_visible_tracking_not_unseen_stair_contacts(reference_path):
    frozen = json.loads((DATA / "reference/expected_verdicts.json").read_text())
    corrected = json.loads(reference_path.read_text())
    assert corrected["verdict_enum"] == frozen["verdict_enum"]
    for original, updated in zip(frozen["cases"], corrected["cases"], strict=True):
        assert updated["case_id"] == original["case_id"]
        assert updated["expected_verdict"] == original["expected_verdict"]
        if original["case_id"] != CASE07:
            assert updated == original
            continue
        observation = updated["gold_observation"]
        assert "remains upright" in observation
        assert "positional offset" in observation
        assert "grows toward the end" in observation
        assert "climb the stairs" not in observation
        assert "through the whole process" not in observation


@pytest.fixture
def recorded_run(monkeypatch):
    evidence_path = RELEASE / "validation-checks/wbc-r02-evidence.json"
    if not evidence_path.exists():
        pytest.skip("r02 WBC artifacts unavailable")
    evidence = json.loads(evidence_path.read_text())
    selection = evidence["selected"]["kimi_after"]
    run_path = Path(selection["run_json"])
    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == selection["sha256"]
    output = run_path.parent / "output"
    report = json.loads((output / "policy_evaluation_report.json").read_text())
    by_id = {item["case_id"]: item for item in report["evaluations"]}
    for item in evidence["case_comparison"]:
        assert by_id[item["case_id"]]["evidence"]["observation"] == item["observation"]
    calls = []

    def recorded_vote(**kwargs):
        question = kwargs["questions"][0]
        case_id = question.splitlines()[0].removeprefix("Case ").removesuffix(".")
        assert by_id[case_id]["evidence"]["observation"] in question
        calls.append(question)
        return {"results": [{"result": evidence["original_judge_votes"][case_id]["answer"]}]}

    monkeypatch.setattr(scorer, "llm_multimodal_binary_questions_sync", recorded_vote)
    monkeypatch.setenv("WBC_EVIDENCE_JUDGE_REPEATS", "3")
    return evidence, output, report, calls


def test_actual_r02_videos_and_original_score(recorded_run):
    evidence, output, report, calls = recorded_run
    for video in evidence["videos"]:
        path = Path(video["path"])
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == video["sha256"]
        capture = cv2.VideoCapture(str(path))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == video["frames"]
        finally:
            capture.release()
    assert next(item for item in evidence["videos"] if item["case_id"] == CASE08)["frames"] == 8400
    assert scorer._visual_demo_errors(report, output) == []
    result = scorer.score_report(
        output / "policy_evaluation_report.json", DATA / "reference/expected_verdicts.json", output
    )
    assert result.score == 0.875
    assert len(calls) == 24


@pytest.mark.parametrize(
    ("changed_case", "verdict", "expected_score"),
    [(None, None, 0.875), (CASE05, "successful", 0.9625), (CASE08, "successful", 0.7875)],
)
def test_corrected_reference_plumbing_with_recorded_votes_not_new_semantic_scores(
    recorded_run, reference_path, tmp_path, changed_case, verdict, expected_score
):
    _, output, report, calls = recorded_run
    if changed_case:
        for item in report["evaluations"]:
            if item["case_id"] == changed_case:
                item["verdict"] = verdict
        for label in scorer.VERDICTS:
            report["summary"][label] = sum(
                item["verdict"] == label for item in report["evaluations"]
            )
    replay_path = tmp_path / "report.json"
    replay_path.write_text(json.dumps(report))
    result = scorer.score_report(replay_path, reference_path, output)
    assert result.score == expected_score
    assert len(calls) == 24
    corrected = json.loads(reference_path.read_text())
    case07 = next(item for item in corrected["cases"] if item["case_id"] == CASE07)
    assert case07["gold_observation"] in next(question for question in calls if CASE07 in question)


def test_actual_report_missing_video_still_fails(recorded_run, reference_path, tmp_path):
    _, output, report, calls = recorded_run
    report["evaluations"][0]["evidence"]["visual_demo_path"] = "visual_demos/missing.mp4"
    replay_path = tmp_path / "report.json"
    replay_path.write_text(json.dumps(report))
    result = scorer.score_report(replay_path, reference_path, output)
    assert result.score == 0
    assert any("missing visual demo" in message for message in result.diagnostics)
    assert calls == []
