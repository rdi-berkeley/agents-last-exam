import importlib
import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


main = importlib.import_module("tasks.engineering.2d_drawings_to_3d_building_model.main")
spec = importlib.util.spec_from_file_location(
    "building_floors", main.SCRIPTS_DIR / "detect_floors.py"
)
floors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(floors)
judge = importlib.import_module(main.evaluate_renders.__module__)


def test_linux_native_contract():
    task = main.load()[0]
    assert task.computer["setup_config"]["os_type"] == "linux"
    assert task.metadata["output_blend"].endswith("/model.blend")
    assert "1 mm" in task.description
    assert "CPU" in task.description
    assert "Rhino" not in task.description
    assert task.metadata["reference_renders_dir"].endswith("/reference_renders_linux")


def test_equivalent_obj_indices_detect_the_same_floor(tmp_path):
    vertices = "v 0 0 0\nv 100 0 0\nv 100 100 0\nv 0 100 0\n"
    absolute = tmp_path / "absolute.obj"
    relative = tmp_path / "relative.obj"
    absolute.write_text(vertices + "f 1 2 3 4\n")
    relative.write_text(vertices + "f -4 -3 -2 -1\n")
    assert floors.detect_floors(str(absolute)) == floors.detect_floors(str(relative))


@pytest.mark.parametrize("unit_factor", [1.0, 0.001])
def test_reference_tower_cut_does_not_follow_dominant_ground_slab(unit_factor):
    peaks = [
        (-49.9, 1427526303.0),
        (2950.7, 87426077.5),
        (3850.9, 193032997.1),
        (5951.4, 110099081.7),
        (11652.7, 902717331.8),
        (12652.9, 763021596.3),
        (33257.6, 71479861.8),
    ]
    scaled_peaks = [(height * unit_factor, weight) for height, weight in peaks]
    original = floors.select_plan_cuts(
        -300 * unit_factor, scaled_peaks, floor_offset_mm=1500 * unit_factor
    )
    corrected = floors.select_plan_cuts(
        -300 * unit_factor,
        scaled_peaks,
        floor_offset_mm=1500 * unit_factor,
        tower_cut_height=17537.68754987531 * unit_factor,
    )
    assert original["tower_typical"] == pytest.approx(4450.7 * unit_factor)
    assert corrected == {
        **original,
        "tower_typical": pytest.approx(17537.68754987531 * unit_factor),
    }
    assert corrected["tower_typical"] - 400 * unit_factor > 12600 * unit_factor


@pytest.mark.parametrize("indices", ["0 2 3", "1 2 99", "-4 -2 -1"])
def test_invalid_obj_indices_are_rejected(tmp_path, indices):
    candidate = tmp_path / "invalid.obj"
    candidate.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf " + indices + "\n")
    with pytest.raises(ValueError):
        floors.parse_obj_mesh(str(candidate))


@pytest.mark.asyncio
async def test_invalid_native_geometry_does_not_wait_for_missing_views():
    class Session:
        async def file_exists(self, path):
            return path.endswith(("exit-code", "native-report.json"))

        async def read_bytes(self, path):
            return b"2" if path.endswith("exit-code") else b'{"valid": false}'

    assert not await main._wait_for_render_outputs(
        Session(), candidate_dir="/render", view_names=["axon_NE"], timeout_sec=1
    )


@pytest.mark.asyncio
async def test_render_requires_successful_exit_and_all_views():
    class Session:
        def __init__(self, missing=False):
            self.missing = missing

        async def file_exists(self, path):
            return not self.missing or not path.endswith("axon_NE.png")

        async def read_bytes(self, path):
            return b"0" if path.endswith("exit-code") else b'{"valid": true}'

    assert await main._wait_for_render_outputs(
        Session(), candidate_dir="/render", view_names=["axon_NE"], timeout_sec=1
    )
    with pytest.raises(RuntimeError, match="did not produce"):
        await main._wait_for_render_outputs(
            Session(True), candidate_dir="/render", view_names=["axon_NE"], timeout_sec=1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "report"),
    [
        (b"127", None),
        (b"124", {"valid": True}),
        (b"1", {"valid": True}),
        (b"0", []),
        (b"0", {"valid": "false"}),
        (b"0", {"valid": False}),
    ],
)
async def test_evaluator_failures_are_not_candidate_zero(status, report):
    class Session:
        async def file_exists(self, path):
            return path.endswith("exit-code") or (
                report is not None and path.endswith("native-report.json")
            )

        async def read_bytes(self, path):
            if path.endswith("exit-code"):
                return status
            if report is None:
                raise FileNotFoundError(path)
            return json.dumps(report).encode()

    with pytest.raises(RuntimeError, match="Building evaluator"):
        await main._wait_for_render_outputs(
            Session(), candidate_dir="/render", view_names=["axon_NE"], timeout_sec=1
        )


@pytest.mark.asyncio
async def test_evaluator_deadline_is_not_candidate_zero():
    with pytest.raises(RuntimeError, match="did not finish"):
        await main._wait_for_render_outputs(
            object(), candidate_dir="/render", view_names=[], timeout_sec=0
        )


@pytest.fixture
def judge_env(monkeypatch):
    for name in (
        "D2T3B_JUDGE_MODEL",
        "LLM_JUDGE_MODEL",
        "D2T3B_JUDGE_API_KEY",
        "OPENAI_API_KEY",
        "D2T3B_JUDGE_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(judge, "dotenv_values", lambda path: {"OPENAI_API_KEY": "fixture-key"})


@pytest.fixture
def render_fixture(tmp_path):
    from PIL import Image

    config = {
        "task_description": "Betonwerk Hall and Tower",
        "judge_questions": list(judge.CRITERION_VIEWS),
        "view_names": list(judge.DEFAULT_VIEW_NAMES),
    }
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(json.dumps(config))
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for directory, color in ((reference, "white"), (candidate, "black")):
        directory.mkdir()
        for view in judge.DEFAULT_VIEW_NAMES:
            Image.new("RGB", (2, 2), color).save(directory / f"{view}.png")
    return reference, candidate, config_path


def install_judge_client(monkeypatch, replies):
    from openai.types.chat import ChatCompletion

    calls = []
    settings = []
    responses = iter(replies)

    class Client:
        def __init__(self, **kwargs):
            settings.append(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            calls.append(kwargs)
            reply = next(responses)
            if isinstance(reply, Exception):
                raise reply
            text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
            return ChatCompletion(
                id=f"reply-{len(calls)}",
                created=1,
                model=kwargs["model"],
                object="chat.completion",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": text},
                    }
                ],
            )

    monkeypatch.setattr(judge, "OpenAI", Client)
    return calls, settings


def test_each_original_criterion_gets_only_relevant_paired_views(
    monkeypatch,
    render_fixture,
    judge_env,
    tmp_path,
):
    calls, settings = install_judge_client(monkeypatch, ["YES", "NO"] * 4)
    path = tmp_path / "judge-report.json"
    report = judge.evaluate_renders(*render_fixture, report_path=path)
    expected = [
        ["axon_NE", "axon_SW"],
        ["section_NS", "section_EW", "elevation_east", "elevation_west"],
        ["plan_hall_ground", "plan_tower_typical", "axon_NE"],
        ["elevation_north", "axon_NW"],
        ["elevation_south", "axon_SE"],
        ["elevation_east", "elevation_west", "axon_NE"],
        ["plan_hall_ground"],
        ["elevation_north", "elevation_south", "elevation_east", "elevation_west"],
    ]
    assert len(calls) == 8
    assert report["score"] == 0.5
    assert report["passed"] is True
    assert report["yes_count"] == report["no_count"] == 4
    assert report["status"] == "completed"
    assert json.loads(path.read_text()) == report
    assert settings[0]["base_url"] == "https://api.openai.com/v1"
    assert "fixture-key" not in path.read_text()
    for request, result, views in zip(calls, report["per_question"], expected, strict=True):
        assert request["model"] == "gpt-6-astra"
        assert "max_tokens" not in request
        assert request["max_completion_tokens"] >= 2048
        content = request["messages"][0]["content"]
        assert result["view_names"] == views
        assert result["question"] in content[0]["text"]
        assert "14 paired" not in content[0]["text"]
        labels = [part["text"] for part in content[1:] if part["type"] == "text"]
        assert labels == [
            f"{label} render — {view}:" for view in views for label in ("Reference", "Candidate")
        ]
        images = [part for part in content if part["type"] == "image_url"]
        assert len(images) == len(views) * 2 <= 8
        assert images[0]["image_url"] != images[1]["image_url"]
        assert result["raw_response"] == result["response"]["choices"][0]["message"]["content"]


def test_reordered_criteria_keep_their_own_views(monkeypatch, render_fixture, judge_env):
    config_path = render_fixture[2]
    config = json.loads(config_path.read_text())
    config["judge_questions"].reverse()
    config_path.write_text(json.dumps(config))
    install_judge_client(monkeypatch, ["YES"] * 8)
    report = judge.evaluate_renders(*render_fixture)
    assert report["score"] == 1.0
    assert report["per_question"][0]["view_names"] == [
        "elevation_north",
        "elevation_south",
        "elevation_east",
        "elevation_west",
    ]
    assert [item["question"] for item in report["per_question"]] == config["judge_questions"]


@pytest.mark.parametrize(
    "mutation",
    ["missing_question", "changed_question", "duplicate_question", "missing_view", "unsafe_view"],
)
def test_invalid_criterion_mapping_is_rejected_before_judging(render_fixture, mutation):
    config_path = render_fixture[2]
    config = json.loads(config_path.read_text())
    if mutation == "missing_question":
        config["judge_questions"].pop()
    elif mutation == "changed_question":
        config["judge_questions"][0] = "Is this beautiful?"
    elif mutation == "duplicate_question":
        config["judge_questions"][0] = config["judge_questions"][1]
    elif mutation == "missing_view":
        config["view_names"].remove("elevation_south")
    else:
        config["view_names"].append("../../other")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="criteria|view_names"):
        judge.evaluate_renders(*render_fixture)


@pytest.mark.parametrize("side", [0, 1])
def test_missing_render_is_an_evaluator_error(
    monkeypatch, render_fixture, judge_env, tmp_path, side
):
    (render_fixture[side] / "elevation_north.png").unlink()
    calls, settings = install_judge_client(monkeypatch, [])
    path = tmp_path / "judge-report.json"
    with pytest.raises(RuntimeError, match="missing required render evidence"):
        judge.evaluate_renders(*render_fixture, report_path=path)
    report = json.loads(path.read_text())
    assert report["status"] == "error"
    assert report["score"] is None
    assert report["missing_views"] == [f"{('reference', 'candidate')[side]}/elevation_north.png"]
    assert calls == settings == []


@pytest.mark.parametrize(
    "reply", ["", "uncertain " * 1000, ("YES", "length"), (None, "content_filter")]
)
def test_full_invalid_reply_is_persisted_without_scoring_zero(
    monkeypatch,
    render_fixture,
    judge_env,
    tmp_path,
    reply,
):
    install_judge_client(monkeypatch, ["YES", reply])
    path = tmp_path / "judge-report.json"
    with pytest.raises(RuntimeError, match="incomplete or non-binary reply for Q2"):
        judge.evaluate_renders(*render_fixture, report_path=path)
    report = json.loads(path.read_text())
    assert report["score"] is None
    assert report["status"] == "error"
    assert report["yes_count"] == 1
    assert report["no_count"] == 0
    expected = reply[0] if isinstance(reply, tuple) else reply
    assert report["per_question"][1]["raw_response"] == expected
    assert report["per_question"][1]["response"]["choices"][0]["message"]["content"] == expected


def test_checkpoint_survives_a_later_api_failure(monkeypatch, render_fixture, judge_env, tmp_path):
    install_judge_client(monkeypatch, ["YES", RuntimeError("route unavailable")])
    path = tmp_path / "judge-report.json"
    snapshots = []
    with pytest.raises(RuntimeError, match="route unavailable"):
        judge.evaluate_renders(
            *render_fixture,
            report_path=path,
            on_report=lambda report: snapshots.append(json.loads(json.dumps(report))),
        )
    report = json.loads(path.read_text())
    assert report == snapshots[-1]
    assert report["status"] == "error"
    assert report["score"] is None
    assert report["per_question"][0]["raw_response"] == "YES"
    assert any(snapshot["yes_count"] == 1 for snapshot in snapshots[:-1])


def test_judge_configuration_uses_working_secret_file_without_global_mutation(
    monkeypatch, judge_env
):
    paths = []

    def secret_values(path):
        paths.append(path)
        return {"OPENAI_API_KEY": "secret-fixture", "OPENAI_API_BASE": "https://secret.test/v1"}

    monkeypatch.setattr(judge, "dotenv_values", secret_values)
    assert judge._judge_settings() == {
        "model": "gpt-6-astra",
        "api_key": "secret-fixture",
        "base_url": "https://secret.test/v1",
    }
    assert paths[0] == main.SCRIPTS_DIR.resolve().parents[3] / "secret" / ".env"
    monkeypatch.setenv("LLM_JUDGE_MODEL", "shared-model")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.test/v1")
    assert judge._judge_settings() == {
        "model": "shared-model",
        "api_key": "environment-fixture",
        "base_url": "https://environment.test/v1",
    }
    monkeypatch.setenv("D2T3B_JUDGE_MODEL", "task-model")
    monkeypatch.setenv("D2T3B_JUDGE_API_KEY", "task-fixture")
    monkeypatch.setenv("D2T3B_JUDGE_BASE_URL", "https://task.test/v1")
    assert judge._judge_settings() == {
        "model": "task-model",
        "api_key": "task-fixture",
        "base_url": "https://task.test/v1",
    }
    assert judge._judge_settings("explicit-model")["model"] == "explicit-model"


def test_judge_requires_credentials(monkeypatch, judge_env):
    monkeypatch.setattr(judge, "dotenv_values", lambda path: {})
    with pytest.raises(RuntimeError, match="requires .*API_KEY"):
        judge._judge_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_task_persists_full_judge_checkpoints_in_guest(monkeypatch, tmp_path, fails):
    config = main.DrawingsTo3DBuildingConfig()
    task = SimpleNamespace(metadata=config.to_metadata())
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=True),
        read_bytes=AsyncMock(return_value=b"{}"),
        write_file=AsyncMock(),
        run_command=AsyncMock(return_value={"stdout": ""}),
        interface=SimpleNamespace(create_dir=AsyncMock()),
    )
    for name in (
        "_reset_remote_dir",
        "_upload_render_scripts",
        "_launch_remote_blender_render",
        "_download_render_pngs",
    ):
        monkeypatch.setattr(main, name, AsyncMock())
    monkeypatch.setattr(main, "_resolve_remote_blender", AsyncMock(return_value="/blender"))
    monkeypatch.setattr(main, "_wait_for_render_outputs", AsyncMock(return_value=True))
    monkeypatch.setattr(main.tempfile, "tempdir", str(tmp_path))
    reply = "long reply " * 1000
    checkpoint = {
        "status": "in_progress",
        "score": None,
        "yes_count": 1,
        "question_count": 8,
        "per_question": [{"raw_response": reply}],
    }

    def evaluate_renders(**kwargs):
        kwargs["on_report"](checkpoint)
        if fails:
            raise RuntimeError("later judge request failed")
        final = {**checkpoint, "status": "completed", "score": 0.125}
        kwargs["on_report"](final)
        return final

    monkeypatch.setattr(main, "evaluate_renders", evaluate_renders)
    if fails:
        with pytest.raises(RuntimeError, match="later judge request failed"):
            await main.evaluate(task, session)
    else:
        assert await main.evaluate(task, session) == [0.125]
    writes = session.write_file.call_args_list
    assert writes[0].args[0].endswith("/base/judge-report.json")
    assert json.loads(writes[0].args[1]) == checkpoint
    assert json.loads(writes[-1].args[1])["per_question"][0]["raw_response"] == reply
