import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tarfile
import subprocess
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

from cua_bench.computers.remote import RemoteDesktopSession

from tasks.utils.evaluation import EvaluationContext, JudgeInfrastructureError
from tasks.visual_media.project_migration import audio_judge, main, runtime
from tasks.visual_media.project_migration.scripts.audio_delivery import check_delivery
from tasks.visual_media.project_migration.scripts.project_check import control_curves


ROOT = Path(__file__).resolve().parents[2] / "tasks/visual_media/project_migration"
SELECTED_VARIANT = "celeste_symphonic_suite"
PRIVATE_VARIANTS = (
    "eora",
    "hollow_knight_symphonic_suite",
    "twilight_princess_credits",
    "undertale_medley",
)


def audio(path, frequency=440, seconds=0.2, gain=0.2):
    values = gain * np.sin(2 * np.pi * frequency * np.arange(round(seconds * 44100)) / 44100)
    sf.write(path, np.column_stack([values, values * 0.6]), 44100, subtype="PCM_24")


def test_all_five_linux_registrations():
    registry = json.loads((ROOT / "assets/sources.json").read_text())
    tasks = main.load()
    assert set(registry) == {SELECTED_VARIANT, *PRIVATE_VARIANTS}
    assert len(tasks) == 5
    assert {task.metadata["variant_name"] for task in tasks} == registry.keys()
    for task in tasks:
        assert task.computer["setup_config"]["os_type"] == "linux"
        assert "project/project.ardour" in task.description
        assert "20%" in task.description and "80%" in task.description
        assert "before downstream group or" in task.description
        assert "Keep all original routing and group/master effects" in task.description
        assert "Do not delete, transpose or rearrange music" in task.description
        assert task.metadata["reference_stems_dir"] not in task.description


def assert_unsolved_bundle(entry):
    path = ROOT / "assets" / entry["archive"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
    with tarfile.open(path) as archive:
        names = archive.getnames()
        assert not any("reference" in name.casefold() or name.endswith(".wav") for name in names)
        session = archive.extractfile("project/project.ardour").read()
        assert b"Unavailable:" in session and b"urn:private" not in session


def test_selected_source_bundle_is_present_and_unsolved():
    registry = json.loads((ROOT / "assets/sources.json").read_text())
    entry = registry[SELECTED_VARIANT]
    assert (ROOT / "assets" / entry["archive"]).is_file(), "Selected Celeste bundle is required"
    assert_unsolved_bundle(entry)


@pytest.mark.parametrize("variant", PRIVATE_VARIANTS)
def test_private_source_bundle_integrity_when_provisioned(variant):
    registry = json.loads((ROOT / "assets/sources.json").read_text())
    entry = registry[variant]
    if not (ROOT / "assets" / entry["archive"]).is_file():
        pytest.skip(f"Nonselected {variant} source bundle is provisioned separately")
    assert_unsolved_bundle(entry)


def test_full_delivery_identity_and_missing_denominator(tmp_path):
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    reference.mkdir()
    (output / "stems").mkdir(parents=True)
    for name in ("Flute", "Harp"):
        audio(reference / (name + ".wav"))
    audio(output / "stems/Flute.wav")
    audio(output / "mixdown.wav")
    delivery = check_delivery(output, reference, tmp_path / "passages")
    assert delivery["mix"]["valid"]
    assert delivery["delivery_score"] == 0.5
    for stem in delivery["stems"]:
        for passage in stem["passages"]:
            passage["native_path"] = passage["candidate_path"]
    result = asyncio.run(audio_judge.score_delivery(delivery, tmp_path / "judgments"))
    assert result["timbre_score"] == 0.5
    assert result["weighted_score"] == 0.5
    assert (
        result["delivery"]["stems"][0]["judgments"][0]["delivery"]["route"] == "exact_decoded_pcm"
    )


def test_prefix_is_not_complete_delivery_and_selection_ignores_candidate(tmp_path):
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    reference.mkdir()
    (output / "stems").mkdir(parents=True)
    audio(reference / "Flute.wav", seconds=0.3)
    audio(output / "stems/Flute.wav", seconds=0.1)
    first = check_delivery(output, reference, tmp_path / "one")
    audio(output / "stems/Flute.wav", frequency=120, seconds=0.3)
    second = check_delivery(output, reference, tmp_path / "two")
    assert first["valid_stems"] == 0
    assert second["valid_stems"] == 1
    assert (
        first["stems"][0]["reference_audio"]["passages"]
        == second["stems"][0]["reference_audio"]["passages"]
    )


def test_missing_reference_is_infrastructure(tmp_path):
    with pytest.raises(RuntimeError, match="Reference stems"):
        check_delivery(tmp_path, tmp_path, tmp_path / "evidence")


@pytest.mark.parametrize(
    "reply",
    [
        {"assessable": False, "rating": None},
        {"assessable": True, "rating": True},
        {"assessable": True, "rating": 5},
    ],
)
def test_unassessable_or_invalid_rating_is_not_zero(reply):
    with pytest.raises(JudgeInfrastructureError):
        audio_judge.validate_judgment(reply)


@pytest.mark.asyncio
async def test_official_provider_request_and_negative_not_bypassed(tmp_path):
    reference, candidate = tmp_path / "reference.wav", tmp_path / "candidate.wav"
    audio(reference)
    audio(candidate, frequency=660)
    reply = {
        "assessable": True,
        "rating": 0,
        "reference_character": "sustained",
        "candidate_character": "different",
        "agreements": [],
        "differences": ["different tone"],
        "limitations": [],
    }

    reply = {
        "reference_character": reply.pop("reference_character"),
        "candidate_a": reply,
        "candidate_b": dict(reply),
    }

    def handle(request):
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "gpt-audio-1.5"
        assert body["messages"][0]["content"] == audio_judge.RUBRIC
        assert body["store"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    result = await audio_judge.judge_window(
        reference,
        candidate,
        candidate,
        tmp_path / "receipt.json",
        transport=httpx.MockTransport(handle),
    )
    assert result["score"] == 0
    assert result["delivery"]["route"] == "audio_semantic_judge"
    assert result["raw"]["choices"]


@pytest.mark.asyncio
async def test_provider_failure_does_not_become_zero(tmp_path):
    reference, candidate = tmp_path / "reference.wav", tmp_path / "candidate.wav"
    audio(reference)
    audio(candidate, frequency=660)
    with pytest.raises(JudgeInfrastructureError):
        await audio_judge.judge_window(
            reference,
            candidate,
            candidate,
            tmp_path / "receipt.json",
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        )


def test_only_simple_setter_redundancy_is_removed():
    events = [
        (0, {"type": "control_change", "channel": 0, "control": 7, "value": 90}),
        (10, {"type": "control_change", "channel": 0, "control": 7, "value": 90}),
    ]
    assert control_curves(events)[0][(0, 7)] == [(0, 90)]
    for controller in (0, 32, 6, 38, 96, 97, 98, 99, 100, 101, 64, 68, 123):
        stateful = copy.deepcopy(events)
        for _, event in stateful:
            event["control"] = controller
        assert len(control_curves(stateful)[1]) == 2


def test_whole_evaluator_missing_project_returns_gate_failure(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "migration_runtime_test", ROOT / "scripts/evaluate_remote.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    references = tmp_path / "references"
    references.mkdir()
    audio(references / "Flute.wav")
    result = module.evaluate_local(
        tmp_path / "output",
        tmp_path / "baseline.ardour",
        references,
        tmp_path / "evidence",
        native=False,
    )
    assert not result["gate_passed"]
    assert "migrated_project/migrated_project.ardour" in result["missing_files"]


@pytest.mark.asyncio
async def test_actual_task_setup_with_runtime_adapter(tmp_path, monkeypatch):
    class LocalInterface:
        async def run_command(self, command):
            result = subprocess.run(command, shell=True, capture_output=True)
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout.decode(),
                stderr=result.stderr.decode(),
            )

        async def write_bytes(self, path, data):
            Path(path).write_bytes(data)

    wrapper = tmp_path / "ale-migration-env"
    commands = tmp_path / "runtime-commands"
    wrapper.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> " + shlex.quote(str(commands)) + '\nexec "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setattr(runtime, "RUNTIME_WRAPPER", str(wrapper))
    config = main.TaskConfig(REMOTE_ROOT_DIR=str(tmp_path), VARIANT_NAME=SELECTED_VARIANT)
    session = RemoteDesktopSession(api_url="http://127.0.0.1:1", os_type="linux")
    session._computer = SimpleNamespace(interface=LocalInterface())
    session._initialized = True
    await main.start(SimpleNamespace(metadata=config.to_metadata()), session)
    snapshot = Path(config.input_dir) / "project/project.ardour"
    assert snapshot.is_file()
    assert str(snapshot.parent) in snapshot.read_text()
    assert (Path(config.software_dir) / "rubric.txt").read_text() == audio_judge.RUBRIC
    assert not (Path(config.software_dir) / "score_audio_remote.py").exists()
    assert not Path(config.reference_dir).exists()
    assert f"python3 {config.software_dir}/rebind.py" in commands.read_text()


@pytest.mark.asyncio
async def test_copied_reference_cannot_hide_wrong_native_playback(tmp_path):
    record = {
        "valid": True,
        "reference_audio": {"audible": True},
        "passages": [
            {
                "reference_path": "ref.wav",
                "candidate_path": "copied-reference.wav",
                "native_path": "wrong-native.wav",
            }
        ],
    }

    async def judge(reference, candidate, native, receipt):
        assert candidate.name == "copied-reference.wav"
        assert native.name == "wrong-native.wav"
        return {"score": 0.0, "delivery": {"score": 1.0}, "native": {"score": 0.0}}

    result = await audio_judge.score_delivery(
        {"delivery_score": 1.0, "stems": [record]}, tmp_path, window_judge=judge
    )
    assert result["timbre_score"] == 0
    assert result["weighted_score"] == 0.2


@pytest.fixture
def task_hook(tmp_path, monkeypatch):
    clip = tmp_path / "parity.wav"
    audio(clip)
    payload = clip.read_bytes()
    stem = {
        "valid": True,
        "reference_audio": {"audible": True},
        "passages": [
            {
                "reference_path": "/remote/reference.wav",
                "candidate_path": "/remote/delivered.wav",
                "native_path": "/remote/native.wav",
            }
        ],
    }
    missing = {"valid": False, "reference_audio": {"audible": True}, "passages": []}
    assessment = {
        "gate_passed": True,
        "delivery": {
            "delivery_score": 0.05,
            "stems": [stem] + [copy.deepcopy(missing) for _ in range(19)],
        },
    }

    class Interface:
        def __init__(self):
            self.written = {}
            self.command_status = 0

        async def run_command(self, command):
            if "nohup" in command:
                assert "evaluate_remote.py" in command and "timeout 7200" in command
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps({"exit_code": 0, "stdout": "", "stderr": ""}),
                    stderr="",
                )
            assert "worker.exit" in command
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"exit_code": 0, "stdout": str(self.command_status), "stderr": ""}
                ),
                stderr="",
            )

        async def read_bytes(self, path):
            if path.endswith("assessment.json"):
                return json.dumps(assessment).encode()
            return payload

        async def write_bytes(self, path, data):
            self.written[path] = json.loads(data)

    async def staged(meta, session, *, evaluating=False):
        assert evaluating
        return "/remote/evaluator", "/remote/tools"

    async def overview(**kwargs):
        return {"score": 1.0}

    def context(**kwargs):
        return EvaluationContext(**kwargs, auto_save=False)

    async def forbid_provider(*args, **kwargs):
        pytest.fail("The hook gate controls must not contact a provider")

    monkeypatch.setattr(main, "stage", staged)
    monkeypatch.setattr(main, "llm_vision_judge", overview)
    monkeypatch.setattr(main, "EvaluationContext", context)
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(httpx.AsyncClient, "post", forbid_provider)
    config = SimpleNamespace(metadata=main.TaskConfig(VARIANT_NAME="eora").to_metadata())
    session = RemoteDesktopSession(api_url="http://127.0.0.1:1", os_type="linux")
    session._computer = SimpleNamespace(interface=Interface())
    session._initialized = True
    return config, session, assessment


@pytest.mark.asyncio
async def test_task_hook_preserves_full_denominator_for_parity_audio(task_hook):
    config, session, assessment = task_hook
    assert await main.evaluate(config, session) == pytest.approx([0.05])
    saved = session.interface.written["/remote/evaluator/evidence/scored.json"]
    assert saved["weights"] == [0.2, 0.8]
    assert len(saved["delivery"]["stems"]) == 20
    window = saved["delivery"]["stems"][0]["judgments"][0]
    assert window["score"] == 1
    assert window["delivery"]["route"] == "exact_decoded_pcm"
    assert window["native"]["reused_identical_audio"]


@pytest.mark.asyncio
async def test_task_hook_rejects_delivery_gate_without_judging(task_hook, monkeypatch):
    config, session, assessment = task_hook
    assessment["gate_passed"] = False

    async def forbidden(**kwargs):
        pytest.fail("Invalid delivery must stop before image/audio judging")

    monkeypatch.setattr(main, "llm_vision_judge", forbidden)
    assert await main.evaluate(config, session) == [0.0]
    assert not session.interface.written


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reference_unavailable", "worker_exit", "native_audio_absent"])
async def test_task_hook_infrastructure_is_not_candidate_zero(task_hook, failure):
    config, session, assessment = task_hook
    if failure == "reference_unavailable":
        assessment["infrastructure_error"] = "Original reference unavailable"
    elif failure == "worker_exit":
        session.interface.command_status = 137
    else:
        del assessment["delivery"]["stems"][0]["passages"][0]["native_path"]
    with pytest.raises(JudgeInfrastructureError):
        await main.evaluate(config, session)
    assert not session.interface.written


@pytest.mark.asyncio
async def test_task_hook_judge_outage_is_not_candidate_zero(task_hook, monkeypatch):
    config, session, assessment = task_hook

    async def unavailable(**kwargs):
        raise JudgeInfrastructureError("Overview judge unavailable")

    monkeypatch.setattr(main, "llm_vision_judge", unavailable)
    with pytest.raises(JudgeInfrastructureError, match="Overview judge unavailable"):
        await main.evaluate(config, session)
    assert not session.interface.written
