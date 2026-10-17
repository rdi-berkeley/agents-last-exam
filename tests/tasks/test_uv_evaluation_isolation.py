from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageOps

from ale_run.tasks.revision import task_revision
from tasks.utils.evaluation import JudgeInfrastructureError
from tasks.visual_media.uv_reproduction import main
from tasks.visual_media.uv_reproduction.scripts import local_soft_eval as soft


def png(color: str, size: tuple[int, int] = (12, 10)) -> bytes:
    with Image.new("RGBA", size, color) as image, BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def original_bundle(frame_pairs: list[dict], directory: Path) -> bytes:
    directory.mkdir()
    images = []
    for index, pair in enumerate(frame_pairs):
        with Image.open(BytesIO(pair["reference_image"])) as source:
            reference = source.convert("RGBA")
        with Image.open(BytesIO(pair["candidate_image"])) as source:
            candidate = source.convert("RGBA")
        with reference, candidate, Image.new(
            "RGBA",
            (reference.width + candidate.width, max(reference.height, candidate.height) + 64),
            (255, 255, 255, 255),
        ) as sheet:
            sheet.paste(reference, (0, 40))
            sheet.paste(candidate, (reference.width, 40))
            ImageDraw.Draw(sheet).text(
                (12, 10),
                f"{pair['view']} | left=reference | right=candidate",
                fill=(20, 20, 20, 255),
            )
            sheet_path = directory / f"pair_{index}.png"
            sheet.save(sheet_path)
        with Image.open(sheet_path) as source:
            images.append(source.convert("RGBA"))
    width = max(image.width for image in images)
    with Image.new("RGBA", (width, sum(image.height for image in images)), "white") as bundle:
        offset = 0
        for image in images:
            if image.width != width:
                padded = ImageOps.pad(image, (width, image.height), color=(255, 255, 255, 255))
                bundle.paste(padded, (0, offset))
                padded.close()
            else:
                bundle.paste(image, (0, offset))
            offset += image.height
            image.close()
        output = directory / "bundle.png"
        bundle.save(output)
    return output.read_bytes()


@pytest.fixture
def judge_calls(monkeypatch):
    calls = []

    def judge(**kwargs):
        calls.append(kwargs)
        return {
            "final_score": 2 / 3,
            "results": ["YES", "YES", "NO"],
            "raw_responses": ["YES", "YES", "NO"] if len(calls) == 1 else ["YES\n", "YES\n", "NO\n"],
        }

    monkeypatch.setattr(soft, "judge_config", lambda: {"api_key": "test-key"})
    monkeypatch.setattr(soft, "resolve_llm_judge_model", lambda: "test-judge")
    monkeypatch.setattr(soft, "llm_vision_binary_questions_sync", judge)
    return calls


def test_memory_scoring_preserves_original_judge_payload(tmp_path, judge_calls):
    pairs = [
        {"view": "front", "reference_image": png("red"), "candidate_image": png("blue")},
        {
            "view": "back",
            "reference_image": png("green", (16, 9)),
            "candidate_image": png("white", (15, 12)),
        },
    ]
    expected_bundle = original_bundle(pairs, tmp_path / "original")
    report = {}
    assert soft.run_local_soft_eval(pairs, report_out=report) == 2 / 3
    assert report["raw_responses"] == ["YES", "YES", "NO"]
    assert report["results"] == ["YES", "YES", "NO"]
    payload = judge_calls[-1]
    assert payload["image_bytes_list"] == [expected_bundle]
    assert payload["questions"] == [
        "Is the texture placement and orientation correct enough to pass?",
        "Does the candidate preserve the reference material appearance and color palette well enough to pass?",
        "Are obvious UV seams, stretching, or texture artifacts absent enough for the result to pass?",
    ]
    assert payload["temperature"] == 0
    assert payload["model"] == "test-judge"
    assert payload["api_key"] == "test-key"

    file_pairs = []
    for index, pair in enumerate(pairs):
        localized = {"view": pair["view"]}
        for key in ("reference_image", "candidate_image"):
            path = tmp_path / f"{index}_{key}.png"
            path.write_bytes(pair[key])
            localized[key] = str(path)
        file_pairs.append(localized)
    diagnostics = tmp_path / "diagnostics"
    assert soft.run_local_soft_eval(file_pairs, diagnostics) == 2 / 3
    assert judge_calls[-1] == payload
    assert (diagnostics / "soft_eval_bundle.png").read_bytes() == expected_bundle
    assert json.loads((diagnostics / "soft_eval_report.json").read_text())["final_score"] == 2 / 3


@pytest.mark.parametrize("failure", [None, "judge", "image"])
def test_default_soft_eval_never_writes_host_files(monkeypatch, judge_calls, failure):
    pairs = [{"view": "front", "reference_image": png("red"), "candidate_image": png("blue")}]
    if failure == "image":
        pairs.append({"view": "back", "reference_image": b"invalid", "candidate_image": png("red")})
    if failure == "judge":
        def fail(**kwargs):
            raise JudgeInfrastructureError("judge unavailable")

        monkeypatch.setattr(soft, "llm_vision_binary_questions_sync", fail)

    def forbid_write(*args, **kwargs):
        raise AssertionError("in-memory scoring must not create host files")

    monkeypatch.setattr(Path, "mkdir", forbid_write)
    monkeypatch.setattr(Path, "write_bytes", forbid_write)
    monkeypatch.setattr(Path, "write_text", forbid_write)
    if failure:
        expected = JudgeInfrastructureError if failure == "judge" else OSError
        with pytest.raises(expected):
            soft.run_local_soft_eval(pairs)
    else:
        report = {}
        assert soft.run_local_soft_eval(pairs, report_out=report) == 2 / 3
        assert report["raw_responses"] == ["YES", "YES", "NO"]


def test_pair_sheet_closes_converted_reference_if_candidate_is_invalid(monkeypatch):
    converted = []
    original_convert = Image.Image.convert

    def convert(image, *args, **kwargs):
        result = original_convert(image, *args, **kwargs)
        converted.append(result)
        return result

    reference = png("red")
    monkeypatch.setattr(Image.Image, "convert", convert)
    with pytest.raises(OSError):
        soft._pair_sheet(reference, b"invalid", "front")
    assert len(converted) == 1
    with pytest.raises(ValueError, match="closed image"):
        converted[0].getpixel((0, 0))


def test_stack_releases_padded_image_on_paste_failure(monkeypatch):
    original_pad = ImageOps.pad
    padded_images = []

    def pad(*args, **kwargs):
        image = original_pad(*args, **kwargs)
        padded_images.append(image)
        return image

    original_paste = Image.Image.paste

    def paste(image, source, *args, **kwargs):
        if source in padded_images:
            raise RuntimeError("paste failed")
        return original_paste(image, source, *args, **kwargs)

    monkeypatch.setattr(ImageOps, "pad", pad)
    monkeypatch.setattr(Image.Image, "paste", paste)
    images = [Image.new("RGBA", (10, 10)), Image.new("RGBA", (20, 10))]
    try:
        with pytest.raises(RuntimeError, match="paste failed"):
            soft._stack(images)
        assert len(padded_images) == 1
        with pytest.raises(ValueError, match="closed image"):
            padded_images[0].getpixel((0, 0))
    finally:
        for image in images:
            image.close()


class MemorySession:
    def __init__(self):
        self.interface = self
        self.files = {}
        self.directories = set()
        self.commands = []

    async def create_dir(self, path):
        self.directories.add(path)

    async def write_file(self, path, content):
        self.files[path] = content.encode()

    async def read_bytes(self, path):
        await asyncio.sleep(0)
        return self.files[path]

    async def file_exists(self, path):
        return path in self.files

    async def directory_exists(self, path):
        return path in self.directories

    async def run_command(self, command, **kwargs):
        self.commands.append(command)
        raise AssertionError("no shared scratch-directory delete is allowed")


def install_remote_job(monkeypatch, reports, *, barrier=None):
    jobs = []

    async def launch(session, *, remote_scripts_dir, args):
        results = args[args.index("--output-dir") + 1]
        job_index = len(jobs)
        jobs.append((remote_scripts_dir, results))
        image_path = main._remote_child(results, "candidate.png")
        reference_path = main._remote_child(results, "reference.png")
        session.files[image_path] = png("red" if job_index == 0 else "blue")
        session.files[reference_path] = png("red")
        report = {
            **reports[job_index % len(reports)],
            "frame_pairs": [{
                "view": "front",
                "reference_image": reference_path,
                "candidate_image": image_path,
            }],
        }
        session.files[main._remote_child(results, "hard_eval_report.json")] = json.dumps(report).encode()
        if barrier is not None:
            if len(jobs) == 2:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), timeout=5)

    monkeypatch.setattr(main, "_launch_remote_job", launch)
    return jobs


def task_config(multipart=False):
    config = main._variant_to_cfg(main.VARIANTS[-1] if multipart else main.VARIANTS[0])
    return SimpleNamespace(metadata=config.to_metadata())


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
async def test_concurrent_same_task_evaluations_are_isolated(
    monkeypatch, tmp_path, judge_calls, multipart
):
    session = MemorySession()
    config = task_config(multipart)
    session.files[config.metadata["output_scene" if multipart else "output_obj"]] = b"submission"
    task_tree = tmp_path / "task"
    task_tree.mkdir()
    (task_tree / "main.py").write_text("frozen task source")
    old_evidence = task_tree / ".tmp_soft_eval" / config.metadata["task_tag"]
    old_evidence.mkdir(parents=True)
    (old_evidence / "old-report.json").write_text("old evidence")
    monkeypatch.setattr(main, "TASK_DIR", task_tree)
    source_revision = task_revision(task_tree)
    real_revision = task_revision(main.SCRIPTS_DIR.parent)
    jobs = install_remote_job(
        monkeypatch,
        [{"score": 1.0}, {"score": 0.5}],
        barrier=asyncio.Event(),
    )

    scores = await asyncio.gather(main.evaluate(config, session), main.evaluate(config, session))

    assert scores == [[pytest.approx(0.3 + 0.7 * 2 / 3)], [pytest.approx(0.15 + 0.7 * 2 / 3)]]
    assert len(set(jobs)) == 2
    raw_by_candidate = {}
    for call_index, call in enumerate(judge_calls):
        with Image.open(BytesIO(call["image_bytes_list"][0])) as bundle:
            raw_by_candidate[bundle.getpixel((12, 40))] = (
                ["YES", "YES", "NO"] if call_index == 0 else ["YES\n", "YES\n", "NO\n"]
            )
    for scripts, results in jobs:
        root = PureWindowsPath(results).parent
        assert PureWindowsPath(scripts).parent == root
        assert root.parent.name == config.metadata["task_tag"]
        assert len(root.name) == 32
        assert main._remote_child(results, "hard_eval_report.json") in session.files
        soft_report = json.loads(session.files[main._remote_child(results, "soft_eval_report.json")])
        assert soft_report["final_score"] == 2 / 3
        assert soft_report["results"] == ["YES", "YES", "NO"]
        with Image.open(BytesIO(session.files[main._remote_child(results, "candidate.png")])) as candidate:
            assert soft_report["raw_responses"] == raw_by_candidate[candidate.getpixel((0, 0))]
    assert judge_calls[0]["image_bytes_list"] != judge_calls[1]["image_bytes_list"]
    assert task_revision(task_tree) == source_revision
    assert task_revision(main.SCRIPTS_DIR.parent) == real_revision
    assert (old_evidence / "old-report.json").read_text() == "old evidence"
    assert not session.commands


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transfer", "judge", "cancel", "report_write"])
async def test_failed_evaluation_leaves_no_host_scratch_and_preserves_guest_evidence(
    monkeypatch, tmp_path, judge_calls, failure
):
    session = MemorySession()
    config = task_config()
    session.files[config.metadata["output_obj"]] = b"submission"
    task_tree = tmp_path / "task"
    task_tree.mkdir()
    (task_tree / "main.py").write_text("frozen task source")
    monkeypatch.setattr(main, "TASK_DIR", task_tree)
    before = task_revision(task_tree)
    jobs = install_remote_job(monkeypatch, [{"score": 1.0}])
    original_read = session.read_bytes

    async def fail_transfer(path):
        if path.endswith("candidate.png"):
            if failure == "cancel":
                raise asyncio.CancelledError
            raise OSError("transfer unavailable")
        return await original_read(path)

    def fail_judge(**kwargs):
        raise JudgeInfrastructureError("judge unavailable")

    original_write = session.write_file

    async def fail_report_write(path, content):
        if path.endswith("soft_eval_report.json"):
            raise OSError("guest report storage unavailable")
        await original_write(path, content)

    if failure == "judge":
        monkeypatch.setattr(soft, "llm_vision_binary_questions_sync", fail_judge)
        expected = JudgeInfrastructureError
    elif failure == "report_write":
        monkeypatch.setattr(session, "write_file", fail_report_write)
        expected = OSError
    else:
        monkeypatch.setattr(session, "read_bytes", fail_transfer)
        expected = asyncio.CancelledError if failure == "cancel" else OSError

    with pytest.raises(expected):
        await main.evaluate(config, session)

    assert task_revision(task_tree) == before
    assert list(task_tree.iterdir()) == [task_tree / "main.py"]
    assert main._remote_child(jobs[0][1], "hard_eval_report.json") in session.files
    assert main._remote_child(jobs[0][1], "candidate.png") in session.files
    assert not session.commands


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "multipart, report, soft_score, expected",
    [
        (False, {"score": 1.0}, 1.0, 1.0),
        (False, {"score": 0.0}, 0.0, 0.0),
        (False, {"score": 0.8}, 0.5, 0.59),
        (False, {"score": 0.0, "metrics": {"gate": {"uv": False}}}, 1.0, 0.0),
        (True, {"score": 0.0, "missing_parts": ["part"]}, 1.0, 0.0),
    ],
)
async def test_scoring_controls_preserve_weights_and_hard_gates(
    monkeypatch, multipart, report, soft_score, expected
):
    session = MemorySession()
    config = task_config(multipart)
    session.files[config.metadata["output_scene" if multipart else "output_obj"]] = b"submission"
    install_remote_job(monkeypatch, [report])
    monkeypatch.setattr(soft, "judge_config", lambda: {"api_key": "test-key"})
    monkeypatch.setattr(soft, "resolve_llm_judge_model", lambda: "test-judge")
    monkeypatch.setattr(soft, "llm_vision_binary_questions_sync", lambda **kwargs: {"final_score": soft_score})

    assert await main.evaluate(config, session) == [pytest.approx(expected)]
