import hashlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.inner_support_elevation_optimization import main, native_task, source_review
from tasks.engineering.inner_support_elevation_optimization.verification import (
    EvaluationUnavailableError,
)
from test_inner_support_contract import declared_family as declared_family


@pytest.fixture
def sources(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    originals = tmp_path / "originals"
    originals.mkdir()
    content = b"Unit-test source bundle"
    (originals / "notes.md").write_bytes(content)
    (assets / "source_manifest.json").write_text(
        json.dumps({"sha256": {"notes.md": hashlib.sha256(content).hexdigest()}})
    )
    monkeypatch.setattr(source_review, "ASSETS", assets)

    def verify(files):
        assert files == {"notes.md": content}

    monkeypatch.setattr(source_review, "verify_source_bundle", verify)
    return originals


@pytest.mark.asyncio
async def test_deterministic_review_needs_no_judge_or_reported_response(
    declared_family, sources, tmp_path, monkeypatch
):
    import builtins

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "tasks.utils.evaluation":
            pytest.fail("A model-service dependency was imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    (declared_family / "answer.json").write_text('{"compliant": true}')
    first = await source_review.review_native_family(declared_family, sources, tmp_path / "review1")
    second = await source_review.review_native_family(
        declared_family, sources, tmp_path / "review2"
    )
    assert first["replay_eligible"] and not first["full_task_acceptance"]
    assert first["input_hashes"] == second["input_hashes"]
    assert first["blocking_findings"] == second["blocking_findings"] == []
    native = json.loads((tmp_path / "review1/native-review.json").read_text())
    assert native["contract"] == "support-deterministic-20261006"
    assert native["input_hashes"] == first["input_hashes"]
    with pytest.raises(FileExistsError):
        await source_review.review_native_family(declared_family, sources, tmp_path / "review1")


@pytest.mark.asyncio
async def test_actual_constraint_failure_blocks_before_native_replay(
    declared_family, sources, tmp_path
):
    path = declared_family / "family.json"
    family = json.loads(path.read_text())
    family["monitors"]["CX1"][1] -= 1
    path.write_text(json.dumps(family))
    result = await source_review.review_native_family(declared_family, sources, tmp_path / "review")
    assert not result["replay_eligible"]
    assert result["outcome"] == "source_violation"
    assert "CX" in result["blocking_findings"][0]["reason"]
    assert not (tmp_path / "review/native-review.json").exists()


@pytest.mark.asyncio
async def test_formal_hook_checks_contract_then_replays_then_scores(monkeypatch):
    observed = []

    async def review(*args):
        observed.append("review")
        return {"replay_eligible": True, "evidence_directory": "/evaluator/review"}

    async def replay(*args, deadline):
        assert deadline > main.time.monotonic()
        observed.append("replay")
        return {"completed": True}

    async def score(*args):
        observed.append("score")
        return {"score": 0.9286}

    monkeypatch.setattr(main, "review_submission", review)
    monkeypatch.setattr(main, "replay_submission", replay)
    monkeypatch.setattr(main, "evaluate_delivery", score)
    assert await main.evaluate(SimpleNamespace(metadata={}), object()) == [0.9286]
    assert observed == ["review", "replay", "score"]


@pytest.mark.asyncio
async def test_failed_public_check_scores_zero_with_receipt(monkeypatch, tmp_path):
    findings = [{"requirement": "model_definition.md", "reason": "Wall coverage gap"}]

    async def review(*args):
        return {
            "replay_eligible": False,
            "outcome": "source_violation",
            "blocking_findings": findings,
            "engineering_caveats": [],
            "evidence_directory": str(tmp_path),
        }

    async def forbidden(*args, **kwargs):
        pytest.fail("Contract failure reached native replay or delivery scoring")

    monkeypatch.setattr(main, "review_submission", review)
    monkeypatch.setattr(main, "replay_submission", forbidden)
    monkeypatch.setattr(main, "evaluate_delivery", forbidden)
    assert await main.evaluate(SimpleNamespace(metadata={}), object()) == [0.0]
    gate = json.loads((tmp_path / "evaluation-gate.json").read_text())
    assert gate["evaluation_status"] == "source_noncompliant"
    assert gate["findings"] == findings and not gate["native_replay_completed"]


@pytest.mark.asyncio
async def test_native_unavailability_is_not_a_solver_zero(monkeypatch):
    async def review(*args):
        return {"replay_eligible": True, "evidence_directory": "/evaluator/review"}

    async def unavailable(*args, **kwargs):
        raise EvaluationUnavailableError("Native replay incomplete")

    async def forbidden(*args, **kwargs):
        pytest.fail("Unavailable replay reached delivery scoring")

    monkeypatch.setattr(main, "review_submission", review)
    monkeypatch.setattr(main, "replay_submission", unavailable)
    monkeypatch.setattr(main, "evaluate_delivery", forbidden)
    with pytest.raises(EvaluationUnavailableError, match="incomplete"):
        await main.evaluate(SimpleNamespace(metadata={}), object())


@pytest.mark.asyncio
@pytest.mark.parametrize("completed,exit_status", [(False, "0"), (True, "1")])
async def test_native_terminal_failure_prevents_presentation_credit(
    tmp_path, monkeypatch, completed, exit_status
):
    evidence = tmp_path / "review"
    evidence.mkdir()
    (evidence / "native-review.json").write_text("{}")
    family = tmp_path / "family"
    family.mkdir()
    for name in ("family.json", "model.mdpa", "Materials.json"):
        (family / name).write_text("fixture")

    async def review(*args):
        return {"replay_eligible": True, "evidence_directory": str(evidence)}

    async def upload(*args):
        pass

    async def command(*args, **kwargs):
        return {"return_code": 0, "stdout": f"MainPID=0\nExecMainStatus={exit_status}"}

    async def forbidden(*args, **kwargs):
        pytest.fail("Incomplete native replay reached presentation scoring")

    class Session:
        async def read_file(self, path):
            assert path.endswith("/family-replay.json")
            return json.dumps({"completed": completed, "error": "fixture native failure"})

    monkeypatch.setattr(main, "review_submission", review)
    monkeypatch.setattr(main, "evaluate_delivery", forbidden)
    monkeypatch.setattr(native_task, "upload_files", upload)
    monkeypatch.setattr(native_task, "run_command", command)
    with pytest.raises(EvaluationUnavailableError, match="Native replay incomplete"):
        await main.evaluate(
            SimpleNamespace(metadata={"remote_output_dir": "/submission"}), Session()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "changed", "space"])
async def test_snapshot_binds_only_source_and_native_inputs(tmp_path, monkeypatch, damage):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "source_manifest.json").write_text(json.dumps({"sha256": {"notes.md": "fixture"}}))
    files = {
        "/original/notes.md": b"original",
        **{
            f"/submission/model_family_support_position/{name}": name.encode()
            for name in ("family.json", "model.mdpa", "Materials.json")
        },
    }
    reads = []

    class Session:
        async def run_command(self, command, **kwargs):
            paths = json.loads(shlex.split(shlex.split(command)[-2])[-1])
            inventory = {
                key: {"bytes": len(files[path]), "sha256": hashlib.sha256(files[path]).hexdigest()}
                for key, path in paths.items()
            }
            return {
                "return_code": 0,
                "stdout": json.dumps(
                    {"return_code": 0, "stdout": json.dumps(inventory), "stderr": ""}
                ),
            }

        async def read_bytes(self, path):
            reads.append(path)
            return files[path] + (b"changed" if damage == "changed" else b"")

    async def review(family, original, output):
        assert (family / "model.mdpa").read_bytes() == b"model.mdpa"
        assert (original / "notes.md").read_bytes() == b"original"
        return {"replay_eligible": True}

    monkeypatch.setattr(source_review, "ASSETS", assets)
    monkeypatch.setattr(source_review, "review_native_family", review)
    monkeypatch.setattr(
        source_review.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=(1 if damage == "space" else 20) * 1024**3),
    )
    if damage:
        with pytest.raises(EvaluationUnavailableError, match="changed|reserve"):
            await source_review.review_submission(
                {"input_dir": "/original", "remote_output_dir": "/submission"}, Session()
            )
    else:
        result = await source_review.review_submission(
            {"input_dir": "/original", "remote_output_dir": "/submission"}, Session()
        )
        assert result["replay_eligible"] and len(reads) == 4
        assert Path(result["evidence_directory"]).is_relative_to(tmp_path)
