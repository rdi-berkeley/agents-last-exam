import csv
import io
import json
import shlex
from pathlib import Path

import pytest
from PIL import Image

from ale_run.tasks.revision import task_revision
from tasks.engineering.inner_support_elevation_optimization import (
    delivery_scoring as scoring,
    native_task,
)


@pytest.fixture
def delivery():
    cases = {}
    for case, scale in zip(scoring.CASES, [1.2, 1.1, 1.4, 1.3, 1.0]):
        cases[case] = {
            "stations": [
                {"monitor": f"DBC{number}", "xyz_m": [number, 0, 6.5], "u_m": [0, 0, -0.01 * scale]}
                for number in range(1, 11)
            ],
            "profiles": [
                {
                    "monitor": f"CX{number}",
                    "xyz_m": [number, 1, level],
                    "u_m": [value * scale, 0, 0],
                }
                for number in range(1, 11)
                for level, value in [(-26.5, 0), (-10, 0.02), (7, 0.01)]
            ],
            "max_settlement_mm": 10 * scale,
            "max_disp_mm": 20 * scale,
        }
    answer, _ = scoring.reference_answer(cases)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["case_label", "support_position_m", "max_settlement_mm", "max_disp_mm"])
    for position, case in enumerate(scoring.CASES):
        writer.writerow(
            [case, position, answer[f"{case}_max_settlement_mm"], answer[f"{case}_max_disp_mm"]]
        )
    image = Image.new("RGB", (320, 240), "white")
    image.paste("black", (20, 20, 200, 200))
    picture = io.BytesIO()
    image.save(picture, format="PNG")
    decision = {key: answer[key] for key in scoring.DERIVED}
    decision["recommended_position_m"] = 3
    decision.update(
        dict.fromkeys(
            (
                "recommendation_reason",
                "assumptions",
                "limitations",
                "installation_tradeoff",
                "rerun_instructions",
            ),
            "Explicit engineering discussion of this model and its computed results.",
        )
    )
    files = {
        "/guest/submission/answer.json": json.dumps(answer).encode(),
        "/guest/submission/case_summary.csv": table.getvalue().encode(),
        "/guest/submission/support_position_engineering_memo.md": (
            "```support-decision\n" + json.dumps(decision) + "\n```\n"
        ).encode(),
        **{f"/guest/submission/{name}": picture.getvalue() for name in scoring.REQUIRED_IMAGES},
        **{
            f"/guest/submission/model_family_support_position/results/{case}/independent_monitors.json": json.dumps(
                row
            ).encode()
            for case, row in cases.items()
        },
    }
    delivered = {
        "cases": {
            case: {"native_result_read": True, "observations": json.loads(json.dumps(row))}
            for case, row in cases.items()
        }
    }
    return cases, files, delivered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage,expected_score",
    [
        (None, 1.0),
        ("missing_png", 0.0),
        ("missing_answer", 0.0),
        ("missing_raw", 0.0),
        ("native_disagrees", 0.94),
        ("raw_disagrees", 0.92),
    ],
)
async def test_final_delivery_keeps_scores_and_receipts_in_guest(
    delivery, monkeypatch, damage, expected_score
):
    cases, files, delivered = delivery
    raw_path = (
        "/guest/submission/model_family_support_position/results/pos_0m/independent_monitors.json"
    )
    if damage in ("missing_png", "missing_answer", "missing_raw"):
        del files[
            {
                "missing_png": f"/guest/submission/{scoring.REQUIRED_IMAGES[0]}",
                "missing_answer": "/guest/submission/answer.json",
                "missing_raw": raw_path,
            }[damage]
        ]
    elif damage == "native_disagrees":
        delivered["cases"]["pos_0m"]["observations"]["stations"][0]["u_m"][2] = -1
    elif damage == "raw_disagrees":
        raw = json.loads(files[raw_path])
        raw["stations"][0]["u_m"][2] = -1
        files[raw_path] = json.dumps(raw).encode()
    before_files = dict(files)
    writes, commands = [], []
    remote = "/guest/evaluator"
    evidence = remote + "/review"

    class Session:
        async def file_exists(self, path):
            return path in files

        async def read_bytes(self, path):
            return files[path]

        async def read_file(self, path):
            return files[path].decode()

    session = Session()

    async def native_reader(actual_session, command, *, timeout, check):
        assert actual_session is session and timeout == 180 and check is False
        arguments = shlex.split(command)
        assert arguments[-5:] == [
            remote + "/scripts/native_verify_delivery.py",
            "/guest/submission/model_family_support_position",
            remote + "/replay/family-replay.json",
            remote + "/review.json",
            remote + "/native-delivery.json",
        ]
        commands.append(command)
        files[arguments[-1]] = json.dumps(delivered).encode()
        return {"return_code": 0}

    async def upload(actual_session, root, payload):
        assert actual_session is session and str(root) == evidence
        for name, content in payload.items():
            assert name in {"native-delivery.json", "score.json"} and isinstance(content, bytes)
            path = str(Path(root) / name)
            assert path.startswith("/guest/")
            writes.append(path)
            files[path] = content

    def forbidden_write(*args, **kwargs):
        pytest.fail("Delivery evaluation attempted a host Path write")

    original_open = Path.open

    def read_only_open(path, mode="r", *args, **kwargs):
        assert not set(mode) & set("wax+"), "Delivery evaluation attempted a host Path write"
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(native_task, "run_command", native_reader)
    monkeypatch.setattr(native_task, "upload_files", upload)
    before_revision = task_revision(native_task.ROOT)
    caveats = [{"reason": "Ranking is model dependent"}]
    with monkeypatch.context() as guard:
        for method in ("write_text", "write_bytes", "mkdir"):
            guard.setattr(Path, method, forbidden_write)
        guard.setattr(Path, "open", read_only_open)
        result = await native_task.evaluate_delivery(
            {"remote_output_dir": "/guest/submission"},
            session,
            {"evidence_directory": evidence, "engineering_caveats": caveats},
            {"remote": remote, "observations": cases},
        )
    assert task_revision(native_task.ROOT) == before_revision
    assert result["score"] == expected_score
    assert all(files[path] == content for path, content in before_files.items())
    if damage in ("missing_png", "missing_answer"):
        assert "Missing original deliverable" in result["reason"]
        assert not writes and not commands and files == before_files
        return
    assert len(commands) == 1
    assert writes == [evidence + "/native-delivery.json", evidence + "/score.json"]
    assert set(files) - set(before_files) == {remote + "/native-delivery.json", *writes}
    assert json.loads(files[writes[0]]) == delivered
    assert json.loads(files[writes[1]]) == result
    assert result["numeric_subscore"] == (0.0 if damage == "missing_raw" else 0.5)
    assert result["presentation_checks"]["images_fraction"] == 1
    assert result["presentation_checks"]["memo_fraction"] == 1
    assert result["engineering_caveats"] == caveats
    checks = result["delivery_checks"]
    assert checks["required_files_present"] == (damage != "missing_raw")
    assert checks["native_results_agree"] == (damage != "native_disagrees")
    assert checks["raw_monitors_agree"] == (damage not in ("missing_raw", "raw_disagrees"))
    assert "delivery_judgment" not in result
