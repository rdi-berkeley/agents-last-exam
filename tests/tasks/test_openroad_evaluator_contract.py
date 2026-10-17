import asyncio
import json
import os
import shlex
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.engineering.openroad_sky130_ibex_pnr_signoff import main as task


METADATA = task.config.to_metadata()


@pytest.fixture
def session():
    return SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        interface=SimpleNamespace(create_dir=AsyncMock()),
        run_command=AsyncMock(return_value={"return_code": 0, "stdout": "/tmp/openroad-test\n"}),
        write_file=AsyncMock(),
    )


def evaluate(session):
    return asyncio.run(task.evaluate(SimpleNamespace(metadata=METADATA), session))


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_candidate_scores_unchanged(session, monkeypatch, score):
    background = AsyncMock(
        return_value=(json.dumps({"normalized_score": score, "total_score": score * 100}), "")
    )
    monkeypatch.setattr(task, "_run_verifier_background", background)
    assert evaluate(session) == [score]
    command = background.await_args.args[1]
    assert "--skip-docker-unsafe" not in command
    assert "--skip-reseed" not in command


def test_missing_submission_still_zero(session):
    session.file_exists.side_effect = lambda path: path != METADATA["remote_output_dir"]
    assert evaluate(session) == [0.0]
    session.write_file.assert_not_awaited()


@pytest.mark.parametrize(
    "key",
    ["reference_dir", "reference_frozen_hashes", "reference_metrics", "reference_starter_zip"],
)
def test_missing_trusted_reference_is_evaluation_error(session, key):
    session.file_exists.side_effect = lambda path: path != METADATA[key]
    with pytest.raises(RuntimeError, match="missing evaluator reference"):
        evaluate(session)


@pytest.mark.parametrize("failure", [RuntimeError("Request failed"), TimeoutError("timed out")])
def test_transport_and_timeout_are_not_scores(session, monkeypatch, failure):
    monkeypatch.setattr(task, "_run_verifier_background", AsyncMock(side_effect=failure))
    with pytest.raises(RuntimeError, match="verifier execution failed") as raised:
        evaluate(session)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    "reply",
    [
        {"return_code": 1, "stdout": "", "stderr": "No space left"},
        {"return_code": 0, "stdout": "\n"},
    ],
)
def test_scratch_failure_is_not_a_score(session, reply):
    session.run_command.return_value = reply
    with pytest.raises(RuntimeError, match="failed to create verifier run dir"):
        evaluate(session)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "Traceback: broken verifier",
        "[]",
        "null",
        "{}",
        "17",
        '{"normalized_score": null}',
        '{"normalized_score": true}',
        '{"normalized_score": "0"}',
        '{"normalized_score": NaN}',
        '{"normalized_score": Infinity}',
        '{"normalized_score": -0.1}',
        '{"normalized_score": 1.1}',
    ],
)
def test_malformed_verifier_output_is_not_a_score(session, monkeypatch, stdout):
    monkeypatch.setattr(task, "_run_verifier_background", AsyncMock(return_value=(stdout, "")))
    with pytest.raises(RuntimeError, match="invalid verifier result"):
        evaluate(session)


def test_missing_required_candidate_files_are_legitimate_zero(session, monkeypatch):
    payload = {"normalized_score": 0, "error": "submission must contain config.mk and JOURNAL.md"}
    monkeypatch.setattr(
        task, "_run_verifier_background", AsyncMock(return_value=(json.dumps(payload), ""))
    )
    assert evaluate(session) == [0.0]


@pytest.mark.parametrize("return_code", ["0", "1", "2", "137", "", "not-a-code"])
def test_background_verifier_exit_contract(session, monkeypatch, return_code):
    monkeypatch.setattr(task, "VERIFIER_POLL_INTERVAL_S", 0)
    session.run_command.side_effect = [
        {"return_code": 0, "stdout": "123\n"},
        {"return_code": 0, "stdout": "DONE\n"},
        {"return_code": 0, "stdout": return_code},
        {"return_code": 0, "stdout": ""},
        {"return_code": 0, "stdout": '{"normalized_score": 0.5}'},
    ]
    coroutine = task._run_verifier_background(
        session, ["python", "verifier.py"], tag="base", run_dir="/tmp/test"
    )
    if return_code in {"0", "1"}:
        assert asyncio.run(coroutine) == ('{"normalized_score": 0.5}', "")
    else:
        with pytest.raises(RuntimeError, match="verifier exited unexpectedly"):
            asyncio.run(coroutine)


@pytest.mark.parametrize("stdout", ["", "\n", "not-a-pid"])
def test_background_requires_launch_pid(session, stdout):
    session.run_command.return_value = {"return_code": 0, "stdout": stdout}
    with pytest.raises(RuntimeError, match="no process ID"):
        asyncio.run(
            task._run_verifier_background(session, ["true"], tag="base", run_dir="/tmp/test")
        )


def test_background_launch_failure_does_not_poll(session):
    session.run_command.return_value = {"return_code": 1, "stderr": "cannot spawn"}
    with pytest.raises(RuntimeError, match="failed to launch verifier"):
        asyncio.run(
            task._run_verifier_background(session, ["true"], tag="base", run_dir="/tmp/test")
        )
    session.run_command.assert_awaited_once()


def test_background_timeout_stops_its_process_group(session, monkeypatch):
    monkeypatch.setattr(task, "VERIFIER_TIMEOUT_S", 0)
    session.run_command.side_effect = [
        {"return_code": 0, "stdout": "123\n"},
        {"return_code": 0, "stdout": ""},
    ]
    with pytest.raises(TimeoutError, match="verifier timed out"):
        asyncio.run(
            task._run_verifier_background(session, ["sleep", "10"], tag="base", run_dir="/tmp/test")
        )
    stop_command = session.run_command.await_args.args[0]
    assert shlex.split(stop_command) == ["bash", "-lc", "kill -TERM -- -123 2>/dev/null || true"]


@pytest.mark.skipif(not os.environ.get("OPENROAD_AUDIT_CUA_URL"), reason="No retained audit guest")
def test_real_guest_launch_does_not_wait_for_background_work(monkeypatch):
    from cua_bench.computers.remote import RemoteDesktopSession

    monkeypatch.setattr(task, "VERIFIER_POLL_INTERVAL_S", 0.1)
    run_command = task._run_command
    launch_durations = []

    async def measure_launch(session, command, **kwargs):
        started = time.monotonic()
        result = await run_command(session, command, **kwargs)
        if not launch_durations:
            launch_durations.append(time.monotonic() - started)
        return result

    monkeypatch.setattr(task, "_run_command", measure_launch)

    async def run():
        session = RemoteDesktopSession(
            api_url=os.environ["OPENROAD_AUDIT_CUA_URL"], os_type="linux"
        )
        run_dir = f"/tmp/openroad-transport-audit-{uuid.uuid4().hex} quote' $literal"
        try:
            assert await session.directory_exists("/tmp")
            await session.interface.create_dir(run_dir)
            program = (
                "import json,time; time.sleep(6); "
                "print(json.dumps({'normalized_score': 0.5, 'literal': '$?'})); "
                "raise SystemExit(1)"
            )
            stdout, stderr = await task._run_verifier_background(
                session, ["python3", "-c", shlex.quote(program)], tag="diagnostic", run_dir=run_dir
            )
            assert json.loads(stdout) == {"normalized_score": 0.5, "literal": "$?"}
            assert not stderr.strip()
            assert len(launch_durations) == 1
            assert launch_durations[0] < 3.0, launch_durations
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.skipif(not os.environ.get("OPENROAD_AUDIT_CUA_URL"), reason="No retained audit guest")
@pytest.mark.parametrize("return_code,score", [(0, 1.0), (1, 0.5), (2, 0.0)])
def test_real_guest_background_transport(monkeypatch, return_code, score):
    from cua_bench.computers.remote import RemoteDesktopSession

    monkeypatch.setattr(task, "VERIFIER_POLL_INTERVAL_S", 0.1)

    async def run():
        session = RemoteDesktopSession(
            api_url=os.environ["OPENROAD_AUDIT_CUA_URL"], os_type="linux"
        )
        run_dir = f"/tmp/openroad-transport-audit-{uuid.uuid4().hex}"
        try:
            assert await session.directory_exists("/tmp")
            await session.interface.create_dir(run_dir)
            program = f"import json; print(json.dumps({{'normalized_score': {score}}})); raise SystemExit({return_code})"
            coroutine = task._run_verifier_background(
                session, ["python3", "-c", shlex.quote(program)], tag="diagnostic", run_dir=run_dir
            )
            if return_code == 2:
                with pytest.raises(RuntimeError, match="verifier exited unexpectedly"):
                    await coroutine
            else:
                stdout, stderr = await coroutine
                assert json.loads(stdout)["normalized_score"] == score
                assert not stderr.strip()
        finally:
            await session.close()

    asyncio.run(run())
