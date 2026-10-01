import asyncio
import ast
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tasks.health_medicine.simglucose_safe_basal_control_instance_1 import main as task


AUDITED_SUMMARY = {
    "episodes": 72,
    "mean_tir_70_180": 0.4710202991452991,
    "catastrophic_episode_count": 21,
    "eligible_for_ranking": False,
}
AUDITED_SCORE = 0.28079937928428617


class FakeSession:
    def __init__(self):
        self.meta = task.config.to_metadata()
        self.status_path = self.meta["eval_status_path"] + ".transporttest"
        self.summary_path = self.meta["hidden_summary_path"] + ".transporttest"
        self.files = {
            self.status_path: "0\n",
            self.summary_path: json.dumps({"summary": AUDITED_SUMMARY}),
        }
        self.submission_paths = {
            self.meta["submission_dir"],
            self.meta["controller_output"],
            self.meta["metadata_output"],
        }
        self.status_states = []
        self.run_command = AsyncMock(side_effect=self._launch)
        self.file_exists = AsyncMock(side_effect=self._file_exists)
        self.directory_exists = AsyncMock(return_value=False)
        self.read_file = AsyncMock(side_effect=self._read_file)

    async def _launch(self, command, *, check=True):
        return {"return_code": 0, "stdout": "123\n", "stderr": ""}

    async def _file_exists(self, path):
        if path == self.status_path and self.status_states:
            state = self.status_states.pop(0)
            if isinstance(state, Exception):
                raise state
            return state
        return path in self.files or path in self.submission_paths

    async def _read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(task.uuid, "uuid4", lambda: SimpleNamespace(hex="transporttest"))
    return FakeSession()


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    async def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(task, "time", SimpleNamespace(monotonic=lambda: state.now))
    monkeypatch.setattr(task, "asyncio", SimpleNamespace(timeout=asyncio.timeout, sleep=sleep))
    monkeypatch.setattr(task, "EVAL_TIMEOUT_S", 30)
    monkeypatch.setattr(task, "EVAL_TRANSPORT_TIMEOUT_S", 5)
    monkeypatch.setattr(task, "EVAL_POLL_INTERVAL_S", 5)
    return state


def evaluate(session):
    return asyncio.run(task.evaluate(SimpleNamespace(metadata=session.meta), session))


def test_default_budget_is_bounded_and_allows_a_full_evaluation():
    assert task.EVAL_TIMEOUT_S == 3600
    assert task.EVAL_POLL_INTERVAL_S == 15
    assert task.EVAL_TRANSPORT_TIMEOUT_S == 60


def test_completed_job_recovers_audited_score_from_file(session):
    assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    session.read_file.assert_any_await(session.summary_path)
    assert session.run_command.await_count == 1
    assert session.run_command.await_args.kwargs == {"check": False}


def test_running_job_is_polled_before_result_retrieval(session, clock):
    session.status_states = [False, False, True]
    assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    assert clock.sleeps == [5, 5]
    assert [call.args[0] for call in session.read_file.await_args_list] == [
        session.status_path,
        session.summary_path,
    ]
    session.run_command.assert_awaited_once()


def test_launcher_detaches_fds_bounds_guest_and_writes_atomic_status(session):
    assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    command = shlex.split(session.run_command.await_args.args[0])
    assert command[:2] == ["python3", "-c"]
    launch = ast.parse(command[2])
    calls = [node for node in ast.walk(launch) if isinstance(node, ast.Call)]
    popen = next(node for node in calls if ast.unparse(node.func) == "subprocess.Popen")
    options = {keyword.arg: ast.unparse(keyword.value) for keyword in popen.keywords}
    assert options == {
        "stdin": "subprocess.DEVNULL",
        "stdout": "log",
        "stderr": "subprocess.STDOUT",
        "close_fds": "True",
        "start_new_session": "True",
    }
    argv = ast.literal_eval(popen.args[0])
    assert argv[:2] == ["bash", "-lc"]
    wrapper = shlex.split(argv[2])
    assert wrapper[:5] == ["timeout", "--kill-after=30s", "3600s", "bash", "-lc"]
    assert "printf '%s\\n' \"$rc\"" in argv[2]
    assert f"mv {session.status_path}.tmp {session.status_path}" in argv[2]
    evaluator_script = wrapper[5]
    assert "PYTHONPATH=input:reference uv run --project" in evaluator_script
    assert f"--submission-dir {session.meta['submission_dir']}" in evaluator_script
    assert f"--output {session.summary_path}" in evaluator_script
    assert "cat " not in evaluator_script


def test_each_invocation_uses_new_result_paths(session, monkeypatch):
    identifiers = iter(["first", "second"])
    monkeypatch.setattr(task.uuid, "uuid4", lambda: SimpleNamespace(hex=next(identifiers)))
    for identifier in ("first", "second"):
        session.files[session.meta["eval_status_path"] + "." + identifier] = "0\n"
        session.files[session.meta["hidden_summary_path"] + "." + identifier] = session.files[
            session.summary_path
        ]
        assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    commands = [call.args[0] for call in session.run_command.await_args_list]
    assert commands[0] != commands[1]
    assert session.meta == task.config.to_metadata()


@pytest.mark.parametrize("status", ["1", "2", "124", "137", "-9", "", "unknown", None])
def test_nonzero_or_invalid_status_is_an_error_even_with_result(session, status):
    session.files[session.status_path] = status
    with pytest.raises(RuntimeError, match="hidden eval failed: status="):
        evaluate(session)
    assert session.summary_path not in [call.args[0] for call in session.read_file.await_args_list]


def test_missing_result_is_an_error_after_successful_exit(session):
    del session.files[session.summary_path]
    with pytest.raises(RuntimeError, match="hidden eval result missing"):
        evaluate(session)


@pytest.mark.parametrize("raw", ["", "truncated{", "null", "[]", "{}", '{"summary": null}'])
def test_invalid_result_is_never_a_zero(session, raw):
    session.files[session.summary_path] = raw
    with pytest.raises(RuntimeError, match="hidden summary"):
        evaluate(session)


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes", None),
        ("episodes", 0),
        ("episodes", True),
        ("mean_tir_70_180", None),
        ("mean_tir_70_180", "0"),
        ("mean_tir_70_180", False),
        ("mean_tir_70_180", float("nan")),
        ("mean_tir_70_180", float("inf")),
        ("mean_tir_70_180", -0.1),
        ("mean_tir_70_180", 1.1),
        ("catastrophic_episode_count", None),
        ("catastrophic_episode_count", -1),
        ("catastrophic_episode_count", 73),
        ("catastrophic_episode_count", False),
    ],
)
def test_invalid_score_inputs_do_not_use_scorer_defaults(session, field, value):
    summary = dict(AUDITED_SUMMARY, **{field: value})
    session.files[session.summary_path] = json.dumps({"summary": summary})
    with pytest.raises(RuntimeError, match="invalid hidden summary score inputs"):
        evaluate(session)


@pytest.mark.parametrize("field", ["episodes", "mean_tir_70_180", "catastrophic_episode_count"])
def test_missing_score_inputs_are_errors(session, field):
    summary = dict(AUDITED_SUMMARY)
    del summary[field]
    session.files[session.summary_path] = json.dumps({"summary": summary})
    with pytest.raises(RuntimeError, match="invalid hidden summary score inputs"):
        evaluate(session)


@pytest.mark.parametrize("mean_tir,catastrophic,score", [(0, 0, 0), (1, 72, 0), (1, 0, 1)])
def test_legitimate_scores_including_zero_are_unchanged(session, mean_tir, catastrophic, score):
    summary = dict(
        AUDITED_SUMMARY, mean_tir_70_180=mean_tir, catastrophic_episode_count=catastrophic
    )
    session.files[session.summary_path] = json.dumps({"summary": summary}).encode()
    session.files[session.status_path] = b"0\n"
    assert evaluate(session) == [score]


@pytest.mark.parametrize("key", ["submission_dir", "controller_output", "metadata_output"])
def test_missing_candidate_artifacts_remain_legitimate_zero(session, key):
    session.submission_paths.remove(session.meta[key])
    assert evaluate(session) == [0.0]
    session.run_command.assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError("REST unavailable"), TimeoutError("transport")])
def test_transient_poll_transport_recovers_without_relaunch(session, clock, failure):
    session.status_states = [failure, False, True]
    assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    assert clock.sleeps == [5, 5]
    session.run_command.assert_awaited_once()


@pytest.mark.parametrize("failed_path", ["status_path", "summary_path"])
def test_transient_file_transport_recovers_without_relaunch(session, clock, failed_path):
    failures = [OSError("read transport interrupted")]

    async def read_file(path):
        if path == getattr(session, failed_path) and failures:
            raise failures.pop()
        return await session._read_file(path)

    session.read_file.side_effect = read_file
    assert evaluate(session) == pytest.approx([AUDITED_SCORE])
    assert clock.sleeps == [5]
    session.run_command.assert_awaited_once()


def test_running_job_reaches_bounded_deadline_without_a_score(session, clock):
    del session.files[session.status_path]
    with pytest.raises(TimeoutError, match="hidden eval deadline exceeded"):
        evaluate(session)
    assert clock.now == 35
    session.read_file.assert_not_awaited()
    session.run_command.assert_awaited_once()


def test_persistent_transport_error_preserves_cause_at_deadline(session, clock):
    failure = RuntimeError("persistent read failure")
    session.read_file.side_effect = failure
    with pytest.raises(TimeoutError, match="deadline exceeded") as raised:
        evaluate(session)
    assert raised.value.__cause__ is failure
    assert clock.now == 35
    session.run_command.assert_awaited_once()


def test_transport_call_time_counts_toward_deadline(session, clock):
    async def slow_read(path):
        clock.now += 20
        raise TimeoutError("slow transport")

    session.read_file.side_effect = slow_read
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        evaluate(session)
    assert session.read_file.await_count == 2
    assert clock.now == 45


@pytest.mark.parametrize(
    "reply",
    [
        {"return_code": 1, "stderr": "cannot spawn"},
        {"stdout": "123\n"},
        {"return_code": 0, "stdout": ""},
        {"return_code": 0, "stdout": "not-a-pid"},
        {"return_code": 0, "stdout": "0"},
    ],
)
def test_launch_failure_is_an_error_and_does_not_poll(session, reply):
    session.run_command.side_effect = None
    session.run_command.return_value = reply
    with pytest.raises(RuntimeError, match="hidden eval launch"):
        evaluate(session)
    session.read_file.assert_not_awaited()
    session.run_command.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError("connection lost"), TypeError("bad response")])
def test_ambiguous_launch_transport_is_not_retried_or_scored(session, failure):
    session.run_command.side_effect = failure
    with pytest.raises(RuntimeError, match="hidden eval launch failed") as raised:
        evaluate(session)
    assert raised.value.__cause__ is failure
    session.run_command.assert_awaited_once()
    session.read_file.assert_not_awaited()


def test_hung_launch_is_bounded_without_session_timeout_keyword(session, monkeypatch):
    async def hang(command, *, check=True):
        await asyncio.Event().wait()

    monkeypatch.setattr(task, "EVAL_TRANSPORT_TIMEOUT_S", 0.01)
    session.run_command.side_effect = hang
    with pytest.raises(RuntimeError, match="hidden eval launch failed") as raised:
        evaluate(session)
    assert isinstance(raised.value.__cause__, TimeoutError)
    session.run_command.assert_awaited_once()


def test_hung_poll_transport_is_bounded(session, clock, monkeypatch):
    async def hang(path):
        await asyncio.Event().wait()

    monkeypatch.setattr(task, "EVAL_TRANSPORT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(task, "EVAL_TIMEOUT_S", 0.01)
    session.read_file.side_effect = hang
    with pytest.raises(TimeoutError, match="deadline exceeded") as raised:
        evaluate(session)
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert clock.now == pytest.approx(0.02)


def test_submission_probe_transport_is_not_a_zero(session):
    session.file_exists.side_effect = RuntimeError("probe transport failed")
    with pytest.raises(RuntimeError, match="probe transport failed"):
        evaluate(session)
    session.run_command.assert_not_awaited()
