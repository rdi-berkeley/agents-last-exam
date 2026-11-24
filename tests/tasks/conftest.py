import sys

import pytest

# Task main.py files import their scorer by bare name, which TaskLoader isolates
# per task. Test modules import task mains directly, so drop the previous task's
# copy before each test module is imported.
_TASK_LOCAL_MODULE_NAMES = ("score_outputs", "verify_outputs")


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        for name in _TASK_LOCAL_MODULE_NAMES:
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def isolated_judge_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTHLE_EVAL_CREDENTIALS_DIR", str(tmp_path / "judge-credentials"))
    for name in (
        "JUDGE_API_KEY",
        "JUDGE_BASE_URL",
        "JUDGE_MODEL",
        "JUDGE_REASONING_EFFORT",
        "JUDGE_MAX_COMPLETION_TOKENS",
        "AUDIO_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def judge_api_credentials(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "test-judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://shared-judge.test/v1")
