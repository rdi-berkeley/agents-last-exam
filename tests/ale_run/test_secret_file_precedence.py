"""Precedence between an explicit ``secret_file:`` and the shell environment."""

import os
import shutil
from pathlib import Path

import pytest

from ale_run.orchestration import config_loader
from ale_run.orchestration.config_loader import load_experiment

REPO_ROOT = Path(config_loader.__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _write_experiment(tmp_path: Path, secret_text: str) -> Path:
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / ".env").write_text(secret_text, encoding="utf-8")
    (tmp_path / "agent.yaml").write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    (tmp_path / "environment.yaml").write_text(
        "provider: static\nendpoint: http://127.0.0.1:5000\n", encoding="utf-8"
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "name: secret-precedence-test\n"
        "secret_file: secret/.env\n"
        "agent: agent.yaml\n"
        "environment: environment.yaml\n"
        "tasks:\n"
        "  - path: demo/hello\n",
        encoding="utf-8",
    )
    return experiment


def test_empty_value_keeps_shell_value(tmp_path):
    os.environ["ALE_TEST_SECRET"] = "from-shell"
    load_experiment(_write_experiment(tmp_path, "ALE_TEST_SECRET=\n"))
    assert os.environ["ALE_TEST_SECRET"] == "from-shell"


def test_non_empty_value_overrides_shell_value(tmp_path):
    os.environ["ALE_TEST_SECRET"] = "from-shell"
    load_experiment(_write_experiment(tmp_path, "ALE_TEST_SECRET=from-file\n"))
    assert os.environ["ALE_TEST_SECRET"] == "from-file"


def test_empty_value_still_defines_an_unset_key(tmp_path):
    os.environ.pop("ALE_TEST_SECRET", None)
    load_experiment(_write_experiment(tmp_path, "ALE_TEST_SECRET=\n"))
    assert os.environ["ALE_TEST_SECRET"] == ""


def test_unfilled_env_example_keeps_exported_api_key(tmp_path):
    """README quick start: ``cp -n secret/.env.example secret/.env``, key exported in the shell."""
    experiment = _write_experiment(tmp_path, "")
    shutil.copyfile(REPO_ROOT / "secret" / ".env.example", tmp_path / "secret" / ".env")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-from-shell"
    os.environ["OPENROUTER_API_KEY"] = "sk-or-from-shell"
    load_experiment(experiment)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-shell"
