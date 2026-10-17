"""Env-substitution rules for the config loader.

Regression guard: `${env:VAR}` refs live in the *documentation comments* of
every shipped agent preset (`configs/agents/*.yaml`). Substitution runs over
raw file text before parsing, so those comments must not be able to fail a
load.
"""

from pathlib import Path
import re

import pytest
import yaml

from ale_run.orchestration import config_loader
from ale_run.orchestration.config_loader import load_experiment

REPO_ROOT = Path(config_loader.__file__).resolve().parents[2]
SHIPPED_AGENT_CONFIGS = sorted((REPO_ROOT / "configs" / "agents").glob("*.yaml"))


def _write_experiment(tmp_path: Path, agent: Path) -> Path:
    # Same shape as tests/ale_run/test_auto_resume.py::_write_experiment.
    (tmp_path / "environment.yaml").write_text(
        "provider: static\nendpoint: http://127.0.0.1:5000\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "name: env-substitution-test\n"
        f"agent: {agent.as_posix()}\n"
        "environment: environment.yaml\n"
        "tasks:\n"
        "  - path: demo/hello\n",
        encoding="utf-8",
    )
    return experiment


@pytest.mark.parametrize("agent_config", SHIPPED_AGENT_CONFIGS, ids=lambda p: p.stem)
def test_shipped_agent_configs_load_with_only_required_env_set(tmp_path, agent_config, monkeypatch):
    """Comment examples do not add requirements; real credentials still do."""
    text = agent_config.read_text()
    for variable in re.findall(r"\$\{env:([A-Z_][A-Z0-9_]*)\}", text):
        monkeypatch.delenv(variable, raising=False)
    parsed = yaml.safe_load(text)
    for variable in re.findall(r"\$\{env:([A-Z_][A-Z0-9_]*)\}", yaml.safe_dump(parsed)):
        monkeypatch.setenv(variable, "test-required-value")
    spec = load_experiment(_write_experiment(tmp_path, agent_config))
    assert spec.agents


def test_unset_ref_in_a_real_value_still_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ALE_TEST_MISSING", raising=False)
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\nconfig:\n  api_key: ${env:ALE_TEST_MISSING}\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="ALE_TEST_MISSING is not set"):
        load_experiment(_write_experiment(tmp_path, agent))


def test_set_ref_substitutes_and_preserves_yaml_typing(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_TEST_KEY", "sk-abc123")
    monkeypatch.setenv("ALE_TEST_TURNS", "42")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\n"
        "config:\n"
        "  api_key: ${env:ALE_TEST_KEY}\n"
        "  max_turns: ${env:ALE_TEST_TURNS}\n",
        encoding="utf-8",
    )
    config = load_experiment(_write_experiment(tmp_path, agent)).agents[0].config
    assert config["api_key"] == "sk-abc123"
    assert config["max_turns"] == 42  # substituted before parse → still an int


def test_hash_inside_a_quoted_scalar_survives(tmp_path, monkeypatch):
    """A `#` in a quoted value is data, not a comment.

    Pins the boundary against a future "just strip comments first" refactor,
    which would truncate this value at the `#`.
    """
    monkeypatch.setenv("ALE_TEST_KEY", "sk-abc123")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\n"
        'config:\n  note: "id #42 uses ${env:ALE_TEST_KEY} # not a comment"\n',
        encoding="utf-8",
    )
    config = load_experiment(_write_experiment(tmp_path, agent)).agents[0].config
    assert config["note"] == "id #42 uses sk-abc123 # not a comment"


def test_hash_inside_a_block_scalar_survives(tmp_path, monkeypatch):
    """Same boundary for a block scalar — the `prompt_suffix: |` shape."""
    monkeypatch.setenv("ALE_TEST_KEY", "sk-abc123")
    (tmp_path / "agent.yaml").write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    (tmp_path / "environment.yaml").write_text(
        "provider: static\nendpoint: http://127.0.0.1:5000\n", encoding="utf-8"
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "name: block-scalar-test\n"
        "agent: agent.yaml\n"
        "environment: environment.yaml\n"
        "tasks:\n"
        "  - path: demo/hello\n"
        "prompt_suffix: |\n"
        "  Write results to out.md # keep this hash\n"
        "  Auth with ${env:ALE_TEST_KEY}\n",
        encoding="utf-8",
    )
    spec = load_experiment(experiment)
    assert "# keep this hash" in spec.prompt_suffix
    assert "sk-abc123" in spec.prompt_suffix


def test_ref_as_a_mapping_key_resolves(tmp_path, monkeypatch):
    """`${env:VAR}` is legal in a key position, not just a value."""
    monkeypatch.setenv("ALE_TEST_KEYNAME", "api_key")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\nconfig:\n  ${env:ALE_TEST_KEYNAME}: sk-from-key\n",
        encoding="utf-8",
    )
    config = load_experiment(_write_experiment(tmp_path, agent)).agents[0].config
    assert config["api_key"] == "sk-from-key"


def test_unset_ref_as_a_mapping_key_still_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ALE_TEST_KEYNAME", raising=False)
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\nconfig:\n  ${env:ALE_TEST_KEYNAME}: v\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="ALE_TEST_KEYNAME is not set"):
        load_experiment(_write_experiment(tmp_path, agent))


def test_commented_ref_does_not_leak_into_parsed_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_TEST_KEY", "sk-should-not-appear")
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "harness: dummy\nmodel: test\nconfig:\n  # api_key: ${env:ALE_TEST_KEY}\n  api_key: null\n",
        encoding="utf-8",
    )
    config = load_experiment(_write_experiment(tmp_path, agent)).agents[0].config
    assert config["api_key"] is None


@pytest.mark.parametrize(
    "comment",
    [
        "# ${env:ALE_TEST_COMMENT}\nvalue: original\n",
        "value: original # ${env:ALE_TEST_COMMENT}\n",
    ],
)
def test_multiline_env_in_comment_cannot_change_yaml(tmp_path, monkeypatch, comment):
    monkeypatch.setenv("ALE_TEST_COMMENT", "ignored\ninjected: true\nvalue: overwritten")
    path = tmp_path / "comment.yaml"
    path.write_text(comment)
    assert config_loader._read_yaml(path) == {"value": "original"}


def test_main_yaml_comment_does_not_add_top_level_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_TEST_COMMENT", "ignored\nunexpected_injected_key: true")
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n")
    experiment = _write_experiment(tmp_path, agent)
    experiment.write_text("# ${env:ALE_TEST_COMMENT}\n" + experiment.read_text())
    assert load_experiment(experiment).name == "env-substitution-test"


def test_missing_ref_in_flow_mapping_still_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.delenv("ALE_TEST_MISSING", raising=False)
    path = tmp_path / "flow.yaml"
    path.write_text("config: {value: ${env:ALE_TEST_MISSING}}\n")
    with pytest.raises(KeyError, match="ALE_TEST_MISSING is not set"):
        config_loader._read_yaml(path)


def test_present_flow_scalar_preserves_yaml_type(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_TEST_COUNT", "42")
    path = tmp_path / "flow.yaml"
    path.write_text("config: {value: ${env:ALE_TEST_COUNT}}\n")
    assert config_loader._read_yaml(path) == {"config": {"value": 42}}


def test_literal_env_reference_inside_replacement_is_not_reinterpreted(tmp_path, monkeypatch):
    monkeypatch.setenv("ALE_TEST_LITERAL", "${env:ALE_TEST_ABSENT}")
    monkeypatch.delenv("ALE_TEST_ABSENT", raising=False)
    path = tmp_path / "literal.yaml"
    path.write_text("value: ${env:ALE_TEST_LITERAL}\n")
    assert config_loader._read_yaml(path) == {"value": "${env:ALE_TEST_ABSENT}"}
