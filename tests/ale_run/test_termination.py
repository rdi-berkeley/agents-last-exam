from __future__ import annotations

from ale_run.orchestration.termination import redact_config


def test_redact_config_redacts_nested_api_keys() -> None:
    config = {
        "api_key": "primary-secret",
        "vision_api_key": "vision-secret",
        "nested": {
            "custom_api_key": "nested-secret",
            "safe": "value",
        },
    }

    assert redact_config(config) == {
        "api_key": "***cret",
        "vision_api_key": "***cret",
        "nested": {
            "custom_api_key": "***cret",
            "safe": "value",
        },
    }


def test_redact_config_preserves_non_string_nested_keys() -> None:
    config = {
        "parameters": {1: "safe", 2: {"api_key": "nested-secret"}, None: "unset"},
    }

    assert redact_config(config) == {
        "parameters": {1: "safe", 2: {"api_key": "***cret"}, None: "unset"},
    }
    assert config["parameters"][2]["api_key"] == "nested-secret"
