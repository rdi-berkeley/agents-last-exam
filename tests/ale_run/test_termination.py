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
