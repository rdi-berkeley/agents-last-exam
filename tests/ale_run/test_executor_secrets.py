from __future__ import annotations

from dataclasses import dataclass, field

from ale_run.executors._secrets import (
    config_to_kwargs_and_secrets,
    restore_config_secrets,
)


@dataclass
class _Config:
    model: str = "model"
    api_key: str | None = "primary-secret"
    vision_api_key: str | None = "vision-secret"
    api_key_env: str = "MODEL_API_KEY"
    access_token: str = "ordinary-value"
    custom_credential: str = field(
        default="metadata-secret",
        metadata={"secret": True},
    )


def test_config_secrets_are_kept_out_of_serialized_kwargs() -> None:
    kwargs, secrets = config_to_kwargs_and_secrets(_Config())

    assert kwargs == {
        "model": "model",
        "api_key": None,
        "vision_api_key": None,
        "api_key_env": "MODEL_API_KEY",
        "access_token": "ordinary-value",
        "custom_credential": None,
    }
    assert set(secrets.values()) == {
        "primary-secret",
        "vision-secret",
        "metadata-secret",
    }
    assert all(secret not in repr(kwargs) for secret in secrets.values())


def test_config_secrets_round_trip_without_polluting_executor_env() -> None:
    kwargs, secrets = config_to_kwargs_and_secrets(_Config())
    restored, clean_env = restore_config_secrets(
        kwargs,
        {"OPENAI_API_KEY": "env-secret", **secrets},
    )

    assert restored["api_key"] == "primary-secret"
    assert restored["vision_api_key"] == "vision-secret"
    assert restored["custom_credential"] == "metadata-secret"
    assert clean_env == {"OPENAI_API_KEY": "env-secret"}
