from __future__ import annotations

import json
from types import SimpleNamespace

from ale_run.agents.openclaw_cli import OpenClawCliConfig, OpenClawCliDeployer


def test_write_config_supports_zai_primary_and_direct_openai_vision(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = OpenClawCliConfig(
        model="glm-5.2",
        provider="zai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_params={
            "temperature": 1.0,
            "extra_body": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
        },
        thinking="max",
        vision_model="openai/gpt-5.4",
        vision_provider="direct",
    )
    executor = SimpleNamespace(
        config=cfg,
        env={
            "GLM_API_KEY": "zai-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        },
        cua_bridge_url=lambda: "http://127.0.0.1:5000",
    )
    deployer = OpenClawCliDeployer(executor)

    deployer._write_config(cfg)

    openclaw_config = json.loads(
        (tmp_path / ".openclaw" / "openclaw.json").read_text()
    )
    defaults = openclaw_config["agents"]["defaults"]
    assert defaults["model"]["primary"] == "zai/glm-5.2"
    assert defaults["imageModel"]["primary"] == "openai/gpt-5.4"
    assert defaults["models"]["zai/glm-5.2"]["params"] == cfg.model_params
    assert defaults["models"]["openai/gpt-5.4"] == {}
    assert openclaw_config["tools"]["media"]["image"]["models"] == [
        {"provider": "openai", "model": "gpt-5.4"}
    ]
    assert {"cua", "zai", "openai"} <= set(
        openclaw_config["plugins"]["allow"]
    )
    assert openclaw_config["models"]["providers"] == {
        "zai": {
            "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
            "api": "openai-completions",
            "models": [{"id": "glm-5.2", "name": "glm-5.2"}],
        }
    }

    auth = json.loads(
        (
            tmp_path
            / ".openclaw"
            / "agents"
            / "main"
            / "agent"
            / "auth-profiles.json"
        ).read_text()
    )
    assert auth["profiles"] == {
        "zai:default": {
            "provider": "zai",
            "type": "api_key",
            "key": "zai-test-key",
        },
        "openai:default": {
            "provider": "openai",
            "type": "api_key",
            "key": "openai-test-key",
        },
    }
    assert auth["lastGood"] == {
        "zai": "zai:default",
        "openai": "openai:default",
    }


def test_write_config_supports_custom_primary_and_direct_openai_vision(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = OpenClawCliConfig(
        model="Azure-Code",
        provider="custom",
        provider_id="azure-code",
        model_id="rc6-for-vlm-team",
        base_url="https://example.openai.azure.com/openai/v1",
        api_key_env="AZURE_CODE_API_KEY",
        provider_api="openai-completions",
        model_params={"temperature": 1.0},
        thinking="off",
        vision_model="openai/gpt-5.4",
        vision_provider="direct",
        plugins_allow=("cua", "memory-core"),
    )
    executor = SimpleNamespace(
        config=cfg,
        env={
            "AZURE_CODE_API_KEY": "azure-test-key",
            "OPENAI_API_KEY": "openai-test-key",
        },
        cua_bridge_url=lambda: "http://127.0.0.1:5000",
    )
    deployer = OpenClawCliDeployer(executor)

    deployer._write_config(cfg)

    openclaw_config = json.loads(
        (tmp_path / ".openclaw" / "openclaw.json").read_text()
    )
    defaults = openclaw_config["agents"]["defaults"]
    assert defaults["model"]["primary"] == "azure-code/rc6-for-vlm-team"
    assert defaults["imageModel"]["primary"] == "openai/gpt-5.4"
    assert defaults["models"]["azure-code/rc6-for-vlm-team"]["params"] == {
        "temperature": 1.0,
    }
    assert openclaw_config["tools"]["media"]["image"]["models"] == [
        {"provider": "openai", "model": "gpt-5.4"},
    ]
    assert "openai" in openclaw_config["plugins"]["allow"]
    assert "azure-code" not in openclaw_config["plugins"]["allow"]
    assert openclaw_config["models"]["providers"] == {
        "azure-code": {
            "baseUrl": "https://example.openai.azure.com/openai/v1",
            "api": "openai-completions",
            "models": [
                {
                    "id": "rc6-for-vlm-team",
                    "name": "Azure-Code",
                    "compat": {
                        "maxTokensField": "max_completion_tokens",
                        "supportsUsageInStreaming": True,
                    },
                },
            ],
        },
    }

    auth = json.loads(
        (
            tmp_path
            / ".openclaw"
            / "agents"
            / "main"
            / "agent"
            / "auth-profiles.json"
        ).read_text()
    )
    assert auth["profiles"] == {
        "azure-code:default": {
            "provider": "azure-code",
            "type": "api_key",
            "key": "azure-test-key",
        },
        "openai:default": {
            "provider": "openai",
            "type": "api_key",
            "key": "openai-test-key",
        },
    }
