import json
import os
from pathlib import Path
import subprocess

import pytest

from ale_run.agents.zcode import ZCodeConfig, ZCodeDeployer


def test_pinned_registry_defaults_reach_real_sdk_transport():
    source = os.environ.get("ALE_ZCODE_UPSTREAM")
    if not source:
        pytest.skip("Set ALE_ZCODE_UPSTREAM to the installed pinned ZCode source")
    upstream = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    assert revision == ZCodeConfig().cli_ref
    builtin_path = upstream / "apps/zcode-cli/packages/cli/dist/provider/zcode-builtin.json"
    upstream_builtin = json.loads(builtin_path.read_text(encoding="utf-8"))
    cases = []
    for model, limit in [
        ("ale-unregistered-test-model", None),
        ("glm-5.3", None),
        ("glm-5.3", 12345),
        ("glm-5.3", 131072),
    ]:
        config = ZCodeConfig(
            model=model,
            provider_id="zai",
            api_key="mock-key",
            base_url="https://mock.invalid/v1",
            context_window=1048576,
            reasoning_request={"reasoning_effort": "max"},
            max_output_tokens=limit,
        )
        cases.append(
            {
                "model": model,
                "override": limit,
                "contextWindow": config.context_window,
                "builtin": ZCodeDeployer._build_builtin_config(config, upstream_builtin),
                "personal": ZCodeDeployer._build_personal_config(config),
            }
        )
    result = subprocess.run(
        [
            str(upstream / "node_modules/.bin/tsx"),
            str(Path(__file__).parent / "fixtures/zcode_provider_transport.mts"),
            str(upstream),
        ],
        input=json.dumps({"upstreamBuiltin": upstream_builtin, "cases": cases}),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads(result.stdout)
    assert len(reports) == len(cases)
    for report in reports:
        assert report["request"]["max_completion_tokens"] == report["effectiveMaxOutputTokens"]
        assert report["request"]["reasoning_effort"] == "max"
        assert "max_tokens" not in report["request"]
        assert "max_output_tokens" not in report["request"]
    print(json.dumps(reports, indent=2))
