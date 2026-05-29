# Hermes Agent External Integration

[Hermes Agent](https://hermes-agent.nousresearch.com/) is the open-source CLI
agent from Nous Research. ALE runs it as an external agent via the
`cua-verse/hermes-agent` fork on the `agenthle` branch, driven through
`hermes chat -q`.

## Operating Systems

**Linux only.** Upstream explicitly does not support native Windows.

| OS | Status |
|---|---|
| Linux (Ubuntu 22.04) | supported |
| Windows native | upstream does not support |
| macOS | not used in benchmark |

## Architecture

```
ALE RUNNER (host)                    SANDBOX (ale-kasm)
HermesDeployer
    install()  ──────────────→  git clone fork + uv pip install -e ".[all]"
                                write ~/.hermes/.env + config.yaml
    launch()   ──────────────→  setsid bash hermes_runner.sh
                                  hermes chat -q "<prompt>" -Q \
                                    --provider openrouter --model <m> \
                                    --toolsets <full csv incl. mcp-cua> \
                                    --yolo --accept-hooks --ignore-rules \
                                    --max-turns 100000 --pass-session-id
                                  → ~/.hermes/state.db (SQLite)
                                  hermes sessions export transcript.jsonl
    parse_artifacts() ←────────  transcript.jsonl + stdout/stderr
```

The CUA bridge is the standard CUA MCP Server (shared with all CLI agents).
The deployer writes ``~/.hermes/config.yaml`` with a ``mcp_servers.cua`` entry.

### Multimodal screenshot path

`mcp_cua_screenshot` flows through a fork patch (`a1a77735`):

1. fork's `mcp_tool.py` saves the base64 PNG to `~/.hermes/mcp_images/`
2. `AIAgent._expand_tool_image_followups` inserts a `user` message with
   `image_url` content after the tool batch
3. Provider adapters translate to vendor multimodal format

Vision-capable models see screenshots natively at ~1.6K vision tokens per image
instead of carrying multi-MB base64 in conversation history.

## Supported Providers

| Provider | Status |
|---|---|
| `openrouter` | shipped (default) |
| `anthropic` direct | untested |
| `nous` (Nous Portal) | untested |

## Smoke Test

```bash
uv run python -m ale_run run experiments/smoke_hermes_docker.yaml
```

## See Also

- Implementation notes: [AGENTS.md](AGENTS.md)
- Upstream docs: <https://hermes-agent.nousresearch.com/>
- Fork: <https://github.com/cua-verse/hermes-agent> (branch `agenthle`)
