# OpenHands CLI (ALE deployer)

[OpenHands V1 CLI](https://docs.openhands.dev/openhands/usage/cli/headless)
adapted for the ALE framework. The CLI runs *inside* the sandbox as a Python
process; ALE coordinates from outside through the executor.

Uses the official `openhands-cli` pip package (no fork required).

## Architecture

```text
ALE HOST                                 SANDBOX
lifecycle.py
  executor.run_deployer()
    OpenHandsCliDeployer
      install()          -->  pip install openhands-cli==1.15.1
                              write ~/.openhands/.env
                              write ~/.openhands/mcp.json
      launch()           -->  openhands --headless --json --yolo \
                                        --override-with-envs \
                                        --exit-without-confirmation \
                                        -t "<prompt>"
                                    |
                                    +-- stdio events (--JSON Event--)
                                    +-- MCP stdio server (CUA bridge)
                                             +-- OS via cua-mcp-server
      parse_artifacts()  <--  stdout.log -> transcript.jsonl
```

## Supported providers

| Model prefix | LLM_BASE_URL | Auth env |
|---|---|---|
| `openrouter/<vendor>/<model>` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `anthropic/<model>` | _(unset; LiteLLM default)_ | `ANTHROPIC_API_KEY` |

OpenHands uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood,
so any provider/model that LiteLLM understands is callable as long as the
correct env triplet is supplied.

## Bridge

The deployer wires the shared **CUA MCP Server** into OpenHands by writing
`~/.openhands/mcp.json`:

```json
{
  "mcpServers": {
    "cua": {
      "command": "/usr/local/bin/node",
      "args": ["/home/user/cua_mcp_server/src/index.js"]
    }
  }
}
```

## YAML example

```yaml
agent:
  harness: openhands_cli
  model: "anthropic/claude-sonnet-4-6"
  config:
    timeout_s: 600
    cli_version: "1.15.1"
    disable_condenser: false
    max_iterations: 100000
    extra_envs: {}
```

| Field | Meaning |
|---|---|
| `model` | LiteLLM model id (with provider prefix when using OpenRouter) |
| `timeout_s` | Wall-clock cap; the deployer terminates the process on overrun |
| `cli_version` | Version of `openhands-cli` to install from PyPI |
| `disable_condenser` | Sets `OPENHANDS_DISABLE_CONDENSER=1` to suppress the `LLMSummarizingCondenser` |
| `max_iterations` | OpenHands AgentBudget cap (set huge; wall-clock owns termination) |
| `extra_envs` | Optional dict of extra env vars exported to the runner |

## See also

- Implementation guide: `AGENTS.md`
- Upstream docs: <https://docs.openhands.dev/openhands/usage/cli/headless>,
  <https://docs.openhands.dev/openhands/usage/cli/mcp-servers>
