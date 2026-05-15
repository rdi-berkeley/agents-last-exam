# cua-mcp-server (vendored)

JS-based MCP server that wraps the in-VM `cua-computer-server`. In-VM
agent CLIs (claude-code, codex, ...) load this as their MCP transport so
the LLM can drive the VM via standard MCP tool calls.

**Source**: vendored from
`agenthle/agenthle/orchestration/external/bridges/cua_mcp_server` (agenthle
v0.x). Independent of cua-bench / cua submodule.

## Layout

- `package.json` — Node package manifest, depends on
  `@modelcontextprotocol/sdk` and `zod`.
- `src/index.js` — MCP server entry point.
- `src/cua-client.js` — HTTP/WebSocket client to `cua-computer-server`.

## Install on VM

`ale.agents.runtime_install.upload_mcp_server(session, install_paths)`
uploads this directory to `install_paths.mcp_server_dir(os)` and runs
`npm install --production` there.

## Updating

If the in-VM API changes upstream, sync this tree by copying from
agenthle's `bridges/cua_mcp_server/` (the canonical source for now).
Re-running `npm install` on the next VM build refreshes lockfile.
