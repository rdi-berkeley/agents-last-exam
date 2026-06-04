# Plan: route ale_claw through MCP (keep tool granularity)

Goal: make the **native** agent `ale_claw` reach the VM through the same MCP
servers that **installed** agents use — `cua_mcp_server` (GUI) and
`vm_mcp_server` (non-GUI) — instead of calling `cb.RemoteDesktopSession`
(the "DesktopSession") directly. Tool *granularity* is preserved: ale_claw
keeps its thick `read`/`write`/`edit`/`exec` tools; only the I/O transport
underneath them changes.

This is **Option B** (composite tools over an MCP backend), chosen over Option A
(drop to raw vm primitives) because the value-add logic — adaptive line paging,
image sanitize, `edit` exact-match + mismatch recovery, `exec` middle-truncation
+ timeout + cwd policy — is eval-relevant and transport-agnostic.

> **Branch base:** `feat/ale-claw-mcp` = the `audit/ale-claw-readability` refactor
> + a merge of `origin/main` (brings in `vm_mcp_server`, commit `efdee29`). The
> readability refactor regrouped the harness tool modules into a `tools/`
> subpackage — anchors below reflect that layout.

## Why it works (the key fact)

ale_claw's thick tools touch the substrate through a **4-operation surface only**:

| Operation | Today (`VMBackend` / `ExecTool`) | vm MCP equivalent |
|---|---|---|
| `read_bytes(path)` | `session.interface.read_bytes` | `read_bytes` (base64 → decode) |
| `write_text(path, content, append)` | `session.interface.write_text` | `write_text` (overwrite) / `write_bytes(append=true)` |
| `create_dir(path)` | `session.interface.create_dir` | `run_command("mkdir -p …")` (vm MCP omits mkdir by design) |
| `run_command(cmd)` (exec) | `session.interface.run_command` | `run_command` |

`resolve(path)` is pure path policy (no I/O) and stays in the harness. Everything
above the backend (`ReadFileTool`/`WriteFileTool`/`EditFileTool` in
`harness/tools/tools_fs.py`, `ExecTool` in `harness/tools/tools_shell.py`) is
unchanged.

Seam confirmed in code (post-refactor `tools/` subpackage):
- `harness/tools/fs_backends.py:40-69` — `FilesystemBackend` ABC = exactly
  `resolve` / `read_bytes` / `write_text` / `create_dir`; it imports
  `_assert_within_workspace` from `._paths` (`fs_backends.py:29`).
- `harness/tools/tools.py:242-261` — where backends and `ExecTool` are constructed
  (`VMBackend` register at :243, `ExecTool(session.interface, …)` at :261). Line
  numbers unchanged by the refactor.
- Python `mcp` SDK **1.26.0** is already installed (`stdio_client`,
  `StdioServerParameters`, `ClientSession.initialize/list_tools/call_tool`).
- The OpenClaw runtime (`cua-agent`, editable at
  `agenthle-base/submodules/cua/libs/python/agent`) has **no MCP client** — so we
  add a thin client ourselves (the harness consumes MCP as a *backend*, it does
  not need to expose MCP tools to the model).

## Scope / phasing

**Phase 1 (this plan): non-GUI → MCP.** Route `read`/`write`/`edit`/`exec` through
`vm_mcp_server`. This removes the bulk of the "confusing" DesktopSession surface
(the bugged `run_command`/fs/clipboard/window/python_exec methods in
`SESSION_API.md §13`). GUI stays on `session.computer`.

**Phase 2 (implemented): GUI → cua MCP.** The `computer` tool routes through
`cua_mcp_server` when `gui_transport="mcp"` (requires `substrate_transport="mcp"`).
`harness/computer_handler.MCPComputerHandler` implements the `AsyncComputerHandler`
protocol over the cua bridge:
- **Coordinates:** the model emits pixel coords in screenshot space; the cua
  bridge speaks normalized [0,1000]. The handler converts px→[0,1000] using the
  screen size from a new `get_screen_size` bridge tool (the bridge previously
  exposed no pixel dimensions); the bridge converts back. Rounding ≤ 1px.
- **Keypress:** the chord-vs-sequence rule stays in the *handler* (a list → one
  `key` call per key = sequence; a `+`/`-` string → one `key` with all keys =
  chord), so no bridge change was needed for it — only `get_screen_size` was
  added to the cua bridge.
- Wiring: deployer adds the `cua` server to the `MCPRuntime` and builds
  `MCPComputerHandler` (lazy-init; the runtime connects around the drive loop);
  `gui_transport` config knob (default `session`).

> With `substrate_transport=mcp` + `gui_transport=mcp`, ale_claw no longer touches
> `RemoteDesktopSession` for tool I/O (the session is still constructed for
> connection/setup). Verified: GUI actions (dims/screenshot/move/keypress/click)
> drive the live VM over the cua bridge, and a full `hello_win` run with both
> bridges completed score 1.0.

## Components

### 1. Bootstrap parity — `ale_run/agents/_bootstrap.py`

Mirror the existing cua helpers for the vm bridge:

- `_VM_BRIDGE_SRC = .../ _assets/vm_mcp_server` (sibling of `_CUA_BRIDGE_SRC`).
- `_vm_bridge_installed(dir)` — same `src/index.js` + `@modelcontextprotocol/sdk`
  check as `_cua_bridge_installed` (`_bootstrap.py:183-199`).
- `ensure_vm_mcp_server(target_dir)` — copy source (skip `node_modules`) +
  `npm install --production`, identical to `ensure_cua_mcp_server`
  (`_bootstrap.py:202-279`).
- `vm_bridge_env(executor)` → `{"CUA_SERVER_URL": executor.cua_bridge_url()}`
  (same as `cua_bridge_env`, `_bootstrap.py:282-291`).

**Host-install nuance for the `local` executor.** claude_code installs into
`sandbox.mcp_server_dir` because it runs *inside* the sandbox VM. ale_claw runs on
the **host** (`local` executor). So `ensure_*_mcp_server` for the native case
should install into a **host** dir (e.g. `<work_dir>/mcp/cua` and
`<work_dir>/mcp/vm`, or a cached `~/.cache/ale/mcp/...`). Node on the host comes
from the existing `ensure_node_npm()` (`_bootstrap.py:113`). Generalize the
ensure functions to take an explicit target dir rather than reading
`sandbox.mcp_server_dir`, or add `*_host` wrappers.

### 2. MCP client lifecycle — new `harness/tools/mcp_runtime.py`

A small async manager that spawns the stdio server(s) and owns their
`ClientSession`s for the life of an episode.

```python
# sketch
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPRuntime:
    """Spawns vm (and optionally cua) MCP stdio servers, holds live sessions.

    One instance per episode. `call(server, tool, args)` -> CallToolResult.
    Async context manager: __aenter__ starts servers + initialize(); __aexit__
    tears them down (terminates the node child processes)."""

    def __init__(self, servers: dict[str, StdioServerParameters]): ...
    async def __aenter__(self) -> "MCPRuntime":
        self._stack = AsyncExitStack()
        for name, params in self._servers.items():
            read, write = await self._stack.enter_async_context(stdio_client(params))
            sess = await self._stack.enter_async_context(ClientSession(read, write))
            await sess.initialize()
            self._sessions[name] = sess
        return self
    async def __aexit__(self, *exc): await self._stack.aclose()
    async def call(self, server: str, tool: str, args: dict): ...
```

`StdioServerParameters` per server:
```python
StdioServerParameters(
    command=node_path,
    args=[f"{vm_bridge_dir}/src/index.js"],
    env={**os.environ, "CUA_SERVER_URL": sb.endpoint},   # = cua_bridge_url()
)
```

Concurrency note: a single `ClientSession` is not safe under concurrent
`call_tool`. ale_claw's loop is serial per agent, but subagents (`delegate_gui`,
`delegate_general`) can run tools too — guard each session with an
`asyncio.Lock` (mirrors the `session` concurrency caveat in `SESSION_API.md §14`).

### 3. `MCPBackend` — `harness/tools/fs_backends.py`

New `FilesystemBackend` implementation; registered as the `vm` target in place
of `VMBackend` when MCP is enabled. (`_assert_within_workspace` is already
imported from `._paths` in this module.)

```python
class MCPBackend(FilesystemBackend):
    name = "vm"
    capabilities = frozenset({"read", "write", "edit"})

    def __init__(self, runtime: MCPRuntime, workspace_root: str | None = None): ...

    def resolve(self, path): _assert_within_workspace(path, self.workspace_root); return path

    async def read_bytes(self, path) -> bytes:
        res = await self.runtime.call("vm", "read_bytes", {"path": path})
        return base64.b64decode(_text(res))          # vm MCP read_bytes returns base64

    async def write_text(self, path, content, *, append):
        if append:
            await self.runtime.call("vm", "write_bytes", {
                "path": path, "content_b64": b64(content.encode()), "append": True})
        else:
            await self.runtime.call("vm", "write_text", {"path": path, "content": content})

    async def create_dir(self, path):
        await self.runtime.call("vm", "run_command",
                                {"command": f'mkdir -p {shlex.quote(path)}'})
```

Keeps `name = "vm"` so the tool `target` enum, descriptions, and the agent's
vocabulary are byte-identical. `HostBackend` is untouched (host I/O can't and
shouldn't go through the vm MCP — it's the framework host's own fs for
journaling).

### 4. Exec adapter — `harness/tools/tools_shell.py` (no change to `ExecTool`)

`ExecTool` calls `self.interface.run_command(wrapped)` and reads
`.stdout/.stderr/.returncode` (`tools_shell.py:226-247`). Provide a duck-typed
shim so `ExecTool` itself is unchanged:

```python
class _MCPExecInterface:
    def __init__(self, runtime: MCPRuntime): self.runtime = runtime
    async def run_command(self, command: str):
        res = await self.runtime.call("vm", "run_command", {"command": command})
        sc = res.structuredContent
        if sc is None:
            raise RuntimeError("vm MCP run_command returned no structuredContent")
        return _CmdResult(stdout=sc.get("stdout", ""), stderr=sc.get("stderr", ""),
                          returncode=int(sc.get("exit_code", 0)))
```

**Decision: fix the server with a schema-validated structured payload, do not
text-parse.** The current `run_command` flattens `return_code`/`stdout`/`stderr`
into one text block (`vm_mcp_server/src/index.js:189-192`), which is **lossy** —
stdout can contain a line `stderr:`, the stderr section is conditionally present,
and `join("\n")` isn't reversible. We own `vm_mcp_server` and **no agent consumes
it yet**, so convert `run_command` to the `registerTool` form with an
`outputSchema` and return BOTH a (readable, unchanged) text block and a
schema-validated `structuredContent`. SDK support is confirmed: JS
`@modelcontextprotocol/sdk ^1.12.1` (structured output, ≥1.10) and Python `mcp`
1.26 (`CallToolResult.structuredContent`, `types.py:1130`).

```js
// vm_mcp_server/src/index.js — run_command (registerTool form; other tools unchanged)
server.registerTool("run_command", {
  description: "Run a command on the VM and wait … Returns stdout, stderr, and the exit code.",
  inputSchema:  { command: z.string().describe("…"), timeout: z.number().optional().describe("…") },
  outputSchema: { exit_code: z.number().int(), stdout: z.string(), stderr: z.string() },
}, async ({ command, timeout }) => {
  const params = { command };
  if (timeout !== undefined) params.timeout = timeout;
  const clientTimeout = timeout !== undefined ? (timeout + 30) * 1000 : 120000;
  const r = await client.sendCommand("run_command", params, clientTimeout);
  const structured = { exit_code: r.return_code ?? 0, stdout: r.stdout ?? "", stderr: r.stderr ?? "" };
  const parts = [`exit_code: ${structured.exit_code}`, `stdout:\n${structured.stdout}`];
  if (structured.stderr) parts.push(`stderr:\n${structured.stderr}`);
  return { content: [{ type: "text", text: parts.join("\n") }], structuredContent: structured };
});
```

`_MCPExecInterface.run_command` reads `res.structuredContent` (raising if `None` —
i.e. an old server build) and returns a `_CmdResult(stdout, stderr, returncode)`
duck-typed object — **no parsing, no ambiguity**. `ExecTool` is untouched
(`tools_shell.py:245-247` reads those attrs) and keeps its truncation/timeout/cwd
shaping. Bonus: the vm MCP accepts a server-side `timeout`; forwarding `ExecTool`'s
timeout (a small follow-up after the parity run) upgrades today's client-side-only
timeout to an actual VM-side process kill.

### 5. `build_tools` wiring — `harness/tools/tools.py:242-261`

Add an optional `mcp_runtime` param. When present:
- line 243: register `MCPBackend(mcp_runtime, workspace_root=...)` instead of
  `VMBackend(session.interface, ...)`.
- line 261: `ExecTool(_MCPExecInterface(mcp_runtime), workspace_root=...)`.

When absent, behavior is exactly as today (DesktopSession). This keeps the change
**flag-gated and reversible**.

### 6. Deployer lifecycle — `ale_run/agents/ale_claw/deployer.py`

- Before `build_tools` (after `session.check_status()`, ~`deployer.py:166`):
  ensure both bridges on the host and construct the `MCPRuntime`.
- Wrap the drive loop so servers are torn down on exit. The loop is already in a
  `try/except` (`deployer.py:312-340`); make `MCPRuntime` an `async with` around
  build_tools + agent.run, or add a `finally` that calls `await runtime.aclose()`.

```python
async with MCPRuntime(servers) as mcp_runtime:      # spawns node children
    tools = build_tools(session, ..., mcp_runtime=mcp_runtime)
    agent = OpenClawComputerAgent(..., tools=tools)
    await _drive()
# node children terminated here
```

Endpoint: `sb.endpoint` (`deployer.py:161`) is the host→VM cua URL, the same one
`cua_bridge_url()` returns for local/docker (`executor.py:126-140`). So the
host-side bridges hit the identical endpoint the harness uses today — no extra
network hop (see perf analysis).

### 7. Config — `ale_run/agents/ale_claw/config.py`

```python
substrate_transport: str = "mcp"   # "mcp" (default, this plan) | "session" (legacy fallback)
```
Validate in `__post_init__` (allow {"mcp","session"}). **Default is `mcp`** —
the MCP path is the intended substrate; `session` is retained only as an escape
hatch for debugging / parity comparison and may be removed once Phase 1 is
validated in CI. GUI transport stays on `session` in Phase 1 regardless (Phase 2
moves it to the cua MCP).

## Lifecycle / placement summary

- **Where servers run:** host (the deployer process), `local` executor.
- **What they talk to:** `CUA_SERVER_URL = sb.endpoint` → the eval VM's cua-server
  (unchanged target).
- **Spawn:** once per episode, before `build_tools`.
- **Teardown:** `AsyncExitStack.aclose()` terminates the node children on loop
  exit / exception / cancellation.

## Risks & mitigations

1. **`run_command` is lossy as flattened text.** Resolved by adding
   `structuredContent` to the vm MCP `run_command` handler (§4) — the harness reads
   the structured fields, never parses prose. Free to do: no agent consumes the
   server yet.
2. **Screenshot/large payloads over stdio.** Not in Phase 1 (GUI stays on
   session). For `read_bytes` of large files: base64 over stdio, same data volume
   as today's whole-file read; decode cost is host-local.
3. **Session concurrency w/ subagents.** Guard each `ClientSession` with a lock.
4. **Node-process startup (~100-300ms once/episode).** New fixed cost for a host
   harness that previously had none; amortized over the episode.
5. **Host bridge install.** Must generalize `ensure_*_mcp_server` to a host dir
   (claude_code path assumes in-sandbox `mcp_server_dir`). See §1.
6. **`write_text` append parity.** vm MCP `write_text` overwrites only; append maps
   to `write_bytes(append=true)`. Covered in §3.

## Testing

- **Unit:** `_MCPExecInterface` maps `structuredContent` → `.stdout/.stderr/
  .returncode` (incl. empty stderr, stderr-like text *inside* stdout — must NOT be
  misattributed, non-zero exit, large output). `MCPBackend` read/write/append/mkdir
  against a live `vm_mcp_server --test`-style harness.
- **Integration:** run an existing ale_claw task end-to-end with
  `substrate_transport="mcp"`; diff the trajectory vs the `session` baseline —
  tool names/params/return shapes must be identical (granularity preserved).
- **Parity assertions:** `read` pagination + `next_offset`, `edit` mismatch hint,
  `exec` 200k truncation + `timed_out` all still fire.
- **Teardown:** assert no orphaned `node` children after a run (incl. on timeout
  cancellation).

## Out of scope (Phase 1)

- GUI via cua MCP (Phase 2; needs keypress-disambiguation ported into the bridge).
- Exposing raw vm primitives as additional model-facing tools (we keep composite
  tools only).
- Changing installed-agent wiring (claude_code etc. unchanged).
- The `host` filesystem backend (stays local Python I/O).
