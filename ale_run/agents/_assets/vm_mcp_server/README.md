# vm_mcp_server

MCP server (stdio) that exposes the **non-GUI** surface of the CUA
computer-server as a small set of irreducible VM **primitives**. It is a thin
substrate, not a toolkit: it surfaces native server capabilities and nothing
more. Agents build their own higher-level tools (edit, glob, grep, etc.) on top
of these — this server intentionally makes no tool-design decisions for them.
The GUI action set (mouse, keyboard, screenshot) is the complementary
[`cua_mcp_server`](../cua_mcp_server) bridge.

- **Server name / version:** `vm-primitives` 0.1.0
- **Transport:** stdio (one process per agent run)
- **Backend:** the CUA computer-server — `POST /cmd` (SSE-framed reply) for
  exec/filesystem, and the `/pty*` REST + SSE-stream endpoints for PTY
- **Endpoint:** read from `CUA_SERVER_URL` (the executor injects this), default
  `http://localhost:5000`

## Design: irreducible primitives

Every tool maps to one native computer-server capability. The set is restricted
to capabilities that are **not substitutable by one another**:

| Axis | Tools | Why it's a primitive |
|------|-------|----------------------|
| Process execution | `run_command` | One-shot exec. |
| Interactive execution | `pty_*` | Streaming, stateful, interactive — cannot be expressed by one-shot `run_command`. |
| Filesystem I/O | `read_text` `write_text` `read_bytes` `write_bytes` | Clean, binary-safe file access that does not go through shell parsing/quoting. |
| Clipboard | `read_clipboard` `write_clipboard` | The desktop-session clipboard; not reachable via `run_command`. |

Deliberately **excluded**: convenience file ops that are trivially reducible to
`run_command` (`ls`, `test -f`, `stat`, `mkdir`, `rm`, `rmdir`), and any
opinionated/composite tools (`edit`, `glob`, `grep`, a `bash` alias). Those are
the agent's to compose.

## Return values

Every tool returns an MCP result with a single **text** content block; the
sections below describe its contents. Backend failures surface as a thrown MCP
error; a bad `pid` returns a text block with `isError: true`.

## Process execution

### `run_command`
Run a command and wait for it to finish. One-shot and blocking — **not**
streaming or interactive, and **no** shell state persists between calls (use
`cd /path && …` for cwd, `VAR=val cmd` for env). For long-running or interactive
work use the PTY tools.

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | yes | Command run via the VM's system shell. |
| `timeout` | number | no | Server-side timeout in **seconds**; the command is killed if it runs longer. |

**Returns** text:
```
exit_code: <int>
stdout:
<stdout>
stderr:        # only present when stderr is non-empty
<stderr>
```

## Filesystem I/O

Paths are absolute on the VM; a leading `~` is expanded to the user's home.

| Tool | Parameters | Returns (text) |
|------|------------|----------------|
| `read_text` | `path: string` | Full file contents (UTF-8). Use `read_bytes` for binary. |
| `write_text` | `path: string`, `content: string` | `Wrote <n> bytes to <path>`. Overwrites; creates if absent. |
| `read_bytes` | `path: string`, `offset?: int` (default 0), `length?: int` (default: to EOF) | Base64 of the requested byte range. |
| `write_bytes` | `path: string`, `content_b64: string`, `append?: bool` (default false) | `Wrote bytes to <path>` (`(appended)` if append). |

## Clipboard

The VM's system clipboard, a capability not reachable through `run_command`.
Tool names are clarified because the underlying native command names are
misleading (`copy_to_clipboard` *reads*, `set_clipboard` *writes*).

| Tool | Parameters | Returns (text) |
|------|------------|----------------|
| `read_clipboard` | — | Current clipboard text (maps to native `copy_to_clipboard`). |
| `write_clipboard` | `text: string` | `Set clipboard (<n> bytes)` (maps to native `set_clipboard`). |

## Interactive execution (PTY)

A PTY is a persistent pseudo-terminal. Unlike `run_command`, it survives across
calls and streams output incrementally; you drive it with input and poll for
output.

**Buffering model:** the computer-server only delivers PTY output to subscribers
connected at the moment it is produced — there is no replay for late
subscribers. So this bridge opens a persistent output subscription at
`pty_start` and accumulates output in a per-session buffer (cap 2 MB; on
overflow the oldest bytes are dropped and the next read is flagged). `pty_read`
drains that buffer. Note: output produced in the brief window before the
subscription attaches (e.g. the very first shell prompt) may not be captured.

| Tool | Parameters | Returns (text) |
|------|------------|----------------|
| `pty_start` | `command?: string` (default interactive `bash`), `cwd?: string`, `env?: {string: string}`, `cols?: int` (default 80), `rows?: int` (default 24) | `pid: <n>` / `cols: <n>` / `rows: <n>`. Keep the `pid`. |
| `pty_input` | `pid: int`, `data: string` | `Sent <n> bytes to pid <pid>`. Raw stdin write — include a trailing `\n` in `data` to submit a line. Control chars allowed, e.g. `\u0003` = Ctrl-C. |
| `pty_read` | `pid: int`, `timeout_ms?: int` (default 2000; 0 = return immediately) | Buffered output since the last read (raw terminal bytes, may contain ANSI escapes), then a status line `[running]` or `[exited, code <n>]`. Waits up to `timeout_ms` if nothing is buffered yet. A leading `[earlier output dropped — buffer overflow]` appears if the cap was hit. |
| `pty_resize` | `pid: int`, `cols: int`, `rows: int` | `Resized pid <pid> to <cols>x<rows>`. |
| `pty_kill` | `pid: int` | `Killed pid <pid>`. Terminates the session and releases its buffer/subscription. |

### Typical PTY flow
```
pty_start { command: "python3 -i" }                  → pid 1234
pty_input { pid: 1234, data: "print(2+2)\n" }
pty_read  { pid: 1234 }                               → "...\n4\n>>> \n[running]"
pty_kill  { pid: 1234 }
```

## Local testing

```bash
npm install
CUA_SERVER_URL=http://<host>:5000 node src/index.js --test   # exec/file/PTY smoke test
CUA_SERVER_URL=http://<host>:5000 node src/index.js           # start stdio server
```

`node_modules` is not vendored — it is rebuilt on the substrate at deploy time
(`npm install --production`), mirroring `cua_mcp_server`.
