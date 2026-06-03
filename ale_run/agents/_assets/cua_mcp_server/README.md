# cua_mcp_server

MCP server (stdio) that exposes the CUA computer-server's **GUI action**
surface — mouse, keyboard, and screenshot — as MCP tools. It is the desktop
control bridge injected alongside agents; the more general non-GUI surface
(bash, filesystem, PTY) lives in the sibling [`vm_mcp_server`](../vm_mcp_server).

- **Server name / version:** `cua-desktop` 0.3.0
- **Transport:** stdio (one process per agent run)
- **Backend:** HTTP `POST /cmd` to the CUA computer-server, SSE-framed reply
- **Endpoint:** read from `CUA_SERVER_URL` (the executor injects this via
  `cua_bridge_env`), default `http://localhost:5000`

## Coordinate system

All coordinates are **normalized to `[0, 1000]`** on both axes, independent of
the VM's real resolution. The bridge queries the screen size once (cached) and
converts to absolute pixels before calling the backend. `[0,0]` is top-left,
`[1000,1000]` bottom-right.

## Return values

Every tool returns an MCP result with a single **text** content block (a short
human-readable confirmation), except `screenshot`, which additionally returns an
**image** block (base64 PNG). Errors are returned as a text block with
`isError: true`.

## Tools

### Keyboard

| Tool | Parameters | Returns |
|------|------------|---------|
| `key` | `keys: string[]` — keys to press together (e.g. `["ctrl","c"]`) | text: `Pressed: <keys>`. Single key → press+release; multiple → hotkey chord. |
| `key_down` | `keys: string[]` | text: `Key down: <keys>`. Presses without releasing (pair with `key_up`). |
| `key_up` | `keys: string[]` | text: `Key up: <keys>`. Releases keys held by `key_down`. |
| `type` | `text: string` | text: `Typed: "<preview>"`. Types into the focused field. |
| `hold_key` | `keys: string[]`, `duration: number` (seconds) | text: `Held <keys> for <d>s`. Holds then releases. |

Key names are normalized (case-insensitive; `Control`/`ctrl`, `ArrowUp`/`up`,
`Return`/`enter`, `meta`/`cmd`, etc.). Valid: `ctrl, shift, alt, cmd/meta,
enter, esc, tab, space, backspace, delete, up, down, left, right, home, end,
page_up, page_down, f1-f12`, or any single character.

### Mouse

| Tool | Parameters | Returns |
|------|------------|---------|
| `mouse_move` | `coordinate: [x,y]` (required) | text: `Moved cursor to [x, y]`. |
| `click` | `coordinate?: [x,y]`, `button?: "left"\|"right"\|"middle"` (default `left`), `clicks?: 1\|2\|3` (default `1`) | text: `Clicked (<button>, <n>x) at [x,y]`. Omit `coordinate` to click at the current position. |
| `drag` | `coordinate: [x,y]` (end, required), `start_coordinate?: [x,y]`, `button?: "left"\|"right"\|"middle"` (default `left`) | text: `Dragged (<button>) from <start> to [x,y]`. Uses mouse_down → move → mouse_up. Omit `start_coordinate` to start at the current position. |
| `mouse_down` | `button?: "left"\|"right"\|"middle"` (default `left`) | text: `Mouse down: <button>`. |
| `mouse_up` | `button?: "left"\|"right"\|"middle"` (default `left`) | text: `Mouse up: <button>`. |
| `scroll` | `direction: "up"\|"down"\|"left"\|"right"`, `amount: number`, `coordinate?: [x,y]` | text: `Scrolled <dir> <amount>`. If `coordinate` given, moves there first. |

### Screen / utility

| Tool | Parameters | Returns |
|------|------------|---------|
| `screenshot` | `save_path?: string` (absolute path on the VM) | text confirmation **+ image** block (base64 PNG). If `save_path` given, the parent dir is validated and the PNG is written there too. |
| `cursor_position` | — | text: `Cursor at [x, y]` (normalized). |
| `wait` | `duration: number` (seconds) | text: `Waited <d>s`. Client-side pause. |

## Local testing

```bash
npm install
CUA_SERVER_URL=http://<host>:5000 node src/index.js --test   # smoke test
CUA_SERVER_URL=http://<host>:5000 node src/index.js           # start stdio server
```

`node_modules` is not vendored — it is rebuilt on the substrate by
`ale_run/agents/_bootstrap.py::ensure_cua_mcp_server` (`npm install --production`).
