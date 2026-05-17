# `cb.RemoteDesktopSession` — Agent VM API

Primary handle: `c = session.computer` (cua-computer SDK). Use `c.interface` for all I/O. The session wrapper carries connection management plus four helpers (§1); the rest of it is either bugged or task-side — see §13.

All methods below are `async` and reach `cua-computer-server` inside the VM over HTTP (PTY also over WebSocket). RPC failures raise; success is implicit.

VM Python:

- **Windows**: `python` = `py` = `3.12.6`, installed at `C:\Users\User\AppData\Local\Programs\Python\Python312\`.
- **Linux**: `python` (PATH-first, used by `run_command "python …"`, `python_exec`, `venv_*`) = `3.14.3` from `/opt/cua-server/.venv/bin/python`. The real system interpreter `/usr/bin/python3` is `3.10.12` — `pip_install --system` writes there, not to the venv (and fails for other reasons, see §12).

## 0. Setup

```python
import cua_bench as cb

session = cb.computers.remote.RemoteDesktopSession(
    api_url="http://HOST:PORT",                       # e.g. "http://34.94.179.145:5000"
    os_type="linux",                                  # or "windows" | "android"
    provider_type="computer",
    headless=True,
    ephemeral=False,                                  # VM is externally managed
)
await session.check_status()                          # first call initializes the SDK
c = session.computer                                  # use this for everything below
```

`session.interface` is the same object as `c.interface`.

## 1. Session helpers

No equivalent on `c`; use from `session`:

| Member | Signature | Purpose |
|---|---|---|
| `session.os_type` | `Literal["linux","windows","android"]` | Platform branching |
| `await session.check_status()` | `-> bool` | Single-RPC reachability probe |
| `await session.wait_until_ready(timeout=60, poll_interval=2.0)` | `-> bool` | Poll `check_status` until ready or timeout |
| `await session.get_snapshot()` | `-> Snapshot` | Active-window metadata (includes PID; interface window methods only return `wid`) |

`Snapshot` schema (`cua_bench.types`):

```python
Snapshot(windows=[
    WindowSnapshot(pid="7180", wid=None,
                   title="Untitled - Notepad",
                   x=200, y=150, width=640, height=480,
                   active=True, minimized=False,
                   window_type="process"),            # or "webview"
])
# windows == [] when nothing is focused
```

## 2. Shell

```python
r = await c.interface.run_command(cmd)
# CommandResult: r.stdout (str), r.stderr (str), r.returncode (int)
```

Constraints:

- Fresh shell per call; `cwd` does not carry over → use `"cd /path && cmd"`.
- No timeout kwarg → wrap as `timeout 60 cmd` (Linux) or `start /b cmd /c "..."` (Windows).
- No `~` expansion → resolve `$HOME` / `%USERPROFILE%` once and reuse.
- Multi-line script body works on Linux (bash); Windows runs only the first line.

## 3. Filesystem

```python
await c.interface.write_text(path, content)
text = await c.interface.read_text(path)
await c.interface.write_bytes(path, data)
data = await c.interface.read_bytes(path, offset=0, length=None)

await c.interface.file_exists(path) -> bool
await c.interface.directory_exists(path) -> bool
await c.interface.list_dir(path) -> list[str]         # basenames, not recursive
await c.interface.get_file_size(path) -> int

await c.interface.create_dir(path)                    # recursive (parents auto-created)
await c.interface.delete_file(path)
await c.interface.delete_dir(path)                    # NOT recursive
```

Constraints:

- Paths must be absolute (no `~` expansion).
- Text I/O is UTF-8.
- `delete_dir` raises on non-empty (`Errno 39` / `WinError 145`); use `run_command("rm -rf …")` / `run_command('rmdir /S /Q "…"')` for trees.

## 4. Screenshot

```python
png = await c.interface.screenshot()                  # PNG bytes
size = c.get_screenshot_size(png)                     # sync; reads PNG header
                                                      # {"width": int, "height": int}
```

## 5. Mouse / Keyboard

```python
await c.interface.left_click(x, y)
await c.interface.right_click(x, y)
await c.interface.double_click(x, y)
await c.interface.mouse_down(x, y, button="left")     # button ∈ {"left","middle","right"}
await c.interface.mouse_up(x, y, button="left")

await c.interface.move_cursor(x, y)
await c.interface.drag_to(to_x, to_y, button="left", duration=0.5)   # from current cursor
await c.interface.drag(path=[(x1,y1), (x2,y2), ...],                 # multi-point path
                       button="left", duration=0.5)
# NOTE: `drag` takes a path list, NOT (from_x, from_y, to_x, to_y).
#       For a simple A→B drag: move_cursor(from_x, from_y) then drag_to(to_x, to_y).

await c.interface.scroll(x, y, clicks)                # +clicks=up, -clicks=down
await c.interface.scroll_up(clicks=3)
await c.interface.scroll_down(clicks=3)

await c.interface.type_text("hello")
await c.interface.press_key("Enter")
await c.interface.hotkey("ctrl", "c")                 # *args, NOT a list
await c.interface.key_down("shift")
await c.interface.key_up("shift")

pos = await c.interface.get_cursor_position()         # {"x": int, "y": int}
sz  = await c.interface.get_screen_size()             # {"width": int, "height": int}
```

Coordinates are pixels, origin top-left, no clamping.

## 6. Clipboard

```python
await c.interface.set_clipboard(text)
text = await c.interface.copy_to_clipboard()
```

Setting `""` then reading raises `RuntimeError("Failed to get clipboard content")`. Use `" "` as a cleared sentinel.

Recipe to verify typed text reached an editor (`activate_window` alone does not guarantee focus on Windows — see §8):

```python
await c.interface.left_click(cx, cy)                  # force focus on the target
await c.interface.set_clipboard("<sentinel>")
await c.interface.type_text("MARKER")
await c.interface.hotkey("ctrl", "a")
await c.interface.hotkey("ctrl", "c")
assert "MARKER" in await c.interface.copy_to_clipboard()
```

## 7. Launch / Open

```python
pid = await c.interface.launch(app, args=None)        # OS PID, NOT a window id
                                                      # e.g. "gedit", "notepad"
await c.interface.open(target)                        # default handler for file/folder/URL
```

For the `wid` after launch: `await c.interface.get_application_windows(app_name)`.

## 8. Window Management

```python
wid = await c.interface.get_current_window_id()
wids = await c.interface.get_application_windows("gedit")

title = await c.interface.get_window_name(wid)        # alias: get_window_title
w, h  = await c.interface.get_window_size(wid)
x, y  = await c.interface.get_window_position(wid)
bounds = await c.interface.get_active_window_bounds() # {"x","y","width","height"}

await c.interface.set_window_size(wid, w, h)
await c.interface.set_window_position(wid, x, y)
await c.interface.maximize_window(wid)
await c.interface.minimize_window(wid)
await c.interface.activate_window(wid)
await c.interface.close_window(wid)
```

Platform notes:

- **Linux (GNOME)**: every mutation (`set_*`, `maximize_window`, `minimize_window`, `close_window`) raises `RuntimeError: Failed to …`. Reads work. Cause: server-side `wmctrl` / `xdotool` cannot drive GNOME client-side decorations.
- **Windows**: `close_window` raises `RuntimeError`. Kill via `run_command("taskkill /F /IM <name>.exe /T")`.
- **`activate_window`** does not reliably steal focus on Windows. Always `left_click(cx, cy)` on the target window's content before typing.

## 9. Desktop / Coordinate Transforms

```python
de = await c.interface.get_desktop_environment()      # "windows" / "gnome" / ... / "unknown"
```

Linux often returns `"unknown"` because `XDG_SESSION_DESKTOP` / `DESKTOP_SESSION` are not set in the cua-server shell env. Branch on `session.os_type` instead.

`c.interface.to_screen_coordinates(x, y)` and `to_screenshot_coordinates(x, y)` always raise `TypeError: Logger.debug() takes 2 positional arguments but 12 were given` (bug in `computer/interface/generic.py:682`). Compute scale from `get_screen_size()` + screenshot dimensions if needed.

## 10. Interactive PTY

For sudo prompts, REPLs, `ssh` — anything needing streamed stdin.

```python
out = []
handle = await c.pty.create(
    command="bash",                                   # or "cmd.exe" / "powershell.exe"
    cols=80, rows=24,
    on_data=lambda b: out.append(b),                  # raw bytes, includes ANSI escapes
)
await handle.send_stdin(b"echo hello\n")
await handle.resize(120, 40)
await handle.send_stdin(b"exit\n")
exit_code = await handle.wait()
# or: await handle.kill()
```

Windows: leave **≥ 2 seconds between consecutive `send_stdin` calls**. Shorter delays silently drop input.

## 11. Run Python in the VM

```python
def add(a, b): return a + b
result = await c.python_exec(add, 2, 3)               # → 5
                                                      # Linux: runs under Python 3.14.3
                                                      # Windows: runs under Python 3.12.6

@c.python_command(use_system_python=True)
def host_info():
    import platform
    return {"system": platform.system(), "node": platform.node()}
info = await host_info()                              # runs in VM; returns dict

pid = await c.python_exec_background(slow_fn)         # returns OS PID
```

Constraints:

- `fn` must live in a real `.py` file — `inspect.getsource(fn)` reads source. REPL / heredoc / `exec()` / lambdas raise `Exception: Cannot retrieve source code for function …`.
- Args and return value are JSON-roundtripped (base64-wrapped); only JSON-friendly types are safe.

## 12. UV Venv Management

Per-task isolated Python envs. Requires `uv` pre-installed in the VM (included in cua images).

```python
await c.venv_install("my-env", ["requests", "pandas"])   # creates ~/.venvs/my-env
await c.venv_cmd("my-env", "python -m my_pkg")
result = await c.venv_exec("my-env", some_fn, *args)     # python_exec but inside the venv
pid = await c.venv_exec_background("my-env", bg_fn, requirements=[])
```

Venv root: `~/.venvs/<name>` (Linux), `%USERPROFILE%\.venvs\<name>` (Windows). Source-code constraint from §11 applies to `venv_exec`. Clean up with `run_command("rm -rf ~/.venvs/<name>")`.

`c.pip_install([...])` is **unusable on Linux**: runs `uv pip install --system`, which on this VM targets `/usr/bin/python3` (3.10) and writes to `/usr/local/lib/python3.10/dist-packages/`. cua-server runs non-root, so it fails with `Permission denied`. Note this also means **`pip_install` and `python_exec` use different Pythons on Linux** (3.10 vs 3.14) — another reason to prefer `venv_install`. Works on Windows (user-local 3.12). For cross-platform installs, use `venv_install`.

## 13. Pitfalls / Do NOT use

### Broken in the session wrapper

| Method | Bug | Use instead |
|---|---|---|
| `session.run_command(cmd)` | Returned dict's `return_code` is always `0`, `success` is always `True`, `check=True` never raises. `remote.py:769` reads `result.return_code` but the SDK exposes `result.returncode`. | `c.interface.run_command(cmd).returncode` |
| `session.scroll(direction, amount)` | `TypeError: ScrollAction.__init__() got an unexpected keyword argument 'x'` on every call. `remote.py:947` passes `x`/`y` kwargs that `ScrollAction` does not accept. | `c.interface.scroll_down(n)` / `c.interface.scroll_up(n)` |

### Broken in the interface

- `c.interface.to_screen_coordinates(x, y)` / `to_screenshot_coordinates(x, y)` — `TypeError` from `generic.py:682`. No clean workaround.

### Task-side / bench_ui helpers, not for agents

- `session.launch_window`, `session.execute_javascript`, `session.get_element_rect`, `session.click_element`, `session.right_click_element` — pywebview prompt UI.
- `session.install_app`, `session.launch_app`, `session.apps.<name>` — task setup app registry. Install via `run_command + apt/dnf/curl` or `venv_install`.
- `session.launch_application(name)` — vague GUI-menu launcher; use `c.interface.launch(app, args)`.
- `session.serve_static(...)` — raises `NotImplementedError`.

### Misleading or no-op

- `session.vnc_url` — placeholder `"http://localhost:8006/?autoconnect=true"` in client-only mode. agenthle VMs do not expose noVNC. Pass `vnc_url=` to the ctor only if a real port is known.
- `session.close_all_windows()` — no-op.
- `session.start(headless=False)` — opens VNC in the host's browser, not the VM. Keep `headless=True`.

### Do not call in client-only mode (VM is externally managed)

`c.start()` · `c.stop()` · `c.restart()` · `c.disconnect()` · `c.update(cpu, memory)` · `c.wait_vm_ready()`. (`c.get_ip()` works but is redundant with `c.api_host`.)

## 14. Concurrency

A single `session` is not thread- or task-safe — concurrent RPCs on the same connection have undefined ordering. Use `asyncio.Lock` if sharing across coroutines.

## 15. Endpoint metadata

```python
c.api_host        # str, real VM IP
c.api_port        # int, cua-server port
c.os_type         # same as session.os_type
c.tracing         # ComputerTracing; is_tracing=False unless opted in
```
