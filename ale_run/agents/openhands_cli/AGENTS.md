# OpenHands CLI -- Implementation Notes

Companion to `README.md`. README explains *what* this integration is; this
file explains *how* it is built, installed, and tested.

## 1. Source / install strategy

- Upstream: `All-Hands-AI/OpenHands-CLI` (V1 CLI).
- Package: `openhands-cli` on PyPI (official pip package, no fork needed).
- Pinned version: `1.15.1` (configurable via `cli_version` in config).
- Install: `uv pip install openhands-cli==<version>` with pip fallback.

## 2. CLI invocation

```bash
openhands \
    --headless \
    --json \
    --yolo \
    --override-with-envs \
    --exit-without-confirmation \
    -t "<prompt>"
```

Argument decisions:

| Flag | Why |
|---|---|
| `--headless` | Forces `exit_without_confirmation = True` and disables the LLM critic. Required when invoking via `-t/-f`. |
| `--json` | Routes events through `json_callback` (delimiter-framed `Event.model_dump()`). Only valid with `--headless`. |
| `--yolo` (`--always-approve`) | Auto-approves every confirmation policy. |
| `--override-with-envs` | Without this the CLI ignores `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` and falls back to persisted settings. |
| `--exit-without-confirmation` | Defensive duplicate of the headless implication. |
| `-t <task>` | Headless requires `-t` or `-f`. |

The deployer writes `~/.openhands/mcp.json` directly -- it never invokes
`openhands mcp add ...`.

## 3. Persistence layout

```
~/.openhands/
+-- .env                         # written by install() (LLM_*, OPENHANDS_*)
+-- mcp.json                     # written by install() (CUA bridge)
+-- conversations/
    +-- <hex_id>/                # SDK-managed
        +-- base_state.json
        +-- ...
```

Per-run scratch lives in the executor work_dir:
`prompt.txt`, `stdout.log`, `stderr.log`, `transcript.jsonl`, `openhands.pid`.

## 4. JSON-Event stream parsing

`utils.json_callback`:

```python
print("--JSON Event--")
print(json.dumps(event.model_dump(), indent=2, sort_keys=True))
```

This is **not** strict JSONL -- each block spans many lines. The deployer
splits the stdout buffer on lines that exactly equal `--JSON Event--`,
uses `JSONDecoder.raw_decode` to consume the leading JSON object from
each chunk, and silently drops trailing non-JSON text.

Persisted to work_dir:

| File | Content |
|---|---|
| `stdout.log` | Raw stdout (delimiter-framed) -- kept for debugging |
| `stderr.log` | Raw stderr |
| `transcript.jsonl` | One event-dict per line; produced by parse_artifacts |

## 5. Tool surface

| Tool | Source | Notes |
|---|---|---|
| `terminal_tool` | OpenHands V1 default | host shell |
| `file_editor_tool` | OpenHands V1 default | str_replace / view / create |
| `task_tracker_tool` | OpenHands V1 default | in-conversation TODO |
| `task_tool` | OpenHands V1 default | sub-task delegation |
| `cua.*` | CUA MCP Server bridge | screenshot, click, type, key, scroll, browser, shell, files |

OpenHands V1 CLI ships without browser, IPython, web search, or finish/think
tools, so no explicit disable list is required.

## 6. Provider routing

Two paths, both flowing through LiteLLM inside the SDK:

### OpenRouter

```ini
LLM_API_KEY=<OPENROUTER_API_KEY>
LLM_MODEL=openrouter/<vendor>/<model>
LLM_BASE_URL=https://openrouter.ai/api/v1
```

### Direct (Anthropic)

```ini
LLM_API_KEY=<ANTHROPIC_API_KEY>
LLM_MODEL=anthropic/<model>
# LLM_BASE_URL unset -> LiteLLM's default Anthropic endpoint
```

## 7. Installation

The deployer's `install()` method handles installation automatically:

```bash
uv pip install openhands-cli==1.15.1
# or fallback:
pip install openhands-cli==1.15.1
```

The CUA MCP Server is expected to be baked into the sandbox image.

## 8. Known pitfalls

- **`--override-with-envs` is mandatory.** Forgetting it makes the CLI
  silently fall back to its persisted settings or raise
  `MissingEnvironmentVariablesError`.
- **dotenv autoload from `$CWD`.** The CLI loads `.env` from the current
  directory if present. The deployer sets cwd to the work_dir (no `.env`
  there) to avoid interference.
- **Multi-line JSON blocks.** The `--JSON Event--` framing requires a
  delimiter-aware parser -- do not use naive `json.loads(line)`.
- **Linux only.** OpenHands CLI is Linux/macOS only.
