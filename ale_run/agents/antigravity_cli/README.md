# Antigravity CLI Agent (`agy`)

Google's Antigravity CLI — the successor to Gemini CLI — run as a one-shot
process inside the sandbox:

```text
agy --print=<prompt> --model <name> --output-format stream-json --dangerously-skip-permissions
  → CUA MCP Server (stdio, from ~/.gemini/config/mcp_config.json)
  → sandbox desktop (screenshot / click / type) + filesystem
```

It is a generalist CLI **and** GUI agent: it gets the shell/files natively from
the sandbox OS, and the desktop (screenshot, click, type…) through the CUA MCP
bridge — so it can do real computer-use tasks, not just terminal ones.

> **This preset validates Google account auth.** `agy` supports both Google
> OAuth and, in recent releases, a Gemini API-key mode. ALE currently stages an
> OAuth token file so the same authenticated account can run headlessly in each
> sandbox. OpenRouter cannot be selected by changing the base URL alone: `agy`
> speaks Gemini's native `streamGenerateContent` protocol while OpenRouter's
> public endpoint is OpenAI-compatible; a translation proxy would be required.

---

## Quick start

Everything below is a **one-time setup on your own machine**. After it, runs are
fully headless.

### 1. Install the pinned `agy` on your host

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
# installs to ~/.local/bin/agy
~/.local/bin/agy --version   # expected: 1.1.25
```

### 2. Log in with your Google account

Run `agy` in an interactive terminal on the host and follow the prompt:

```bash
~/.local/bin/agy
```

- It prints a **Google sign-in URL**.
- Open it in a browser (on **any** machine — this also works on a headless
  server: just copy the URL to your laptop's browser).
- Approve with a **Google account that has Antigravity access**.
- The `antigravity.google/oauth-callback` page shows an **authorization code** —
  paste it back into the terminal.

That writes your credential to:

```text
~/.gemini/antigravity-cli/antigravity-oauth-token
```

> Tip: use the plain `agy` command to log in (it waits for you). Avoid the
> `agy -p "…"` one-shot form for login — it only gives a ~30-second window to
> paste the code, too short for a browser round-trip.

### 3. Verify the login works headlessly

```bash
~/.local/bin/agy -p "Reply with exactly: PONG"
# → PONG     (no browser, reusing the saved token)
```

If you see `PONG`, the credential is good. It carries a refresh token, so it
keeps working across runs without logging in again.

### 4. Point ALE at the credential

Add the path to `secret/.env` (or export it in your shell):

```bash
export ANTIGRAVITY_OAUTH_TOKEN_PATH=$HOME/.gemini/antigravity-cli/antigravity-oauth-token
```

ALE forwards that file into the sandbox each run and `agy` silent-auths there —
you never log in inside the sandbox.

### 5. Run it

Pick a task list and reference the agent preset from an experiment:

```bash
echo "demo/seecheck" > selected_tasks/my_tasks.txt   # one task id per line
```

```yaml
# my_exp.yaml
secret_file: secret/.env
agents:      [configs/agents/antigravity_cli.yaml]
environment: configs/environments/docker.yaml      # or your GCE env
tasks:       selected_tasks/my_tasks.txt
```

```bash
uv run python -m ale_run run my_exp.yaml
```

---

## Choosing a model

`--model` takes the display names from `agy models` verbatim. Set it in the
preset's `model:` field:

```
Gemini 3.1 Pro (High)        Gemini 3.7 Flash (High)
Claude Sonnet 4.6 (Thinking) Claude Opus 4.6 (Thinking)
GPT-OSS 120B (Medium)
```

Which models you can use — and how much — depends on your Google plan. **Quota
is per account AND per model**, so if one model is throttled another may still
work (e.g. Gemini exhausted but Claude Sonnet 4.6 fine).

## Config

```yaml
# configs/agents/antigravity_cli.yaml
harness: antigravity_cli
model: Claude Sonnet 4.6 (Thinking)
config:
  dangerously_skip_permissions: true   # required headless
  cli_version: "1.1.25"                # pinned official release
  print_timeout: "120m"                 # ALE wall time remains authoritative
  download_url: ""                      # optional custom archive/binary
  download_sha512: ""                   # required with a custom URL
```

## How auth gets into the sandbox

1. **Host (once):** you log in → `~/.gemini/antigravity-cli/antigravity-oauth-token`.
2. **Per run:** ALE reads that file (via `ANTIGRAVITY_OAUTH_TOKEN_PATH`) and
   forwards its content into the sandbox; the deployer writes it back into place
   and `chmod 600`s it.
3. **In sandbox:** `agy` finds the token and silent-auths — no browser, no
   re-login. The token self-renews via its refresh token.

Treat the token file as a secret — it's a long-lived credential for your Google
account.

## Troubleshooting

- **Which account am I logged in as?** `cat ~/.gemini/google_accounts.json`
  (the `active` field). There is no agy CLI quota command — the web
  **Settings → Models** tab (signed in as that account) shows your plan + usage.
- **Runs 429 with `Individual quota reached … please upgrade your
  subscription`** → you're on a **free-tier** account (the "upgrade" wording is
  the tell), or that model's quota is spent. Either switch `model:` to one with
  quota left (e.g. `Claude Sonnet 4.6 (Thinking)`), or re-login with the right
  account: delete `~/.gemini/antigravity-cli/antigravity-oauth-token` and redo
  step 2.
- **Windows GUI tasks**: the CUA tools load reliably (the deployer warms up the
  node bridge + primes agy's first-run migration so they win agy's MCP
  tool-discovery race). No action needed.

## Notes

- The CUA GUI tools are declared in `agy`'s **native** `~/.gemini/config/mcp_config.json`
  (not gemini-cli's `settings.json`).
- `agy --output-format stream-json` writes logical model-generation usage and
  tool durations to `transcript.jsonl`. ALE converts it to ATIF and writes one
  `metrics_summary.json` with an `api_calls` list of per-generation
  tokens/latency, tool calls and latency, multimodal output groups, and physical
  transport request count.
- Each task passes `--log-file=<work_dir>/agy_cli.log`, so agy's complete native
  log is pulled with that task even under high concurrency. Native `agy` does
  not expose cache-write tokens, request cost, or per-transport-retry
  tokens/latency; the summary marks these as unavailable instead of zero.
- CLI policy lives in
  `~/.gemini/antigravity-cli/settings.json`; the CUA MCP declaration lives in
  the separate `~/.gemini/config/mcp_config.json`.

## Smoke test

Run the experiment from step 5 against **`demo/seecheck`** (a vision smoke test:
read a code off the desktop) — the quickest end-to-end check that auth + the GUI
bridge both work:

```bash
uv run python -m ale_run run my_exp.yaml
```

On Windows use `demo/seecheck_win` instead.
