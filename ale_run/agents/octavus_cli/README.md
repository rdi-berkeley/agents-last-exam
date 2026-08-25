# Octavus Agent CLI

An Octavus agent driven by the public `octoagent` CLI (`@octavus/agent`),
deployed in `sandbox` environments.

The agent under test is a cloud Octavus agent. The CLI runs on the ALE eval box
and drives that machine's own desktop, filesystem, and shell; the brain (model,
prompt, tools, workers, skills, memory) runs in the Octavus cloud. Only the
computer is local, so the single credential the harness needs is the agent key.

Official CLI docs: https://octavus.ai/docs/workforce-agents/cli

## Architecture

```
FRAMEWORK                                 SANDBOX (ALE eval box)
lifecycle.py                              cua-computer-server (:0 desktop)
  install()   -> npm install -g @octavus/agent, install Chrome for Testing
  launch()    -> octoagent run --json --workdir <task variant dir> "<prompt>"
                   |                      (agent brain runs in the Octavus cloud;
                   |                       the CLI drives the local computer)
  parse_artifacts() <- octoagent.result.json  (threadId / threadUrl / status)
                    <- thread read API         (transcript, usage/cost, model)
```

`launch()` spawns the CLI detached and polls it under the orchestration wall
budget, so a run that exceeds the budget is reaped. `parse_artifacts()` runs
host-side: it reads the single `--json` result off stdout, then reads the
observable thread with the same agent key to attach the transcript, per-run cost,
tokens, and effective model to the trajectory. Pure stdlib (subprocess / pathlib
/ json / urllib) plus the shared node bootstrap; no Octavus-internal or admin
surface.

## Auth

Set `OCTAVUS_AGENT_API_KEY` in `secret/.env` to the agent's `oct_agt_*` key,
minted from the agent's Settings -> API tab. The agent's Computer must be set to
"Your own machine (CLI)" in the dashboard. The key is named `api_key` in the
config, so the framework carries it as a secret (into the sandbox via a read-once
sidecar; never written to gathered host logs) and passes it to the run via the
`OCTAVUS_API_KEY` env var, never argv.

## Per-run overrides

The stored agent already defines the model, prompt, tools, workers, skills, and
memory. These `config:` keys optionally override, per run, without touching the
stored agent (unset inherits the dashboard default):

| Key | CLI flag | Notes |
|---|---|---|
| `model` (or top-level `model:`) | `--model` | Primary model, `provider/model-id`. |
| `backup_model` | `--backup-model` | Backup model, `provider/model-id`. |
| `capabilities` | `--capability slug=on\|off` | Per-capability toggle. A toggle only applies to a capability the agent's protocol declares; toggling an undeclared one is rejected (HTTP 400). |
| `record` / `record_visibility` | `--record` / `--record-public` | Record the execution view to a shareable video. Gated to funded tiers; `public` yields a permanent URL. |

A model override must resolve to a provider the org has a key for (project/org
BYOK or platform default), otherwise the run is rejected at session creation.

## Run it

Reference the preset from an experiment (see `example_exp.yaml` for the shape):

```yaml
agents:
  - configs/agents/octavus_cli.yaml
environment: configs/environments/environment_gcloud.yaml
tasks: selected_tasks/ale_cli.txt      # 105 cpu-free-ubuntu (Linux) tasks
```

```bash
uv run python -m ale_run run <experiment>.yaml --dry-run   # preview units
uv run python -m ale_run run <experiment>.yaml             # run
```

## Browser tasks

Branded Chrome 137+ cannot load the automation extension (`--load-extension` was
removed), the Chromium snap's confinement blocks it too, and the ALE images bake
branded Google Chrome (which the CLI refuses). `install()` therefore installs
Chrome for Testing via `@puppeteer/browsers` to `~/.octavus/browsers` - the exact
location the CLI auto-detects (matching the public installer) - and also points
the CLI at the binary with `--chrome-path`, so the browser tools come up during
`computer-ensure-ready`. Set `chrome_path` to pin your own Chrome for Testing or
Chromium instead. The display stack (Xvfb, AT-SPI + its python bindings, xdotool,
scrot, Chrome libs) is best-effort `apt-get`ed at setup unless the box already has
both the screenshot/automation tools and the AT-SPI2 python bindings that
`computer-use__label` needs (checking only the former misses images that bake a
desktop but omit `python3-gi`); set `install_prereqs: false` on an image that
already bakes the full stack.

`computer-use__label` runs an AT-SPI driver as `python3`, which must import the
distro `python3-gi` bindings. Those live only in the system interpreter, so
`launch()` pins the CLI's `PATH` to `/usr/bin` first: the ALE image otherwise
prepends a gi-less uv venv (`/opt/cua-server/.venv`, Python 3.14) that a bare
`python3` would resolve to, and labeling would fail with "AT-SPI2 not available"
even though the bindings are installed for `/usr/bin/python3`.
