# ALE on-disk log spec

This document is the **single source of truth** for what `ale` writes to
disk after running one task × one agent. It is intentionally
self-contained — you should be able to read this file end-to-end and
understand every byte under a run dir, what code path produced it,
which rules govern it, and what each consumer is allowed to assume.

`docs/DESIGN.md §"On-disk run output"` is the one-screen summary; this
file is the longform.

A complete sample run (a Claude Code × demo task, captured on
2026-05-18) lives next to this file under `sample_run/`. Every file
referenced below has a real example you can open. The zip in this
folder bundles the doc + the sample for offline sharing.

> **Heads up — secret scrubbing.** The captured `sample_run/` had
> one real OpenRouter key in `origin_log/claude-code/run_claude.sh`
> (the inlined `ANTHROPIC_AUTH_TOKEN=sk-or-v1-…`). That value has been
> replaced with `***REDACTED-FOR-DOC-ILLUSTRATION***`. No other file
> contained credentials. This is a property of the live framework
> (the deployer inlines the key into the bash launcher so the VM-side
> process gets a deterministic env) — when sharing real run dirs
> externally, scrub that file or the entire `origin_log/claude-code/`
> subtree.

---

## 1. Identity & path layout

Every run lives in **exactly one directory** keyed by five slugs +
a UTC timestamp:

```
<output_root>/<agent_id>/<model_slug>/<task_slug>/v<variant_index>/<YYYYMMDD_HHMMSS>/
```

`output_root` is operator-chosen (defaults to `<exp_root>/<exp_name>/`
when launched via `python -m ale run <yaml>`). The five segments below
it are **slugified** through these rules — they're the same rules the
writer uses to detect collisions, so external tooling that reproduces
paths must use the identical normalization.

| Segment | Slug rule |
|---|---|
| `<agent_id>` | yaml `id:` field, lowercased, `-` → `_`, non-`[a-z0-9_]` → `_`, stripped. Empty → `"unknown"`. |
| `<model_slug>` | model id, lowercased, `.` `/` `_` → `-`, non-`[a-z0-9-]` → `-`, stripped. Empty → `"unknown-model"`. |
| `<task_slug>` | task path with `/` → `__`, leading/trailing `/` stripped (matches agenthle convention). |
| `v<variant_index>` | integer variant index, zero-based, prefixed with literal `"v"`. |
| `<YYYYMMDD_HHMMSS>` | UTC `strftime("%Y%m%d_%H%M%S")` taken **once at writer construction**. |

Reference Python (verbatim from the writer):

```python
import re
_SLUG_RE = re.compile(r"[^a-z0-9-]+")

def slug_model(model: str) -> str:
    s = model.lower().replace(".", "-").replace("/", "-").replace("_", "-")
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "unknown-model"

def slug_task(task_path: str) -> str:
    return task_path.strip("/").replace("/", "__")

def slug_agent(agent_name: str) -> str:
    s = (agent_name or "unknown").lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_") or "unknown"
```

### `run_id` (logical identifier)

The run dir also has a flat logical id used as a foreign key across
events / trajectory / GCS prefix / VM-side work_dir naming:

```
run_id = "{slug_agent(agent_id)}__{slug_model(model)}__"
         "{slug_task(task_path)}__v{variant_index}__{ts}"
```

For the sample run that's:

```
cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746
```

### Collision policy

The writer **refuses to overwrite** an existing run dir — if
`(agent, model, task, variant, ts)` collide at second-resolution the
constructor raises `FileExistsError`. This is the only place ALE
strictly guards against accidental clobbering. Resume / rerun
explicitly creates a new timestamped sibling.

---

## 2. Directory contents (top level)

After a successful run the directory looks exactly like this:

```
<run_dir>/
├── run.json              one-shot summary, written once at the very end
├── trajectory.json       ATIF-v1.0 episode, written once at the very end
├── eval_result.json      eval-only outcome, written once at the very end
├── events.jsonl          append-only phase trace, fsync after each line
├── origin_log/           deployer work_dir (agent-side artifacts)
│   └── <agent_name>/     one subdir per deployer instance
│       └── …             agent-specific files (see §6)
└── output/               task `remote_output_dir` mirrored from the VM
    └── …                 whatever the task / agent produced for eval
```

`origin_log/` and `output/` are **always created as empty dirs** by the
writer at construction time, even if the run later fails to populate
them. So a run dir always has the same six top-level entries.

Files only appear if their corresponding phase ran. The "graceful
degradation" guarantee: the writer wraps each top-level write in a
`try/except` and only emits a `logger.warning` on failure — a partial
run dir is **always valid**, never half-written. Consumers must treat
missing files as legitimate (failed-very-early run) rather than
corrupted.

---

## 3. `events.jsonl` — the phase trace

The single ground truth for "what happened, when". Append-only, one
JSON object per line, fsync'd after every write so SIGTERM / kill -9
never loses an event.

### Format

```json
{"ts": "<RFC3339 UTC>", "type": "<event_type>", "run_id": "<run_id>",
 "data": {<event-specific fields>}}
```

- `ts` uses `strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())` — second resolution, `Z` suffix.
- `type` is one of the strings in the table below.
- `run_id` is repeated on every line so concatenated jsonl from multiple
  runs can be re-keyed.
- `data` is omitted when the event has no payload.

### Event types (canonical set)

Emitted by `ale.runner.lifecycle.run_one_unit` in roughly this order:

| `type` | When | Notable fields |
|---|---|---|
| `run_started` | First line — written immediately after the writer opens. | `agent`, `agent_class`, `model`, `task`, `variant_index`, `runtime` |
| `provision_wait` | Before acquiring `provision_sem`. Lets you see queue depth. | (none) |
| `provision_started` | Semaphore acquired; `env.reset_async` is about to run. | (none) |
| `provision_done` | `env.reset_async` returned successfully. | `vm_id` |
| `run_wait` | Before acquiring `run_sem`. | (none) |
| `run_started` (second occurrence) | Run-sem acquired; agent install/launch is next. | (none) |
| `incremental_pull_started` | Only for `runtime: vm` deployers that declared `hot_artifacts`. Background tail loop kicked off. | `targets[]` (VM paths), `interval_s` |
| `agent_run_started` | About to call `executor.run_deployer`. | `runtime`, `work_dir` |
| `agent_finished` | Deployer returned an `AgentRunResult`. | `status`, `error` |
| `incremental_pull_final_failed` | Final reconcile after agent stop threw. | `error` |
| `post_launch_fanout_started` | Three concurrent pipelines (origin/output/eval) about to be gathered. | `gcs_bucket` (or `"(cua direct)"`) |
| `origin_log_gather_done` / `_failed` | Mirror of deployer work_dir finished. | `report = {"transport", "files", "error"}` |
| `output_gather_done` / `_failed` / `_skipped` | Mirror of task `remote_output_dir` finished. | `vm_path`, `report` (or `reason` for skipped) |
| `run_cancelled` | KeyboardInterrupt / asyncio.CancelledError caught. | `reason`, `phase` |
| `run_failed` | Any other exception caught. | `error_type`, `message`, `phase`, `category` |
| `best_effort_gather_started` / `_done` / `_failed` / `_timeout` | Triggered only on cancel / fail paths to scoop whatever the agent left on the VM (vm runtime only, 60s timeout). | (none on start/done, `error`/`timeout_s` otherwise) |
| `run_completed` | Always last line. | `status`, `score`, `total_duration_s` |

`run_started`, `agent_run_started`, etc. are **soft markers** — they can
be missing if the lifecycle errored before reaching that line. Always
parse with `if entry.get("type") == "X"`, never assume order beyond
"`run_started` is first, `run_completed` is last (when present)".

### Reading

For a quick glance:

```bash
jq -r '.ts + "  " + .type' events.jsonl
```

For the sample run that yields:

```
2026-05-18T22:17:46Z  run_started
2026-05-18T22:17:46Z  provision_wait
2026-05-18T22:17:46Z  provision_started
2026-05-18T22:20:58Z  provision_done
…
2026-05-18T22:24:52Z  run_completed
```

VM acquire took ~3m12s (`provision_wait` → `provision_done`), agent
itself ran ~3m20s, post-launch fanout + writeback ~34s, total 7m06s.

---

## 4. `run.json` — one-shot summary

Written **once** at the very end by `RunWriter.write_run_json`. It's a
single dict whose shape is canonical:

```jsonc
{
  "schema_version": 2,
  "run_id": "<run_id>",
  "timestamp_utc": "<RFC3339, evaluated at WRITE time, not start>",

  "agent": {
    "id":            "<yaml id>",
    "class":         "<yaml class: e.g. claude_code | ale_claw>",
    "name":          "<deployer config.name, e.g. claude-code>",
    "version":       "<deployer.version or null>",
    "model":         "<config.model>",
    "runtime":       "vm" | "local" | "docker",
    "config_repr":   { … yaml config dict with api-key-like fields REDACTED … }
  },

  "task": {
    "slug":          "<task_slug>",
    "path":          "tasks/<task_path>",
    "variant_index": <int>
  },

  "status": "completed" | "timeout" | "failed" | "cancelled" | "not_executed",
  "score":  <float|null>,                 // task.evaluate's reward[0]; null unless eval ran

  "termination": {
    "reason":   "<status, or 'completed' on success>",
    "phase":    "env_start" | "stage_inputs" | "task_setup" | "agent_run"
              | "stage_reference" | "evaluation" | "cleanup" | "unknown" | null,
    "category": "rate_limited" | "vm_quota_exhausted" | "auth_failed"
              | "gcs_missing" | "transport_error" | "rpc_timeout" | null,
    "error": null | {
      "type":       "Exception",
      "message":    "<str(exc)>",
      "traceback":  "<full traceback string>"
    }
  },

  "timings": { "duration_s": <float, 2 decimals> },

  "usage": null | {
    "total_steps":                   <int>,
    "total_input_tokens":            <int>,
    "total_output_tokens":           <int>,
    "total_cache_read_tokens":       <int>,
    "total_cache_creation_tokens":   <int>,
    "total_cost_usd":                <float>,
    "total_duration_ms":             <int>,
    "reward":                        <float|null>,
    "status":                        "completed" | "timeout" | "failed"
  }
}
```

### Field-by-field rules

- **`schema_version: 2`** — bumped when this shape changes incompatibly.
  Consumers must check.
- **`timestamp_utc`** is the moment `run.json` was written — distinct
  from `events.jsonl`'s `run_started.ts` (run dir creation). Use
  `events.jsonl` for true start/end timing.
- **`agent.config_repr`** is a shallow copy of the yaml `config:`
  block, with these keys redacted to `***<last 4 chars>` (or `***`
  when <4 chars):
  `anthropic_api_key`, `openrouter_api_key`, `openai_api_key`,
  `brave_api_key`, `api_key`. Lookup is case-insensitive. Any other
  secret-like field is **not** auto-redacted — schema is open.
- **`status`** semantics (lifecycle authoritative):
  - `"completed"` — agent loop exited cleanly (success or
    step-budget-hit), and eval ran (regardless of score).
  - `"timeout"` — agent wall budget exceeded.
  - `"failed"` — agent threw OR eval threw OR pre-agent setup threw.
  - `"cancelled"` — SIGTERM / SIGHUP / SIGINT mid-flight.
  - `"not_executed"` — initialization didn't reach the agent run
    (resolve_agent / env.make crashed).
- **`score`** is whatever `task.evaluate()` returned, element 0. `null`
  whenever eval didn't run.
- **`termination.phase`** is set ONLY when `status != "completed"`. The
  resolver prefers `env.current_phase` (more granular: `env_start`,
  `stage_inputs`, `task_setup`, `stage_reference`, `evaluation`,
  `cleanup`) over the lifecycle's own coarse tracker (`agent_run`,
  `unknown`).
- **`termination.category`** is the first match in this lowercased-substring
  table (None when no pattern matches):

  | category | substrings checked in `str(exc).lower()` |
  |---|---|
  | `rate_limited` | `rate limit`, `ratelimit`, `429`, `too many requests` |
  | `vm_quota_exhausted` | `quota`, `stockout`, `resource_exhausted`, `does not have enough resources`, `cpus_per_vm_family` |
  | `auth_failed` | `401`, `403`, `authentication_failed`, `permission denied`, `unauthorized`, `forbidden`, `llm auth failed`, `user not found`, `invalid api key` |
  | `gcs_missing` | `matched no objects`, `no urls matched`, `bucketnotfoundexception`, `no such object` |
  | `transport_error` | `connection reset`, `connection refused`, `503`, `service unavailable`, `deadline exceeded`, `broken pipe`, `remote end closed connection` |
  | `rpc_timeout` | `timeout`, `timed out` (also: `TimeoutError` / `asyncio.TimeoutError` matched by isinstance before string-match) |

  `KeyboardInterrupt` / `CancelledError` always yield `category=None`.

- **`termination.error.traceback`** is the same string as `.message`
  when the error originated from `AgentRunResult.error` (agent-level
  failures already preformat). It's the full `traceback.format_exc()`
  when the lifecycle itself caught an exception. **Don't try to parse
  this** — it's diagnostic text only.
- **`usage`** is taken from `trajectory.final_metrics`. `null` when the
  trajectory builder never finalized (e.g. very-early crash).
  `total_steps` counts ATIF Steps including the framework-injected
  `user` instruction step.

### Example (sample run)

```json
{
  "timestamp_utc": "2026-05-18T22:24:52Z",
  "agent": {
    "id": "cc_sonnet",
    "class": "claude_code",
    "name": "claude-code",
    "version": "@anthropic-ai/claude-code@2.1.85",
    "model": "anthropic/claude-sonnet-4-6",
    "runtime": "vm",
    "config_repr": {
      "model": "anthropic/claude-sonnet-4-6",
      "max_turns": 20, "timeout_s": 900,
      "dangerously_skip_permissions": true
    }
  },
  "task": { "slug": "demo__demo_desktop_note_linux",
            "path": "tasks/demo/demo_desktop_note_linux",
            "variant_index": 0 },
  "status": "failed", "score": 0.0,
  "termination": {
    "reason": "failed", "phase": "agent_run", "category": null,
    "error": {
      "type": "Exception",
      "message": "agent failed (rc=1) | LLM auth failed (check api keys)",
      "traceback": "agent failed (rc=1) | LLM auth failed (check api keys)"
    }
  },
  "timings": { "duration_s": 425.63 },
  "usage": { … total_steps: 2, all tokens 0 … },
  "schema_version": 2,
  "run_id": "cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746"
}
```

The `phase: "agent_run"` + `error.message` immediately tell you the
agent's CLI exited rc=1 and the wrapper-level diagnoser flagged it as
auth. Category is `null` because the **wrapper** diagnostic text
("agent failed (rc=1) | LLM auth failed (check api keys)") doesn't
contain any of the substrings above — the underlying transcript does,
but the category classifier only sees the wrapper's `str(exc)`. This
is a known coarse-classifier gap; rely on `error.message` for triage
when category is null and phase is `agent_run`.

---

## 5. `trajectory.json` — ATIF-v1.0 episode

Written **once** at the very end by `RunWriter.write_trajectory` via
`trajectory.model_dump_json(indent=2)`. It is a Pydantic model
serialization; the schema below is the model definition (the deployer-
agnostic `Trajectory` lives in `ale.agents.trajectory`).

### Top-level shape

```jsonc
{
  "schema_version": "ALE-v1.0",
  "episode_id":     "<32-char uuid4 hex>",
  "agent": {
    "name":    "<deployer config.name>",
    "version": "<deployer.version or null>",
    "model":   "<config.model>",
    "extra":   { … }
  },
  "task_path":      "<original task path, NOT slugified>",
  "variant_index":  <int>,
  "instruction":    "<exact prompt sent to the agent>",
  "steps":          [ <Step>, … ],
  "final_metrics":  <FinalMetrics>,
  "started_at":     "<RFC3339-with-microseconds UTC>",
  "ended_at":       "<RFC3339-with-microseconds UTC>",
  "subagent_trajectories": [ <Trajectory>, … ],
  "continued_trajectory_ref": null | "<rel path of preceding chunk>",
  "extra":          { … agent-specific debug metadata … }
}
```

### `Step` (one element of `steps[]`)

```jsonc
{
  "step_id":   <int, monotonically increasing from 1>,
  "timestamp": "<RFC3339-with-microseconds UTC, when the step was added>",
  "source":    "user" | "agent" | "environment" | "system",
  "message":   <string | [<ContentPart>, …] | null>,
  "reasoning": <string | null>,
  "tool_calls": [<ToolCall>, …],
  "observation": <Observation | null>,
  "metrics":     <StepMetrics | null>,
  "extra":       { … }
}
```

Sub-shapes:

```jsonc
ToolCall    : { "id":"call_<12hex>", "name":"<tool name>", "arguments":{…} }

Observation : { "results":[<ToolResult>,…], "error":null|"<str>" }

ToolResult  : { "tool_call_id":"<matches a prior ToolCall.id>",
                "content":[<ContentPart>,…],
                "is_error":false|true }

ContentPart : { "type":"text", "text":"<str>" }                       // or:
              { "type":"image", "image": <ImageSource> }

ImageSource : { "type":"path"|"url"|"base64",
                "path":"<rel to run_dir>"|null,
                "url":null|"<https://…>",
                "data":null|"<base64>",
                "media_type":"image/png",
                "alt_text":null|"<str>" }

StepMetrics : { "input_tokens":           <int|null>,
                "output_tokens":          <int|null>,
                "cache_read_tokens":      <int|null>,
                "cache_creation_tokens":  <int|null>,
                "cost_usd":               <float|null>,
                "duration_ms":            <int|null> }
```

### Source semantics

| `source` | What it represents | Typical fields |
|---|---|---|
| `user` | Human / framework-issued instruction. Exactly **one** at step_id=1, populated by the lifecycle from `obs.instruction`. | `message` (string) |
| `agent` | One model turn. Combines text output, reasoning, and tool calls into a single step. | `message`, `reasoning`, `tool_calls`, `metrics`, `extra` |
| `environment` | Tool results / env updates returned to the agent. | `observation` (with `results` keyed to prior tool_call ids) |
| `system` | Framework note: gather failure, parse error, cancellation, missing transcript. | `message`, `extra.reason` |

The `extra.reason` strings emitted by the framework are stable enough
to be matched by tooling:

| `extra.reason` | Emitted by |
|---|---|
| `"parse_error"` | `_origin_log_pipeline` when `deployer_cls.parse_artifacts` raised |
| `"no_transcript"` | claude_code deployer when transcript.jsonl missing |
| `"no_work_dir"` | ale_claw deployer when work_dir vanished |

### `final_metrics` (rolled up at finalize)

```jsonc
{
  "total_steps":                  <int>,
  "total_input_tokens":           <int, Σ steps[*].metrics.input_tokens>,
  "total_output_tokens":          <int, Σ steps[*].metrics.output_tokens>,
  "total_cache_read_tokens":      <int>,
  "total_cache_creation_tokens":  <int>,
  "total_cost_usd":               <float, Σ non-null cost>,
  "total_duration_ms":            <int, monotonic ns since builder ctor>,
  "reward":                       <float | null>,
  "status":                       "completed" | "timeout" | "failed"
}
```

`final_metrics.status` is forced to `"failed"` whenever the lifecycle
status is not in `{"completed", "timeout", "failed"}` (cancelled /
not_executed map to "failed" for ATIF compatibility — the lifecycle's
real status lives in `run.json`).

### `extra` conventions

`Trajectory.extra` is open-ended; deployers stash agent-specific
debug metadata here. By convention each deployer uses its own
sub-key:

- **claude_code** sets `extra["claude_code"] = {exit_code, transcript_path, stderr_path}`
  and also `extra["system_events"]` (raw stream-json `type:"system"`
  events the agent emitted) and `extra["result"]` (the terminal
  `type:"result"` event when present).
- **ale_claw** sets `extra["ale_claw"] = {work_dir, version,
  transcript_path, run_status, usage, raw_transcript}` — `usage` here
  is the exact OpenClaw token accounting (with cache breakdown) which
  the per-Step `metrics` sum cannot reproduce because the harness
  doesn't propagate cache tokens to per-step boundaries.

### Long-episode chunking (not yet emitted; schema-reserved)

`continued_trajectory_ref` is reserved for splitting multi-megabyte
trajectories across multiple files (consumer concatenates by walking
the chain backwards). The current writer always emits one file; the
field is `null` in every existing run.

### Example (sample run, 2 steps)

```jsonc
{
  "schema_version": "ALE-v1.0",
  "episode_id": "0370596babc14aee851145171c977efa",
  "agent": { "name":"claude-code",
             "version":"@anthropic-ai/claude-code@2.1.85",
             "model":"anthropic/claude-sonnet-4-6", "extra":{} },
  "task_path": "demo/demo_desktop_note_linux",
  "variant_index": 0,
  "instruction": "Goal: Create a note in a text editor from staged task data.…",
  "steps": [
    { "step_id":1, "source":"user",
      "message": "Goal: Create a note…",
      "timestamp":"2026-05-18T22:20:58.780948Z", … },
    { "step_id":2, "source":"agent",
      "message":"Failed to authenticate. API Error: 401 …",
      "timestamp":"2026-05-18T22:24:23.432872Z",
      "metrics":{ "input_tokens":0, "output_tokens":0, … },
      "extra":{ "stop_reason":null } }
  ],
  "final_metrics":{ "total_steps":2, "reward":0.0, "status":"failed", … },
  "extra":{
    "system_events":[ … the stream-json init + retry events … ],
    "result":{ … the terminal "type":"result" event from claude CLI … },
    "claude_code":{
      "exit_code":1,
      "transcript_path":"<work_dir>/transcript.jsonl",
      "stderr_path":"<work_dir>/stderr.log"
    }
  }
}
```

---

## 6. `eval_result.json` — evaluation outcome

Written **once** at the very end by `RunWriter.write_eval_result`. It's
the narrowest possible view of "did the task's `evaluate()` succeed,
and what score did it return":

```jsonc
{
  "eval_status":     "success" | "failed" | "not_executed",
  "score":           <float | null>,
  "eval_duration_s": <float | null>,
  "error":           null | { … evaluate-side exception … }
}
```

Rules:

- `eval_status = "success"` ⇔ `task.evaluate()` returned a value
  (any value — score 0.0 is still success). `score` is set in this
  case; `error` is `null`.
- `eval_status = "failed"` ⇔ `task.evaluate()` raised. `score = null`,
  `error` populated.
- `eval_status = "not_executed"` ⇔ env.step(Submit()) was never
  reached (agent crashed beforehand, KeyboardInterrupt during
  agent_run, …). All other fields `null`.
- `eval_duration_s` is wall-clock around `task.evaluate()` itself,
  measured by the framework's env. `null` when not_executed.

This file is the **easiest way to compute task pass rate** across a
batch — much cheaper to parse than the full trajectory.

For the sample run:

```json
{
  "eval_status": "success",
  "score": 0.0,
  "eval_duration_s": 3.1470300420187414,
  "error": null
}
```

Even though the run as a whole failed (agent never authenticated),
`task.evaluate()` itself ran successfully and scored 0.0 — it found no
output file. The two outcomes are orthogonal and each has its own
status field for that reason.

---

## 7. `origin_log/<agent_name>/` — deployer work_dir

This is where the **deployer** (the agent driver) writes its own
artifacts. The structure inside is deployer-specific — the framework
imposes only one rule: it must be self-contained under a single
subdirectory named after the deployer's `config.name`.

How it gets populated depends on the runtime:

| Runtime | `runtime.work_dir` points at | Populating it |
|---|---|---|
| `local` | `<run_dir>/origin_log/<agent_name>/` directly | Deployer runs in host Python; files land there natively. |
| `docker` | `<run_dir>/origin_log/<agent_name>/` on host | Container bind-mounts it as `/work`; deployer in container writes to `/work`, host sees them immediately. |
| `vm` | `/home/user/.ale/<agent_name>/<run_id>/` on the VM | Deployer runs inside the VM, writes to that path; framework pulls to host at end of run (and incrementally during run if `hot_artifacts` is set). |

### vm-runtime gather mechanics

Pull is driven by `ArtifactMirror.pull_dir` from
`<runtime.work_dir>` on the VM to `<run_dir>/origin_log/<agent_name>/`
on host, via two transports:

1. **GCS bridge** (primary, when `ALE_ARTIFACT_GCS_BUCKET` is set):
   - VM:    `gsutil -m -q cp -r <vm_path> gs://<bucket>/<run_id>/origin_log/<agent_name>`
   - Host:  `gsutil -m -q cp -r gs://<bucket>/<run_id>/origin_log/<agent_name>/* <local_dest>/`
2. **CUA direct** (fallback / no bucket): recursive walk via
   `session.list_dir` + `session.read_bytes`. Per-file 3-retry with
   1s/3s/9s backoff. Files larger than 50 MB are NOT slurped via
   `read_bytes` — they're dumped head (25 MB) + sentinel + tail
   (25 MB) via `dd … | base64 -w0`, with a sibling
   `<file>.truncated` JSON marker recording original size. Unreadable
   files get a sibling `<file>.unreadable` JSON marker.

Result of the gather lands as one event:

```jsonc
{"type":"origin_log_gather_done",
 "data":{"report":{"transport":"gcs"|"cua"|"skipped",
                   "files":<int>, "error":null|"<str>"}}}
```

### vm-runtime incremental pull (during run)

If the deployer declares `hot_artifacts` (a `ClassVar tuple[str, ...]`
of filenames relative to `work_dir`), the lifecycle starts a
background `IncrementalPuller` that ticks every 15 s, calling a
`stat | tail | head | base64` round-trip on the VM and appending only
new bytes to host disk. JSONL files use boundary-safe slicing at the
last `\n` so a half-written record is held until the next tick.

After agent stop, the framework cancels the loop and runs ONE
`reconcile_final()` pass with up to 3 size-equality retries to catch
the last flush. The 60-second timeout on this reconcile bounds the
hang if the VM has become unresponsive.

This is what guarantees that **even SIGTERM mid-agent doesn't lose
the transcript** — at worst you lose ≤15 s of trailing JSONL plus
whatever was after the last `\n` at that moment.

### The two reference deployers

#### `claude_code/` (runtime: vm)

`hot_artifacts = ("transcript.jsonl", "stderr.log")`

After gather, the folder contains:

```
origin_log/claude-code/
├── prompt.txt          The exact instruction passed to the agent CLI.
├── mcp_config.json     MCP server config consumed by claude CLI.
├── run_claude.sh       Bash wrapper: env exports + pipe prompt into `claude -p -`.
│                       NOTE: ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY are
│                       inlined here. Scrub before sharing externally.
├── launch.sh           setsid wrapper around run_claude.sh. Writes claude.pid.
├── claude.pid          PID of the detached claude process.
├── done.marker         Single-line file containing `echo $?` after the agent exits.
│                       Polled by the deployer to detect completion.
├── transcript.jsonl    Stream-json transcript from claude CLI. One event per line.
└── stderr.log          stderr of the claude CLI process.
```

The deployer's `launch()` writes `prompt.txt` + `mcp_config.json` +
`run_claude.sh` + `launch.sh`, runs `launch.sh`, polls for
`done.marker`. The agent itself writes `transcript.jsonl`,
`stderr.log`, and ultimately `done.marker`.

`transcript.jsonl` event shapes the parser cares about:

```jsonc
{"type":"system","subtype":"init", … }     // → trajectory.extra.system_events
{"type":"system","subtype":"api_retry", …} // → same
{"type":"assistant","message":{
   "content":[{"type":"text","text":"…"},
              {"type":"tool_use","id":"…","name":"…","input":{…}}],
   "usage":{"input_tokens":…,"output_tokens":…,
            "cache_read_input_tokens":…,"cache_creation_input_tokens":…},
   "stop_reason":"…"}}                     // → one Step source="agent"
{"type":"user","message":{
   "content":[{"type":"tool_result","tool_use_id":"…",
               "content":"…" or [{"type":"text","text":"…"}],
               "is_error":false|true}]}}    // → one Step source="environment"
{"type":"result", … }                       // → trajectory.extra.result
```

#### `ale_claw/` (runtime: local|docker)

`hot_artifacts = ()` (no incremental pull — work_dir is host-visible
on both substrates).

After the agent runs, the folder contains:

```
origin_log/ale-claw/
├── openclaw_memory/                Long-term memory store (per-task scratchpad).
│   └── …
├── openclaw_sessions/
│   └── <task_id>/
│       ├── transcript.jsonl        Append-only: session header + per-turn entries +
│       │                            compaction entries. Parsed into ATIF Steps.
│       ├── state.json              Running totals (tokens, steps, compactions, model).
│       └── subagent-runs.jsonl     Subagent ledger (claude-code-style Task sub-runs).
└── trajectories/<traj_id>/turn_NNN/<NNNN>_api_result.json
                                    Raw LiteLLM API responses, one per turn — preserves
                                    cache token breakdown that the transcript itself
                                    loses. Aggregated into trajectory.extra.ale_claw.usage.
```

The parser glob-finds every `*/transcript.jsonl` under
`openclaw_sessions/` and walks each in order. Assistant messages
collapse text+thinking+tool_calls into one Step; tool results emit
one Step source=`environment`. Aggregated usage from the per-turn
`*_api_result.json` files lands in
`trajectory.extra.ale_claw.usage`.

---

## 8. `output/` — the task's `remote_output_dir`

Mirrored from the VM to host by `_output_pipeline`. This is the
**agent's produced output** — whatever the task config declared as its
output directory, regardless of which agent ran. The eval reads files
on the VM directly; the host mirror is for offline scoring /
debugging.

### Selection logic

The lifecycle reads the task's metadata in this priority order
(first non-empty wins):

1. `remote_output_dir` — used by the majority of LinuxTaskConfig tasks
   and most Windows tasks.
2. `output_path` — fallback for ~10 CTF / forensics tasks under
   `computing_math/`.
3. `runtime_output_dir` — fallback for 2 `transport_safety` tasks.

If none of those keys are set, the pipeline emits

```json
{"type":"output_gather_skipped", "data":{"reason":"no_output_dir_in_metadata",
 "checked_keys":["remote_output_dir","output_path","runtime_output_dir"]}}
```

and the host `output/` directory stays empty. Eval on the VM still
scores correctly; it just won't be reproducible from the host
artifacts.

### Transport

Same `ArtifactMirror.pull_dir` machinery as `origin_log` — GCS bridge
when bucket configured, CUA direct otherwise. The event emitted is
`output_gather_done` (or `_failed` / `_skipped`) with a `report`
field carrying `{transport, files, error}`.

### Failure is best-effort

Unlike the origin gather, a failure here does NOT raise — eval is
what matters. You see it in `events.jsonl` (`output_gather_failed`),
but the run still completes with whatever score eval produced. For
the sample run:

```json
{"type":"output_gather_done", "data":{
  "vm_path":"/media/user/data/agenthle/demo/demo_desktop_note_linux/base/output",
  "report":{"transport":"cua","files":0,"error":null}}}
```

The path existed on the VM but was empty (the agent never wrote any
output — auth failure prevented it).

---

## 9. Write ordering & failure semantics

Mental model: **`events.jsonl` is the only file that's append-streamed.
Everything else is written exactly once, at finalize.**

```
                  (writer ctor)
                       │
                       ▼
                ┌─────────────────┐
                │ mkdir run_dir   │  ← refuses if exists (collision guard)
                │ mkdir output/   │
                │ mkdir origin_log/│
                │ open events.jsonl (append, line-buffered, fsync per line) │
                └─────────────────┘
                       │
       lifecycle emits events all the way through here:
       run_started, provision_*, agent_*, post_launch_*, …
                       │
                       ▼  (no matter the outcome:)
                ┌──────────────────────────────────┐
                │ rw.write_trajectory(trajectory)  │
                │ rw.write_eval_result(...)        │
                │ rw.write_run_json(meta)          │
                │ rw.emit_event("run_completed",…) │
                │ rw.close()                       │
                └──────────────────────────────────┘
```

The lifecycle wraps each final write in its own `try/except` and
demotes failure to `logger.warning`. So:

- A run dir always exists (modulo the very first mkdir failing,
  which raises before the writer construction returns).
- `events.jsonl` always exists with at least `run_started`.
- The three finalize files (`trajectory.json`, `eval_result.json`,
  `run.json`) each independently may or may not exist depending on
  whether the corresponding builder reached `finalize()` without a
  fatal error inside the writer itself.
- `origin_log/<agent>/` is created empty even if `install()` never ran.
- `output/` is created empty even if no output was gathered.

Consumers should be defensive: open each file individually, treat
missing-file as a legitimate (early-failure) signal, and prefer
`events.jsonl` (`run_failed` / `run_completed`) as the authoritative
phase trace when the other files are absent.

---

## 10. Cross-references

These names are used in code & docs interchangeably; this map is the
intended canonical translation:

| In code | In log | Where it lives |
|---|---|---|
| `RunUnit` | one run dir | one `(agent_id, task_path, variant_index)` triple in yaml |
| `runtime.work_dir` | `origin_log/<agent_name>/` | host (local/docker) or `/home/user/.ale/<agent_name>/<run_id>/` (vm) |
| `task.metadata['remote_output_dir']` | `output/` | the eval-relevant output directory on the VM |
| `RunWriter.run_id` | `run_id` in every event line + `run.json` | derived once from slugified `(agent, model, task, variant, ts)` |
| `Trajectory.episode_id` | `episode_id` in `trajectory.json` | independent UUID, *not* equal to `run_id` |
| `task.evaluate()` reward | `eval_result.json.score` and `run.json.score` and `trajectory.json.final_metrics.reward` | populated only when `eval_status == "success"` |

The four `score` fields above are written from the same
`final_obs.reward` value — they cannot disagree by construction.
Trajectory `final_metrics.status` can differ from `run.json.status`
because the former is constrained to `{completed, timeout, failed}`
while the latter additionally allows `cancelled` / `not_executed`.

---

## 11. Sample run

The directory next to this doc is a real ALE run captured 2026-05-18:

```
sample_run/
├── run.json                                  schema_version=2; status=failed; phase=agent_run
├── trajectory.json                           ALE-v1.0; 2 steps; OpenRouter auth-fail at step 2
├── eval_result.json                          eval_status=success; score=0.0
├── events.jsonl                              13 events; run_started → run_completed
├── origin_log/claude-code/                   8 files; transcript.jsonl has 13 stream-json events
│   ├── prompt.txt           launch.sh        the bash wrappers + the inputs
│   ├── run_claude.sh        mcp_config.json
│   ├── claude.pid           done.marker      bookkeeping (pid + exit code)
│   ├── transcript.jsonl     stderr.log       agent output (stderr empty here)
└── output/                                   empty — agent never wrote output (auth fail)
```

It is a single failed run, deliberately — failures exercise the
phase / category / error.message paths that successes don't. The
file shapes are identical to a successful run.
