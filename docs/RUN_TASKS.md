# Running tasks

ALE ships several curated task lists. Pick one and reference its path
under `tasks:` in your experiment yaml:

```yaml
tasks: selected_tasks/unlicensed/near-term.txt
```

The framework loads the list at config-parse time
([`ale_run/orchestration/config_loader.py`](../ale_run/orchestration/config_loader.py)),
expands each entry into one `RunUnit` per variant, then enumerates the
Cartesian product `agents × tasks × variants` for the runner.

---

## Task lists

| Track | List | Tasks | Notes |
|---|---|---|---|
| Hello-world | `selected_tasks/helloworld.txt` | 1 | Smoke test. What `example_exp.yaml` uses. |
| CLI-only leaderboard | `selected_tasks/cli.txt` | 106 | Terminal/code tasks. **TODO: Docker image not yet published.** |
| Full benchmark — Near-Term | `selected_tasks/full/near-term.txt` | 59 | Most cost-effective tier |
| Full benchmark — Full-Spectrum | `selected_tasks/full/full-spectrum.txt` | 55 | One task per subdomain |
| Full benchmark — Last-Exam | `selected_tasks/full/last-exam.txt` | 36 | Hardest tier (~0% pass rate) |
| Unlicensed — Near-Term | `selected_tasks/unlicensed/near-term.txt` | 59 | No license required |
| Unlicensed — Full-Spectrum | `selected_tasks/unlicensed/full-spectrum.txt` | 50 | No license required |
| Unlicensed — Last-Exam | `selected_tasks/unlicensed/last-exam.txt` | 33 | No license required |

### File formats

`.txt` — one task path per line, runs variant 0 only.
[`selected_tasks/cli.txt`](../selected_tasks/cli.txt) is the canonical
example. `#` starts a comment; blank lines are ignored.

`.yaml` — same paths plus per-task variant control:

```yaml
- path: demo/hello
  variants: [0, 1, 2]      # run all three variants of this task
- path: demo/hello_win
  variants: [0]
```

[`selected_tasks/unlicensed/near-term.yaml`](../selected_tasks/unlicensed/near-term.yaml)
shows the full pattern.

---

## The three tiers

### Near-Term (~30% pass rate ceiling)

Workflows that current frontier agents can **partially** complete, with
top pass rates reaching ~30%. These are the most cost-effective target
for short-term leaderboard competition and rapid iteration.

When to use: weekly leaderboard runs, iterating on a new agent, A/B
testing prompts.

### Full-Spectrum (broad coverage)

Covers, **by design**, each of ALE's 55 subdomains with at least one
task instance. Ensures broad domain coverage for comprehensive
evaluation.

When to use: monthly leaderboard updates, comparing agents across
domains, reporting headline numbers.

### Last-Exam (~0% pass rate)

The hardest workflows in the benchmark, on which most agents achieve a
0% pass rate. Anchors the benchmark's long-term headroom and is best
reserved for milestone evaluations, **not routine testing**.

When to use: quarterly capability checkpoints, frontier-model releases,
research write-ups.

---

## Licensed vs. unlicensed tracks

The **full benchmark** includes ~10 tasks that need commercial software
pre-installed on the VM image with an active license signed in. The
**unlicensed track** is a strict subset that excludes these tasks, so
it runs against the published `ale-unified-v1` image with zero
extra setup.

**Recommended first full-benchmark run:** unlicensed Near-Term (59 tasks,
no license setup).

### Licensed software (TODO)

The full list of licensed applications and their per-task usage is
pending. To be populated:

| Software | Tasks using it | License type | Account setup |
|---|---|---|---|
| _TODO_ | _TODO_ | _TODO_ | _TODO_ |

When this table is filled in, each row will link to a per-software
setup guide covering: how to obtain the license, how to inject it into
a custom GCP image, how to keep the account signed in across VM boots.

Until then, if you need a licensed task you can rebake the image
yourself — see step 6 of [SETUP_GCP.md](SETUP_GCP.md) for the image-build
recipe.

---

## Filtering at the command line

`ale run` supports filtering without editing the yaml:

```bash
# only run a specific agent within the experiment's agent list
uv run python -m ale_run run my_exp.yaml --agent claude_code

# only run specific tasks (repeatable)
uv run python -m ale_run run my_exp.yaml \
  --task demo/hello \
  --task demo/hello_win
```

See [`ale_run/cli.py`](../ale_run/cli.py) for the full flag set.

`--dry-run` prints the unit matrix without executing anything — useful
to sanity-check a large task list before spending VM minutes.

---

## Output layout

Runs land under `<output_root>/<experiment>/<run_id>/`. Default
`output_root` is `.logs/` (gitignored). The shape per-unit follows
`sample_run/LOG_SPEC.md`:

```
.logs/<experiment>/<run_id>/
└── <agent>/<task-slug>/v<variant>/<timestamp>/
    ├── run.json           run metadata + final score
    ├── events.jsonl       per-event log stream (provision, setup, agent, eval, cleanup)
    ├── trajectory.json    ATIF trajectory (agent steps, tool calls, observations)
    ├── eval_result.json   evaluation hook output
    ├── origin_log/        raw artifacts pulled from the VM
    └── output/            agent-produced files (per `artifacts_path.output_path`)
```

A complete recorded run lives in [`sample_run/`](../sample_run/) — open
it before your first real run to see exactly what to expect.
