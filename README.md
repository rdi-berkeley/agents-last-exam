<div align="center">

<h1>
  <img src="assets/logo.png" alt="" width="43" valign="middle" />
  &nbsp;&nbsp;Agents' Last Exam
</h1>

*Challenge and measure AI agents on economically valuable, real-world tasks.*

[![Website](https://img.shields.io/badge/agents--last--exam.org-1a1a1a?style=for-the-badge&logo=googlechrome&logoColor=white)](https://agents-last-exam.org/)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-3a3a3a?style=for-the-badge)](https://agents-last-exam.org/#leaderboard)
[![License: Apache-2.0](https://img.shields.io/badge/software-Apache--2.0-3a3a3a?style=for-the-badge)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-3a3a3a?style=for-the-badge)](LICENSE-DATA)
[![Mailing list](https://img.shields.io/badge/news-subscribe-3a3a3a?style=for-the-badge)](https://groups.google.com/g/agenthle-news)

Led by **[UC Berkeley RDI](https://rdi.berkeley.edu/)** × **RDI Foundation**

<br/>

<a href="assets/teaser.pdf"><img src="assets/teaser.png" alt="ALE benchmark: domains and example workflows" width="100%" /></a>

</div>

---

Agents' Last Exam aims to build the **broadest-coverage agent
evaluation benchmark to date**, measuring performance on long-horizon,
economically valuable tasks with verifiable outcomes. Co-led by Berkeley RDI and
300+ industry experts, dALE covers non-physical industries defined with
reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). 

<table align="center">
  <tr>
    <td align="center" width="25%"><b>Broadest Coverage</b><br/><sub>55 industries<br/></sub></td>
    <td align="center" width="25%"><b>Verifiable Outcomes</b><br/><sub>Hidden references<br/>+ deterministic graders</sub></td>
    <td align="center" width="25%"><b>Long-Horizon</b><br/><sub>Multi-step workflows<br/>on real OS sandboxes</sub></td>
    <td align="center" width="25%"><b>Economically Valuable</b><br/><sub>Sourced and validated<br/>by industry experts</sub></td>
  </tr>
</table>

<!-- <table align="center">
  <tr>
    <td align="center" width="25%"><h3>50+</h3><sub>INDUSTRIES</sub></td>
    <td align="center" width="25%"><h3>1.5K+</h3><sub>COLLECTED TASKS</sub></td>
    <td align="center" width="25%"><h3>300+</h3><sub>EXPERT CONTRIBUTORS</sub></td>
    <td align="center" width="25%"><h3>$100K+</h3><sub>AWARD POOL</sub></td>
  </tr>
</table> -->

---

## About this repository

This is the open evaluation framework for ALE. It ships:

- The `ale_run` orchestration toolkit: provision sandboxes, run agents, evaluate.
- **150 reference tasks** across 55 industries, the current public subset of a 1,500+ task corpus. Many tasks need private data or licensed software and stay in a separate private pool. ALE uses rolling evaluation: every ~6 months we publish a new public subset with fresh instances; private tasks rotate in and retired public tasks rotate out, to limit benchmark leakage.
- Two reference agent harnesses: the official Claude Code CLI and the in-tree OpenClaw harness. 
- Curated leaderboard slices: `cli`, `near-term`, `full-spectrum`, `last-exam`, plus an `unlicensed` track.

The full corpus, the live leaderboard, and the contributor program live at
**[agents-last-exam.org](https://agents-last-exam.org/)**.

---

## Quick start: your first run (≈5 min on Google Cloud)

**What you'll run:** one hello-world task (`demo/hello_win`) on a GCP Windows
VM. The agent opens Notepad, types a string, saves a file; the grader checks
the result.

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), an LLM API
key (Anthropic or OpenRouter), and a Google Cloud account. New GCP accounts
get **$300 in free trial credits** (90 days), more than enough to explore
ALE before you spend anything meaningful.

### One-time GCP setup

You need your own GCP project, a service-account key, and network access to
the VM. 
Step-by-step `gcloud` commands, image forks, disk-type pitfalls, and GCS
staging: [docs/SETUP_GCP.md](docs/SETUP_GCP.md).

### Clone, configure, run

```bash
git clone https://github.com/rdi-berkeley/agents-last-exam.git && cd agents-last-exam
uv sync --extra dev

cp secret/.env.example secret/.env       # fill in API keys + GCP project
uv run python -m ale_run run example_exp.yaml --dry-run   # validate config
uv run python -m ale_run run example_exp.yaml             # real run
```

> **Evaluator credentials (`secret/eval_time/`).** Some tasks score their output
> with an LLM/VLM judge (OpenAI or Gemini). Those judge keys live in **per-service**
> files under `secret/eval_time/`, shipped as redacted `*.env.example` templates.
> Copy each one you need to its real filename and fill in the key — the judges
> load them automatically at scoring time (no manual `source` needed):
>
> ```bash
> cp secret/eval_time/openai.env.example secret/eval_time/openai.env   # OpenAI vision/LLM judges
> cp secret/eval_time/gemini.env.example secret/eval_time/gemini.env   # Gemini video judge
> # then edit each file and fill in the API key(s)
> ```
>
> Real `secret/eval_time/*.env` files are gitignored; only the `.example`
> templates are committed. A task's `task_card.json` `evaluatorCredentials` field
> names which file that task's judge needs.

[`example_exp.yaml`](example_exp.yaml) is intentionally minimal: only four
blocks (`name`, `agent`, `environment`, `tasks`). It runs the
[`demo/hello_win`](tasks/demo/hello_win/) task on a GCP Windows VM with
the Claude Code CLI as the agent.

Successful runs print a results table and write artifacts under
`.logs/<experiment>/<run_id>/`:

```
agent                 task                                      var  status      score     dur
----------------------------------------------------------------------------------------------
claude_code           demo/hello_win                              0  completed    1.00   42.3s
```


---

## The three target environments

| Env | Status | Use case | Setup doc |
|---|---|---|---|
| **GCloud** | ✅ supported | Large-scale leaderboard runs, parallel VMs | [docs/SETUP_GCP.md](docs/SETUP_GCP.md) |
| **VMWare** | 🚧 TODO | Small-scale or single-task debugging on your own hardware | [docs/SETUP_VMWARE.md](docs/SETUP_VMWARE.md) |
| **Local** | 🚧 TODO | Run the agent against your local machine | [docs/SETUP_LOCAL.md](docs/SETUP_LOCAL.md) |

**GCloud** is the fully supported path today. Copy the published image
`ale-unified-v1` into your project (see
[docs/SETUP_GCP.md](docs/SETUP_GCP.md)); it boots a Windows or Ubuntu
sandbox with `cua-server` and the agent CLIs pre-installed. Point
your experiment's `environment:` at a gcloud env config such as
`configs/environments/gcloud_ubuntu.yaml` (it sets `provider: gcloud`)
and fill in your project via its `${env:GCP_PROJECT}` references.

**VMWare** and **Local** providers are not yet implemented (see TODO
notes in the respective docs). As an interim, the `static` provider
([`ale_run/environments/providers/static.py`](ale_run/environments/providers/static.py))
can wrap any pre-existing VM that exposes `cua-server` on TCP 5000,
useful if you bring up a VMware/local VM by hand.

---

## Running tasks

Pick a task list and pass its path under `tasks:` in your experiment yaml.

| Track | List | Tasks | Notes |
|---|---|---|---|
| **Hello-world** | `selected_tasks/helloworld.txt` | 1 | What `example_exp.yaml` uses by default |
| **CLI-only leaderboard** | `selected_tasks/cli.txt` | 106 | Terminal/code tasks. **TODO: Docker image not yet published.** |
| **Full benchmark** | `selected_tasks/full/{near-term,full-spectrum,last-exam}.txt` | 59 / 55 / 36 | Requires licensed software for ~10 tasks (see below) |
| **Unlicensed track** | `selected_tasks/unlicensed/{near-term,full-spectrum,last-exam}.txt` | 59 / 50 / 33 | Recommended for first full-benchmark run; runs against the published image as-is |

The three full-benchmark tiers:

- **Near-Term**: workflows current frontier agents can partially complete,
  top pass rates reaching ~30%. The most cost-effective target for
  short-term leaderboard competition and rapid iteration.
- **Full-Spectrum**: at least one task per each of ALE's 55 industries.
  Ensures broad coverage for comprehensive evaluation.
- **Last-Exam**: the hardest workflows, on which most agents achieve a
  0% pass rate. Anchors the benchmark's long-term headroom; reserve for
  milestone evaluations, not routine testing.

For the YAML form of these lists (per-task variant selection), see
e.g. [`selected_tasks/unlicensed/near-term.yaml`](selected_tasks/unlicensed/near-term.yaml).

**Licensed software (TODO)**: full-benchmark tasks need ~10 commercial
applications pre-installed and signed in on the VM image. The complete
list and license-account setup checklist is pending; until then, prefer
the unlicensed track:

| Software | Tasks using it | License type | Setup notes |
|---|---|---|---|
| _TODO_ | _TODO_ | _TODO_ | _TODO_ |

See [docs/RUN_TASKS.md](docs/RUN_TASKS.md) for the full task-selection guide.

---

## System overview

The quick start above runs *one* agent on *one* environment against
*one* task. Read this section if you want to go beyond that — to
**benchmark your own agent on ALE**, **run ALE on your own
infrastructure**, or **author a new task** for the benchmark.

ALE keeps **agent**, **environment**, and **task** as three
independent slots. An experiment YAML picks one of each; swapping
any single slot leaves the other two unchanged.

```
                ┌────────────────────────────────┐
                │        experiment.yaml         │
                │   agent + environment + tasks  │
                └────────────────┬───────────────┘
                                 │  orchestrator wires them up
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
   │     AGENT      │   │  ENVIRONMENT   │   │      TASK      │
   │  what to test  │   │  where to run  │   │  what to do    │
   │                │   │                │   │   + grader     │
   ├────────────────┤   ├────────────────┤   ├────────────────┤
   │  claude_code   │   │     gcloud     │   │  150 tasks /   │
   │  ale_claw      │   │     static     │   │  55 industries │
   │  your harness  │   │  your provider │   │  + your task   │
   └────────────────┘   └────────────────┘   └────────────────┘
```

At runtime the orchestrator provisions the environment, stages the
task's inputs onto it, hands the prompt to the agent, then runs the
task's grader against the artifacts the agent leaves behind. Because
each piece talks to the others through a stable contract, a new
agent runs against existing tasks unchanged, and a new task runs on
the existing environments unchanged.

### Extending each slot

Pick the guide that matches what you want to do — each one points at
a reference implementation you can copy:

| Goal | What you implement | Guide |
|---|---|---|
| Benchmark **your own agent** | A [`BaseAgentDeployer`](ale_run/base_interface/agent_deployer.py) — `install` / `launch` / `parse_artifacts`. Models: [`claude_code`](ale_run/agents/claude_code/deployer.py) (CLI baked into the image), [`ale_claw`](ale_run/agents/ale_claw/) (host-side harness). | [docs/EXTEND_AGENTS.md](docs/EXTEND_AGENTS.md) |
| Run ALE on **your own infra** | A [`Provider`](ale_run/base_interface/sandbox.py) — `acquire` / `release` / `open_session`. Reuse the published [`ale-unified-v1`](docs/SETUP_GCP.md) image, or register your own. Model: [`gcloud`](ale_run/environments/providers/gcloud.py). | [docs/EXTEND_ENVIRONMENTS.md](docs/EXTEND_ENVIRONMENTS.md) |
| Add **a new task** | A two-file package: `task_card.json` (metadata + VM spec) and `main.py` (`load` / `start` / `evaluate`). Template: [`tasks/demo/hello_win/`](tasks/demo/hello_win/). The VM-side session API: [docs/SESSION_API.md](docs/SESSION_API.md). | [docs/EXTEND_TASKS.md](docs/EXTEND_TASKS.md) |

---

## Repo layout

```
agents-last-exam/
├── ale_run/                  Framework code (cli.py is the entry point)
│   ├── agents/                 Pre-installed agents: claude_code, ale_claw
│   ├── base_interface/         Provider / Executor / Deployer / Trajectory ABCs
│   ├── environments/           Providers (gcloud, static) + image registry
│   ├── executors/              sandbox / local / docker(stub) substrates
│   ├── orchestration/          Run lifecycle, config loader, factories
│   └── tasks/                  Task discovery + driver
├── tasks/                    151 task packages, grouped by domain
│   └── demo/                   `hello` (Linux) + `hello_win` (Windows) templates
├── configs/                  Reusable agent + environment configs (referenced by path)
├── selected_tasks/           Curated task lists (cli, full, unlicensed)
├── secret/                   `.env.example` + GCP keys; `eval_time/*.env.example` judge keys (real values gitignored)
├── docs/                     Setup, task-running, and extension guides
├── sample_run/               A recorded run's output (events, trajectory, eval)
├── example_exp.yaml          The minimal experiment; start here
└── pyproject.toml            uv workspace; Python ≥3.12, <3.14
```

---

## Where to go next

- [docs/SETUP_GCP.md](docs/SETUP_GCP.md): full Google Cloud walkthrough (VPC,
  service account, image, firewall).
- [docs/RUN_TASKS.md](docs/RUN_TASKS.md): picking and running task
  batches; tier descriptions and licensed-software status.
- [docs/EXTEND_AGENTS.md](docs/EXTEND_AGENTS.md): plug in your own
  agent (CLI in image / host-side harness / docker, TODO).
- [docs/EXTEND_TASKS.md](docs/EXTEND_TASKS.md): author a new task.
- [docs/EXTEND_ENVIRONMENTS.md](docs/EXTEND_ENVIRONMENTS.md): add a new
  cloud or sandbox provider.
- [docs/SESSION_API.md](docs/SESSION_API.md): `cb.RemoteDesktopSession`
  reference for task authors (everything you can call on the VM).
- [sample_run/](sample_run/): what successful run artifacts look like.

---

## Contributing

PRs welcome for:

- new environment providers (Azure, AWS, on-prem hypervisors)
- new tasks (mirror `tasks/demo/hello` and submit under the right domain)
- new agent deployers (any CLI or SDK harness)

Code-style and review rules live in [AGENTS.md](AGENTS.md). For
non-trivial changes, open an issue describing the scope before sending
the PR. Domain experts who want to contribute workflows without writing
code can submit directly through [agents-last-exam.org/submit](https://agents-last-exam.org/submit).

---

## Partners

<div align="center">

**Led by**

UC Berkeley RDI · RDI Foundation

**Academic institutions**

UC Berkeley · MIT · Stanford · Harvard · Oxford · USC ·
UC San Diego · UCSF · Syracuse · NIH · University of Colorado ·
Peking University · SJTU

**Industry partners**

Genentech · Hippocratic AI · Goldman Sachs · Morgan Stanley · JPMorgan ·
Citadel · PIMCO · Tesla · Meta · TDK · Brix · Photon Fund

**Sponsorship**

Snorkel AI · Unipat AI

</div>

The full advisory committee and contributor roster is on
[agents-last-exam.org](https://agents-last-exam.org/).

---

## License

This repository uses a split license:

| Component | License | Scope |
|---|---|---|
| **Software** | [Apache-2.0](LICENSE) | `ale_run/`, `configs/`, `docs/`, `tests/`, and other framework files at the repo root |
| **Data** | [CC-BY-4.0](LICENSE-DATA) | `tasks/`, `selected_tasks/`, `sample_run/`, and other benchmark task content |

---

## Citation

If you use ALE in published work, please cite the benchmark; citation
metadata coming with the v1 paper release. Until then, link to
[agents-last-exam.org](https://agents-last-exam.org/) and
[this repository](https://github.com/rdi-berkeley/agents-last-exam).

<div align="center">

**Stay updated:** [Mailing list](https://groups.google.com/g/agenthle-news) · **Contact:** [rdi_research@berkeley.edu](mailto:rdi_research@berkeley.edu)

</div>
