<div align="center">

<h1>
  <img src="assets/logo.png" alt="" width="43" valign="middle" />
  &nbsp;&nbsp;Agents' Last Exam
</h1>

*Challenge and measure AI agents on economically valuable, real-world tasks.*

[![Website](https://img.shields.io/badge/agents--last--exam.org-1a1a1a?style=for-the-badge&logo=googlechrome&logoColor=white)](https://agents-last-exam.org/)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-3a3a3a?style=for-the-badge)](https://agenthle.org/leaderboard)
[![License: Apache-2.0](https://img.shields.io/badge/software-Apache--2.0-3a3a3a?style=for-the-badge)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-3a3a3a?style=for-the-badge)](LICENSE-DATA)
[![Mailing list](https://img.shields.io/badge/news-subscribe-3a3a3a?style=for-the-badge)](https://groups.google.com/g/agenthle-news)

Led by **[UC Berkeley RDI](https://rdi.berkeley.edu/)** × **RDI Foundation**

<br/>

<a href="assets/teaser.pdf"><img src="assets/teaser.png" alt="ALE benchmark: domains and example workflows" width="100%" /></a>

</div>

---

Agents' Last Exam aims to build the **broadest-coverage agent evaluation
benchmark to date**, measuring performance on long-horizon, economically
valuable tasks with verifiable outcomes. Co-led by Berkeley RDI and 300+
industry experts, ALE covers non-physical industries defined with reference to
O\*NET / SOC 2018 (the U.S. federal occupational taxonomy).

<table align="center">
  <tr>
    <td align="center" width="25%"><b>Broadest Coverage</b><br/><sub>55 industries<br/></sub></td>
    <td align="center" width="25%"><b>Verifiable Outcomes</b><br/><sub>Hidden references<br/>+ deterministic graders</sub></td>
    <td align="center" width="25%"><b>Long-Horizon</b><br/><sub>Multi-step workflows<br/>on real OS sandboxes</sub></td>
    <td align="center" width="25%"><b>Economically Valuable</b><br/><sub>Sourced and validated<br/>by industry experts</sub></td>
  </tr>这里首先把我们刚刚在 README 里关于 Agent 的讨论和分类详细说清楚，然后可以稍微展开一点，说得细一点。

然后再讲到 Deployer 这个抽象接口，以及它和 Executor 的关系：
1. In-Sandbox Agent：就是用 Sandbox Executor 直接在 Sandbox 里运行 Deployer。
2. Out-of-Sandbox Agent：可以选择 Yum Local 或 Yum Docker。

其中 Docker 是为了把 Agents 的运行环境和本地工作环境隔离开，从而获取一个更安全的情况，避免 Agents 误删掉本地的重要文件。
</table>

This repository is the **open evaluation framework**: the `ale_run` toolkit that
provisions sandboxes, runs agents, and grades them, plus **150 reference tasks**
across 55 industries (the current public subset of a 1,500+ task corpus) and two
reference agent harnesses. The full corpus, live leaderboard, and contributor
program live at **[agenthle.org](https://agenthle.org/)**.

---

## Quick start

One command boots a real cloud sandbox, runs an agent on a hello-world task, and
grades the result — after a one-time Google Cloud setup (~10 min, covered by the
$300 free trial).

→ **[docs/SETUP_GCP.md](docs/SETUP_GCP.md)** walks it end to end: create a
project, copy the sandbox image, fill in two keys, and run your first task. You can 
manually setup your account then hand the doc to your coding agents to finish the rest.

---

## How ALE works

ALE targets **frontier agent systems** — a harness orchestrating a foundation
model, carrying its own action loop, tools, memory, and sub-agents. Rather than
puppeteer such a system step by step — which would strip away the very machinery
that makes it capable — ALE hands it **only a task description**, lets it **work
to completion** on a real machine, and **scores the artifacts it leaves behind**.
Each system keeps its full capabilities, and very different agents become
comparable on the one axis that matters: did the work get done.

Every run is built from three interchangeable pieces:

- **Agent harness** — the system under test (Claude Code, Codex, Openclaw, …), a harness driving a foundation model through its own loop. Real
  workflows need both a terminal and a screen, so ALE evaluates what the paper
  calls **Generalist CUA-agents**: agents that combine CLI *and* GUI, not just
  one. Most harnesses are CLI-native, so ALE lifts any of them to that surface
  with a **unified, cross-OS CUA MCP bridge** — desktop actions (screenshot,
  click, type, scroll, …) exposed as ordinary tools in the agent's loop.
- **Environment (sandbox)** — a virtual machine that **faithfully reproduces the
  real production work context**: a full Windows or Linux OS with the actual
  professional software installed and the task's real data on disk — not a
  simplified or sanitized environment.
- **Task** — a unit of real professional work, written as an executable
  `main.py`: an instruction, its input data, and a
  **hidden reference** that the grader `evaluate()` scores the output against,
  in [0, 1].

### A run, end to end

An experiment is just a pairing — **one agent × one environment × one task** —
that the orchestrator runs through a fixed loop:

```
  provision the sandbox  →  stage the task's inputs  →  run the agent to completion
       →  stage the hidden reference  →  grade the output  →  score + collect logs and a unified trajectory
```

The hidden reference is staged **only after** the agent finishes, so the answer
can never leak into the run. Every run is then recorded in full — a **uniform
trajectory** (each step, tool call, and observation in one schema), the agent's
**raw logs** (transcripts and the like), the **evaluation result**, and the
**artifacts in play** (the files the agent wrote, the screenshots it saw) — so a
run can be replayed and audited end to end.

A harness reaches the sandbox in one of two shapes. **In-sandbox** harnesses are
CLIs that run inside the VM: at launch, ALE injects the CLI and the CUA MCP bridge
into the freshly-booted machine. A **out-of-sandbox** harness instead runs in ALE's own process, outside the VM. It
operates on two fronts at once: it drives the VM remotely through **two MCP
bridges — one CLI-based (shell, files), one GUI** — while running its own
out-of-VM machinery (memory, sub-agents, context management) right alongside.
**ALE-Claw** is the reference: [ale_run/agents/ale_claw](ale_run/agents/ale_claw/README.md).

> Deeper dives into the system design — the sandbox & providers, the executor/deployer split, task data
> & grading, and the trajectory format — live in the **[documentation site](#)**.

---

## Running the benchmark

Past the demo, ALE ships curated task lists across three difficulty tiers
(near-term, full-spectrum, last-exam) and several leaderboard slices
(`cli`, `unlicensed`, …).

- **Choosing & running task lists with your own configurations** — tracks, providers, agent configs → **[docs](#)**
- **Best practices for full-benchmark runs** — concurrency, output pullback, licensed software → **[docs](#)**
- **Browse the tasks and the results** — the live **[tasks gallery](https://agenthle.org/demo)**, trajectory viewers **[TODO](#)** 

---

## Build on ALE

ALE is built to be extended, and we want your contributions.

- **Benchmark your own agent harness** — implement a small deployer; any CLI or SDK works → **[docs](#)**
- **Submit results to the leaderboard** → **[agenthle.org/leaderboard](https://agenthle.org/leaderboard)**
- **Contribute a task** — mirror a demo task and submit under the right domain → **[docs](#)**
- **Add an environment provider** — Azure, AWS, on-prem hypervisors → **[docs](#)**

Domain experts can also submit workflows without writing code at
**[agenthle.org/submit](https://agenthle.org/submit)**.

---

## Repository layout

```
agents-last-exam/
├── ale_run/                  Framework code (python -m ale_run is the entry point)
│   ├── agents/                 Agent deployers: claude_code, ale_claw, …
│   ├── base_interface/         The contracts: Provider / Executor / Deployer / Trajectory
│   ├── environments/           Providers (gcloud, static) + image registry
│   ├── executors/              Where a deployer runs: sandbox / local / docker
│   ├── orchestration/          Run lifecycle, config loader, factories
│   └── tasks/                  Task discovery + driver
├── tasks/                    Task packages, grouped by domain (demo/ has the templates)
├── configs/                  Reusable agent + environment configs (referenced by path)
├── selected_tasks/           Curated task lists (cli, full, unlicensed)
├── secret/                   .env + GCP key + per-judge eval keys (real values gitignored)
├── docs/                     Setup, task-running, and extension guides
├── example_exp.yaml          The minimal experiment; start here
└── pyproject.toml            uv workspace; Python ≥3.12, <3.14
```

Code-style and review rules: [AGENTS.md](AGENTS.md). For non-trivial changes,
open an issue describing the scope before sending a PR.

---

## Partners

<div align="center">

**Led by** · UC Berkeley RDI · RDI Foundation

**Academic** · UC Berkeley · MIT · Stanford · Harvard · Oxford · USC ·
UC San Diego · UCSF · Syracuse · NIH · University of Colorado · Peking University · SJTU

**Industry** · Genentech · Hippocratic AI · Goldman Sachs · Morgan Stanley · JPMorgan ·
Citadel · PIMCO · Tesla · Meta · TDK · Brix · Photon Fund

**Sponsorship** · Snorkel AI · Unipat AI

</div>

The full advisory committee and contributor roster is on
[agenthle.org](https://agenthle.org/).

---

## License

| Component | License | Scope |
|---|---|---|
| **Software** | [Apache-2.0](LICENSE) | `ale_run/`, `configs/`, `docs/`, and other framework files |
| **Data** | [CC-BY-4.0](LICENSE-DATA) | `tasks/`, `selected_tasks/`, `sample_run/`, and other benchmark content |

---

## Citation

If you use ALE in published work, please cite the benchmark; citation metadata
comes with the v1 paper release. Until then, link to
[agenthle.org](https://agenthle.org/) and this repository.

<div align="center">

**Stay updated:** [Mailing list](https://groups.google.com/g/agenthle-news) · **Contact:** [rdi_research@berkeley.edu](mailto:rdi_research@berkeley.edu)

</div>
