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
- **147 reference tasks** across 55 industries, the current public subset of a 1,500+ task corpus. Many tasks need private data or licensed software and stay in a separate private pool. ALE uses rolling evaluation: every ~6 months we publish a new public subset with fresh instances; private tasks rotate in and retired public tasks rotate out, to limit benchmark leakage.
- Two reference agent harnesses: the official Claude Code CLI and the in-tree OpenClaw harness. 
- Curated leaderboard slices: `cli`, `near-term`, `full-spectrum`, `last-exam`, plus an `unlicensed` track.

The full corpus, the live leaderboard, and the contributor program live at
**[agents-last-exam.org](https://agents-last-exam.org/)**.

---

# Detailed guidance will be ready in two days. Thank you for the patience:)



## Citation

If you use ALE in published work, please cite the benchmark; citation
metadata coming with the v1 paper release. Until then, link to
[agents-last-exam.org](https://agents-last-exam.org/) and
[this repository](https://github.com/rdi-berkeley/agents-last-exam).

<div align="center">

**Stay updated:** [Mailing list](https://groups.google.com/g/agenthle-news) · **Contact:** [rdi_research@berkeley.edu](mailto:rdi_research@berkeley.edu)

</div>
