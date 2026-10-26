# Daytona snapshot tooling

Scripts that build and maintain the ALE Windows runtime snapshot on Daytona,
and re-score finished runs. Usage and the stage descriptions are in
[docs/daytona-windows.md](../../docs/daytona-windows.md).

- `build_snapshot.py`: stock Windows Server 2025 snapshot -> ALE runtime snapshot (stages: create, bootstrap, packages, data, verify, snapshot)
- `bootstrap_runtime.ps1`: in-sandbox bootstrap run by the `bootstrap` stage
- `add_git.py`, `add_softwares.py`, `add_hardened_python.py`, `slim_snapshot.py`: optional follow-ups that re-open the builder and re-snapshot
- `regrade.py`: re-grade finished runs with references staged from the host (`localref`)

`builder_state.json` (the current builder's sandbox id and completed stages) is written next to the scripts and gitignored.
