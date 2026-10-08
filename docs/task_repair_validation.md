# Task repair validation

These changes correct ten task verifiers while retaining their numerical pass thresholds. Task
code runs through the existing `TaskLoader`, `TaskDriver`, and CUA 0.2.7 session API. No provider,
executor, queue, or lifecycle changes are required.

| Task | Change | Regression coverage |
| --- | --- | --- |
| DiT | Export the public model stubs and scheduler config interface. Bound worker execution and distinguish reported candidate failures from missing worker results. | Real worker processes, explicit `PYTHONPATH`, candidate import/SystemExit failures, timeouts, output shape/numerical/CFG checks. |
| Prostate IMRT | Use the staged matRad wrapper for QA preflight and evaluation. Make dependency provisioning an explicit preparation action. | Wrapper arguments, spaces in paths, no host proxy forwarding, no installation during grading, original completion/clinical gates. |
| WGS | Count successful checks as integers, then divide by ten. | Real synthetic VCF evaluation, exact full score, empty calls despite perfect self-reports, missing evaluator data. |
| Amber minimization | Parse Bash words and trace explicit calls through static branches and invoked functions. | Quoting, variables, true/false branches, functions, duplicate/dead/ambiguous calls and unchanged MD parameters. |
| CFR | Normalize the alternative Leduc information-set encoding before best response. | Canonical, abbreviated and mixed encodings; duplicate/missing/unknown sets and action counts. |
| Poster SVG | Apply ancestor transforms and root viewport scaling to image corners. | Nested affine transforms, units, viewBox offsets, malformed/singular/off-canvas geometry. |
| Ranking recovery | Replay in two independently prepared workspaces. Require both attempts to pass every gate. | First-attempt failures cannot be erased; protected/shard integrity, honest cleanup and manifests remain required. |
| Genome browser SVG | Recognize static polygon signal geometry. | Polyline equivalence; hidden, flat, unrelated, malformed and wrong-locus signals. |
| Equity workbook | Find a real section header satisfying all required styles in one cell. | Leading source notes, split styles, missing headers, incorrect values and formulas. |
| Amber MMGBSA | Prove a topology split across stages using authoritative task paths and the original input PDB. | Absolute/relative/script-directory/SLURM paths; wrong trajectories, masks, topology roles, energy and frame counts. |

## Host environment

Install the repository and task dependencies with the documented uv workspace setup:

```bash
uv sync --all-packages --extra dev
uv run pytest tests/test_amber_submission_verifier.py \
  tests/tasks/test_dit_worker_runtime.py \
  tests/tasks/test_imrt_runtime_score_gates.py \
  tests/tasks/test_wgs_variant_calling.py \
  tests/tasks/test_cfr_game_theory_equilibrium.py \
  tests/tasks/test_poster_geometry_portability.py \
  tests/tasks/test_ranking_fresh_replay.py \
  tests/tasks/test_three_verifier_repairs.py
```

Bash parsing requires `bashlex`, declared in the task dependency package. Shell candidate code is
never executed by the Amber verifiers. DiT workers use the current Python interpreter and its
normal environment; arbitrary changes to the parent's `sys.path` are not exported to workers.
A worker killed without a result is an evaluation error with unknown attribution, not a passing
or zero-scored candidate. Ordinary candidate exceptions and candidate `SystemExit` during import
remain scored failures.

## IMRT image preparation

The staged `software/run_matrad.sh` is the authoritative way to enter the `rtplan-matrad`
environment. The task no longer guesses a different user's home, creates home-directory
symlinks, or forwards the evaluator host's proxy to the VM.

During image preparation, copy `scripts/ensure_runtime.py` to the VM and run it through the
staged wrapper. Substitute the actual task and helper paths:

```bash
bash /path/to/task/software/run_matrad.sh python -I /path/to/ensure_runtime.py --install
bash /path/to/task/software/run_matrad.sh python -I /path/to/ensure_runtime.py \
  --reference /path/to/task/reference/RTDOSE_reference.dcm
```

`--install` is an explicit provisioning operation that may access PyPI and modify the activated
environment. It pins a tested numerical dependency set. The three versions required by the
agent-facing specification are pydicom 3.0.1, pymedphys 0.41.0, and numba 0.65.0; other numerical
packages are checked by imports and functional probes, without rejecting an otherwise working
image solely because it has another version.

Task setup and grading only run preflight. They never install packages. Preflight checks DICOM
reading mode, a gamma identity case, and reference dose decoding/finite values. An invalid
reference must be fixed as reference data and does not trigger dependency reinstallation. The
original matRad replay, clinical gates, completion checks, and score threshold remain in place.

## Static shell and SVG boundaries

The minimization verifier traces constant true/false conditions, short-circuit commands,
brace groups, and called functions without arguments. Unknown branches must produce the same
explicit pmemd invocation sequence. Loops, dynamic evaluation, unsupported state mutation,
ambiguous calls, recursion beyond the analysis limit, and unsupported Bash syntax do not provide
proof. Quoted text and uncalled functions are not invocation evidence. This is a bounded static
analysis, not a general Bash interpreter.

The cross-stage verifier receives `task_dir`, `input_dir`, and `output_dir` from task metadata.
Relative initial paths and `SLURM_SUBMIT_DIR` use the task root, while `BASH_SOURCE[0]` points to
the submitted script in the output directory. Supported explicit `cd` statements update the
working directory. The verifier never guesses the authoritative input location from candidate
text. A CLI caller can supply `--task-dir` and `--input-dir`; otherwise the output directory's
parent is the task root.

SVG checks cover their supported static geometry. Unsupported CSS transforms, nested viewports,
and dynamic rendering cannot establish the newly accepted geometry.

## Validation boundary

Local unit, subprocess, SDK contract, and provisioning checks do not replace a full task run on
an official VM image. Before release, IMRT still needs full matRad/Octave replay on that image,
and Ranking needs confirmation of the default-path sudo permissions. These checks do not require
an agent/model run. Historical benchmark records should not be rewritten by applying these patches.
