# Task software artifact — Lean 4.27.0 + Mathlib v4.27 (OPTIONAL offline fallback)

> **NOT used by default.** The task fetches Lean + the Mathlib cache from
> upstream at setup (`start()` → `elan` + `lake exe cache get`), so no `software/`
> upload is needed and the submission is code-only. This document describes an
> *optional* offline/hermetic fallback: pre-staging the same prebuilt cache as
> task-data `software/`, e.g. if the pinned upstream Mathlib cache is ever
> garbage-collected. To use it, set `REQUIRES_TASK_DATA = True` in `main.py` and
> point the toolchain paths at `software_dir` (see git history for that variant).

This fallback follows the standard ALE data model (same as
`engineering/chisel_verilog_alignment_seq_1`, which ships yosys/firtool/sbt):
the heavy toolchain is **task-data `software/`**, staged by the framework into
the sandbox's `software_dir` (from `gs://ale-data-public` in production, or baked
into the local image), NOT baked into a custom Docker image and NOT committed to
this git repo.

## Required `software/` layout (staged to `<base>/software/`)

```
software/
├── elan/
│   └── toolchains/leanprover--lean4---v4.27.0/    # the Lean 4.27.0 toolchain
│       └── bin/lean                                # → main.py's lean_bin
└── mathlib/
    └── packages/<pkg>/.lake/build/lib/lean/*.olean # prebuilt v4.27 oleans
        # <pkg> ∈ {aesop, batteries, Cli, importGraph, LeanSearchClient,
        #          mathlib, plausible, proofwidgets, Qq}
```

`main.py` computes `lean_bin` and `LEANPATH` from these paths; `start()` writes
`input/LEANBIN` and `input/LEANPATH` pointer files and symlinks `lean` onto PATH.

Total size ≈ 8.1 GB (elan 2.5 GB + oleans 5.6 GB; mathlib alone is 5.5 GB /
7524 oleans).

## Building the artifact

Produced from a machine with Lean 4.27.0 (`elan`) and a built Mathlib v4.27
`.lake` cache (e.g. a `lake exe cache get` checkout). Copy the toolchain and the
nine dependency `build/lib/lean` olean dirs into the layout above. A reference
build script is `scripts/build_software.sh`.

## Submission

Upload the `software/` tree to the ALE task-data location for
`computing_math/imo2005p6_lean_v1` (the reviewers place it under
`<gcs_prefix>/software/`). This is the same mechanism every heavy-software ALE
task uses; no custom image is required.

## Local calibration

Bind-mount a locally-built `software/` tree to the sandbox `software_dir`:
```
# (local-dev only; requires a docker provider that supports extra RO bind-mounts)
ALE_DOCKER_EXTRA_MOUNTS="<local software/>:<software_dir>"
```
(verified: reference proof compiles + axiom-audits clean from this mount).
