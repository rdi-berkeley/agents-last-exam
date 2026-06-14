# Dev-VM Environment Gaps (record only — do NOT auto-patch)

This file = **deliverable 4**: task environments that are **needed but not cleanly available on the dev VM
`dev-ubuntu22`** (project agenthle-488519, zone us-west2-a). Per instruction, these are *recorded, not
auto-fixed* — they need an owner decision (rebuild on dev VM? bake into the image? re-pull container images?).

Method: verified by SSH on 2026-06-14 — checked `/opt`, `/opt/toolchains`, `/home/*/.local`, conda/micromamba
envs, `docker images`, and `which`. Important correction to an earlier `/opt`-only view: **a lot of the
benchmark's heavy software lives under `/home/user/.local/` (per-user pip/micromamba installs by the `user`
account), not in `/opt`.** Those are listed here as "per-user only" because they are not reproducible from
`/opt` or from the task data `software/` dir, so they need proper packaging regardless.

---

## 1. Present on dev VM ONLY as per-user installs (`/home/user/.local`) — not in /opt, not in task data

These exist and work for the `user` account, but are invisible to a fresh container and not packaged.
They are the real blockers for the CT/imaging tasks.

### LEAP (LLNL CT projector) + CPU PyTorch stack
- **Where:** `/home/user/.local/lib/python3.10/site-packages/leaptorch.py` (+ `leapctype`), and a CPU torch in
  `/home/user/.local/agenthle/graphst-env/` (`torch-2.11.0+cpu`). Supporting: numpy 2.2.6, scipy 1.15.3,
  scikit-image 0.25.2, imageio 2.37.3 in the same per-user site-packages.
- **Tasks:** `health_medicine/ct_geometry_calibration_catphan`, `health_medicine/limited_angle_ct_dps_reconstruction`.
  Both tasks' `software/` wrappers assume `import leaptorch, torch` from system site-packages — which is empty in a clean container.
- **Gap:** no `/opt` package, no task `software/` bundle. Needs a reproducible package: **LEAP 1.26 CPU build
  (from LLNL source, GPU symbols stubbed) + torch-cpu + scipy + scikit-image + imageio.** This is a heavy
  source build — flagged for a decision before I build it (don't want to guess the exact LEAP build flags).

### matRad QA micromamba env — NO GAP (verified, matches ground truth)
- **Where:** `/home/user/.local/share/micromamba/envs/rtplan-matrad` (per-user); matRad source at `/opt/matrad-c014dc82`.
- **Task:** `health_medicine/prostate_imrt_matrad_reproduction`.
- **Verified:** the dev-VM env contains **octave 6.4 + python3.10 + pip + numpy + scipy ONLY** — it does NOT
  contain pymedphys/pydicom/numba/matplotlib/scikit-image (the only pydicom on the whole VM is inside Slicer's
  bundled python3.9, unrelated). My `matrad-rtplan` package creates exactly `octave=6.4 numpy scipy` + python →
  **already matches ground truth.** The QA/gamma libs are meant to be agent-installed at solve time via the
  wrapper (`software/run_matrad.sh python -m pip install pymedphys pydicom ...`), consistent with the
  "system software baked, Python libs at solve time" policy. No packaging change needed (earlier audit worry retracted).

### GraphST spatial env
- **Where:** `/home/user/.local/agenthle/graphst-env/` (GraphST 1.1.1 + scanpy 1.10.4 + torch 2.11.0+cpu).
- **Tasks:** likely `life_sciences/spatial_transcriptomics_*` / `pseudotime` family (uses squidpy/scanpy). These
  currently install scanpy/squidpy at solve time via uv, which works, so this per-user env is informational
  (shows the authors' reference stack), not a hard blocker.

---

## 2. Genuinely absent from the dev VM entirely

### InterProScan 5.77-108.0
- **Checked:** `/opt`, `/opt/toolchains` (only `miniforge3`, `r-biostatistics`), `/media` — no `interproscan*` anywhere.
- **Task:** `life_sciences/protein_function_annotation_instance_1`. The provisioning plan claims a Stage-4 prebake
  at `/opt/toolchains/interproscan-5.77-108.0-fresh/`, but it is **gone** (history of disk-exhaustion-corrupted installs).
- **Gap:** ~15 GB tool + DB. The task `software/interproscan.sh` hardcodes a task-local
  `runtime/interproscan-5.77-108.0` path and `install_software.sh` does a ~15 GB EBI FTP download at solve time.
- **Decision needed:** pre-bake the InterProScan runtime (and align the wrapper path) vs. guarantee the 15 GB
  download path + disk. Do not auto-download.

### Neurodesk `brain_science` computer-use bundle + Slicer/FSLeyes `.simg` images
- **Checked:** `/home/user/brain_science` is **empty**; the `run_scene.sh` orchestration the scene wrappers call
  does not exist on the dev VM. (Slicer 5.0.3 and FSLeyes 1.18.1 themselves ARE at `/opt/slicer-5.0.3` / `/opt/fsleyes-1.18.1`.)
- **Tasks:** `psychology_neuro/scene2_resample`, `health_medicine/scene3_skullstrip_qc`. Both `launch_gui.sh`
  scripts exec `/home/user/brain_science/computer_use_benchmark_bundle/run_scene.sh <scene>`.
- **Gap:** the bundle (run_scene orchestration + `containers/*.simg` Neurodesk images, multi-GB) is not on the
  dev VM and not in the task `software/`. My `neurodesk-brain-science` package install.sh correctly hard-blocks
  (exit 3) without a supplied `$BRAIN_SCIENCE_BUNDLE`.
- **Decision needed:** locate/obtain the brain_science bundle + Neurodesk `.simg` images (where is the source of
  truth?). Until then scene2/scene3 cannot be reconstructed into a clean container. These are also GUI/X tasks.

### rtg-tools (RTG `vcfeval`)
- **Checked:** `which rtg` empty; no `*rtg*` under `/opt`.
- **Task:** `life_sciences/WGS_Variant_Calling` (RTG `vcfeval` benchmark = 0.6 of score).
- **Gap:** packageable (rtg-tools is a self-contained Java jar release) — queued as a new package in PROGRESS.md.
  Recorded here because it's absent from the dev VM, so the dev VM is not a usable reference for it.

### sgfmill (Go SGF parser, Python)
- **Task:** `computing_math/go_game_reconstruction_1` grader (`verify_sgf.py`). Not present in any system python.
- **Gap:** trivial pip package, but note the bigger blockers for this task: Sabaki AppImage needs libfuse2 + an
  X/Xvfb desktop on the headless snapshot, plus the documented uniqueness reservation (see TASK_VALIDITY_ISSUES.md).

---

## 3. Container images NOT pre-pulled on the dev VM

`docker images` on the dev VM shows only `openroad/orfs` (4.61 GB). The following are referenced by tasks but
not present, so a no-egress run would fail:

- **`flowable/all-in-one:6.5.0`** — `business_finance/bpmn_*_l3` (×2). (Static grader still scores without it.)
- **ensembl-vep `release_111` + nf-core/sarek 3.8.1 process images** — `life_sciences/hg002_chr22_germline_variant_pipeline`.
  The sarek *pipeline code* is at `/opt/nf-core-sarek-3.8.1` and nextflow at `/opt/nextflow-25.10.4`, but the
  per-process container images (GATK/Picard/VEP/samtools) are not pulled. hg002's no-egress contract requires them pre-pulled.
- **gatk / picard** — only available via the `gatk-picard` package's container path; no `/opt` binary, no pre-pulled image on dev VM.

**Decision needed:** pre-pull (and `docker save`/stage) these images for offline runs, or allow registry egress at solve time.

---

## Summary table

| Item | Tasks | Dev-VM status | Class |
|---|---|---|---|
| LEAP + torch-cpu | ct_geometry, limited_angle_ct | per-user only (`/home/user/.local`) | needs package (heavy build) — decision |
| matRad QA libs (pymedphys…) | prostate_imrt | per-user `rtplan-matrad` env | packaging fix queued |
| InterProScan 5.77 | protein_function | absent | decision (15 GB) |
| brain_science bundle + .simg | scene2, scene3 | absent (`/home/user/brain_science` empty) | decision (source unknown) |
| rtg-tools | WGS_Variant_Calling | absent | package queued |
| sgfmill | go_game | absent | pip (minor) |
| flowable image | bpmn ×2 | not pulled | decision (egress/prebake) |
| vep/sarek/gatk images | hg002 | not pulled | decision (egress/prebake) |
