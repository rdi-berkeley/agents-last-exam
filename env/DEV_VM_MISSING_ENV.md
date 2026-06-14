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

### LEAP (LLNL CT projector) + CPU PyTorch stack — FEASIBLE on CPU; package is modest (not a heavy build)
- **Where on dev VM:** user `user`'s `pip install --user` site — `/home/user/.local/lib/python3.10/site-packages/`
  has the FULL CT stack: `leapct-1.26`, `leapctype.py`, `leaptorch.py`, `libleapct.so`, `torch` (2.11.0+cpu),
  numpy 2.2.6, scipy 1.15.3, scikit-image 0.25.2, imageio 2.37.3. (This is the standard per-user pip location
  tied to system python3.10 — NOT a separate interpreter.)
- **CPU verified:** `ldd libleapct.so` has NO CUDA/nvidia linkage (it's a CPU build), and a smoke
  (`import torch, leaptorch; from leaptorch import Projector`) prints `OK 2.11.0+cpu`. LLNL's README states the
  projectors are implemented for "multi-GPU and multi-core CPU", so CPU is a first-class mode.
- **Feasibility:** both task machine types are CPU-only — `ct_geometry_calibration_catphan` = c4-standard-4,
  `limited_angle_ct_dps_reconstruction` = c4-standard-16. The reference fixtures were generated with this same
  CPU stack (torch +cpu). So CPU is the *intended* mode, not a degradation — the tasks are feasible.
- **Packaging is NOT a heavy from-source build:** LLNL/LEAP publishes a **prebuilt `libleapct.so`** as a release
  asset for each tag (incl. v1.26). So a `leap-cpu-torch` package = fetch the v1.26 `libleapct.so` + the pure-Python
  wrappers (leapctype.py/leaptorch.py/etc. from the v1.26 tag) into site-packages, + pip install torch-cpu +
  numpy/scipy/scikit-image/imageio. Modest. (The tasks' wrappers `exec /usr/bin/python` and assume these are
  importable, so they must be on the system python path, not in a venv.)
- **RESOLVED (vendored, approved):** there is NO installable CPU build from upstream — the release `.so` is a
  CUDA build (needs libcufft.so.11); the supported pip/`setup.py`→`etc/build.sh` path runs cmake with a
  `CMakeLists` that `find_package(CUDA 11.7 REQUIRED)`; and `cpu_CMakeLists.txt` is broken (64 ungated GPU
  symbols in filtered_backprojection.cpp). So the `leap-cpu-torch` package **vendors the reference CPU
  `libleapct.so` (LEAP 1.26, MIT) + pure-Python wrappers** in `env/packages/leap-cpu-torch/vendor/` and pip-installs
  torch-cpu/numpy/scipy/scikit-image/imageio. Verified clean on the lean base (ldd no CUDA; CPU forward projection
  computes). (An alternative "official-downloads-only" route — release CUDA `.so` + PyPI `nvidia-cufft-cu12`, run
  `use_gpu=False` — was tested and computes, but spams `cudaMalloc failed` and runs the CUDA build's fallback path,
  not the reference pure-CPU build; rejected for fidelity.)

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

### InterProScan 5.77-108.0 — NOT a gap (solve-time download by design; verified)
- **Re-examined the actual task files (not the provisioning doc):** the task is SELF-CONSISTENT.
  `software/install_software.sh` downloads `interproscan-5.77-108.0-64-bit.tar.gz` from the EBI FTP
  (`https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/5.77-108.0/...`) and extracts to `${TASK_ROOT}/runtime/
  interproscan-5.77-108.0`; the wrapper `software/interproscan.sh` reads from that SAME path; and the task PROMPT
  explicitly instructs: "Install InterProScan by running base/software/install_software.sh (~15 GB download,
  requires Java 11)". The `/opt/toolchains/...-fresh/` path was only in a provisioning *doc*, not the task.
- **So the design is:** the agent installs InterProScan itself at solve time (network is on by default). The only
  baked dependency is **Java 11**, which is already packaged + mapped (`openjdk-11`). The earlier "decision" is
  retracted — nothing to provision; just ensure the solve env has Java 11 (it does) and enough disk for ~15 GB.

### Neurodesk `brain_science` computer-use bundle + Slicer/FSL/Workbench `.simg` images — SOURCE FOUND
- **Source of truth (found):** GCP snapshot `agenthle-ubuntu-brain-science-0204` in project **sunblaze-4**.
  Verified by creating a disk from it and mounting it on a probe VM. The bundle is at
  **`/home/user/Desktop/brain_science/computer_use_benchmark_bundle/`** (NOT `/home/user/brain_science` — the
  scene wrappers expect the latter, so deployment must move/symlink it, OR the wrapper path needs fixing).
- **Bundle contents (18 GB total):** `run_scene.sh`, `smoke_test.sh`, `RUNME.md`, `bundle_manifest.json`,
  `containers_manifest.yaml`, `task/{input,output}` (5 scene inputs), and `containers/`:
  - `slicer_5.0.3_20221025.simg` (8.1 GB) — scenes scene1_roi_stats, **scene2_resample** (`Slicer`)
  - `fsl_6.0.7.18_20250928.simg` (6.8 GB) — **scene3_skullstrip_qc**, scene5 (`fsleyes`)
  - `connectomeworkbench_2.1.0_20251212.simg` (2.4 GB) — scene4 (`wb_view`)
- **Confirms the scene3 "Slicer vs FSLeyes" inconsistency:** per RUNME, scene3 uses **fsl/fsleyes** (the launcher
  is right; the card text "3D Slicer" is wrong).
- **It's a portable Apptainer bundle** (`run_scene.sh <scene>` runs the GUI in its `.simg`). Reconstruction into a
  clean container = install apptainer (system, easy) + stage this 18 GB bundle at the wrapper-expected path +
  provide a GUI display (X11/VNC — these are computer-use GUI tasks).
- **RESOLVED (staged into task data, both buckets):** per-task split done. Each scene's `.simg` + bundle scripts
  + scene inputs are now staged under `<task>/base/software/computer_use_benchmark_bundle/` in BOTH
  `ale-data-all` and `ale-data-public`:
  - `scene2_resample` (psychology_neuro): `containers/slicer_5.0.3_20221025.simg` (8.1G) + run_scene.sh +
    smoke_test.sh + containers_manifest.yaml + `task/input/scene2_resample/`.
  - `scene3_skullstrip_qc` (health_medicine): `containers/fsl_6.0.7.18_20250928.simg` (6.8G) + same scripts +
    `task/input/scene3_skullstrip_qc/`.
  Each task's `software/launch_gui.sh` was patched to resolve the bundle relative to itself (no more hardcoded
  `/home/user/brain_science/...`). System software = `apptainer-1.3.0` (packaged + mapped). Integration-tested:
  apptainer + patched launch_gui.sh + run_scene.sh produce the correct `apptainer exec … fsleyes …` command from
  the staged `.simg` + inputs (GUI display is the computer-use harness's job). Source: snapshot
  `agenthle-ubuntu-brain-science-0204` (sunblaze-4). scene3 tool name also corrected (3D Slicer → FSLeyes).

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
| LEAP + torch-cpu | ct_geometry, limited_angle_ct | per-user (`/home/user/.local`), CPU build, works | **package is modest** (prebuilt .so) — recommend build; CPU is intended mode |
| matRad QA libs (pymedphys…) | prostate_imrt | per-user `rtplan-matrad` env (octave+py+numpy+scipy only) | NO gap — matches package; QA libs are solve-time pip |
| InterProScan 5.77 | protein_function | absent | NO gap — solve-time download by design (15 GB); only needs Java 11 (mapped) |
| brain_science bundle + .simg | scene2, scene3 | RESOLVED — staged per-task into both buckets (slicer→scene2, fsl→scene3) + launch_gui.sh patched + apptainer-1.3.0 mapped; integration-tested | done |
| rtg-tools | WGS_Variant_Calling | absent | DONE — package added |
| sgfmill | go_game | absent | DONE — package added |
| flowable / vep / sarek / gatk images | bpmn ×2, hg002 | not pre-pulled | NO blocker (network on by default; agent/nextflow pulls). Optional: pre-pull for speed |

Net: of the original "gaps", only TWO need a real decision — **LEAP** (build the modest CPU package? recommended yes)
and **brain_science** (where to host the 18 GB bundle). The rest dissolved on closer inspection.
