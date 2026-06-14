# Task Input Validity Issues (Linux-only ALE tasks)

Findings from a deep per-task audit (105 Linux-only tasks) of task descriptions, `task_card.json`,
graders, staged `input/`, and the data `software/` dirs, cross-checked against the dev VM
(`dev-ubuntu22`) and the `gs://ale-data-public` bucket.

This file = **deliverable 3**: tasks whose *input is self-contradictory, under-specified, or
ships a broken/impossible pin*. It is descriptive (cause → effect → suggested fix). It does NOT
cover pure packaging work (see `PROGRESS.md`) or dev-VM-missing software (see `DEV_VM_MISSING_ENV.md`).

Severity legend:
- **BLOCKING** — task cannot be completed/scored as shipped.
- **DEGRADED** — partially solvable; some checkpoints unreachable.
- **MAPPING** — env mapping/staging mismatch; I fix this in `packages/` + `task_card.json` (cross-ref PROGRESS.md).
- **COSMETIC** — misleading field/doc; no functional impact.

---

## A. Genuinely broken / impossible inputs

### health_medicine/nsclc_radiomics_cox_signature_v1 — BLOCKING (task-data)
- **Cause:** `input/runtime_env/pyproject.toml` pins `pyradiomics==3.1.0` with `requires-python>=3.10,<3.12`.
  PyPI has wheels for pyradiomics 3.1.0 only up to **cp39**; for py3.10/3.11 pip falls back to the
  3.1.0 sdist, which self-identifies as `3.0.1a1` and is **missing its C headers (`cmatrices.h`)** → compile fails.
- **Effect:** `uv sync` / `pip install` of the shipped manifest fails → no working radiomics env.
  Verified on lean base AND confirmed dev VM has no pyradiomics anywhere (not in /opt, /home/user, conda).
- **Fix (task-data, one line — VERIFIED):** in `input/runtime_env/pyproject.toml` change
  `requires-python = ">=3.10,<3.12"` → `">=3.9,<3.10"`. pyradiomics 3.1.0 ships a prebuilt **cp39 wheel**, so on
  python 3.9 `uv sync` installs it (and SimpleITK/lifelines/pandas/numpy/sklearn/scipy, all of which have cp39
  wheels) with NO compile. Tested in the lean base: `uv sync --python 3.9` → `pyradiomics v3.1.0` imports incl.
  `featureextractor`. (Alternative: `pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics@v3.1.0` on 3.10
  — also works but needs build tools; the 3.9 repin is simpler.)
- **Not an env-package issue** — `python-default-3.10` + `build-essential` + `python3.10-dev` are correct and present.

### computing_math/clustered_cyclic_code_circuit_level_simulation — BLOCKING (offline)
- **Cause:** runtime_env depends on `quits @ git+https://github.com/mkangquantum/quits.git@v1.0.0`
  (no PyPI wheel, no vendored copy, no commit SHA pin) plus `stim`, `ldpc`.
- **Effect:** building the runtime requires outbound git+https to GitHub at solve time. A network-isolated
  VM cannot construct the environment at all.
- **Fix:** vendor the QUITS source into the task data (pin a SHA), or confirm git egress is allowed at solve time.

### education_info/marc_remediation_folio_overlay — MAPPING/BLOCKING
- **Cause:** `requiredSystemPackages` is empty `[]`, but `software/python3.12` hard-execs `/usr/bin/python3.12`.
- **Effect:** on a base without python3.12 the wrapper fails ("python3.12 not found") → CLI never runs → score 0.
- **Fix:** map `python-default-3.12` (done in card update). Stdlib-only otherwise, so this fully resolves it.

### physical_sciences/silicon_bse_absorption — BLOCKING (staging)
- **Cause:** `task_card` directs the agent to run `base/software/bin/{pw.x,…,inteqp.cplx.x}`, but the bucket
  only stages `software/bootstrap_remote_software.sh` — the `/opt`-backed `bin/*` wrappers (present for the
  sibling `mose2_bse_absorption_soc`) are **absent**.
- **Effect:** an agent following the prompt finds no QE/BGW binaries; bootstrap's seed path may not exist → forced from-source rebuild.
- **Fix (task-data):** stage the mose2-style `software/bin/*` wrappers + `inteqp.cplx.x` (the qe-bgw-6.7.0-4.0
  runtime itself IS on the dev VM /opt, so this is a shim/staging gap, not a missing binary).

### physical_sciences/exact_diag_heisenberg_j1j2 — DEGRADED (staging)
- **Cause:** card promises `base/software/python` "resolving to /usr/bin/python3.10 with NumPy/SciPy", but no
  `software/` dir is staged in the bucket for this task.
- **Effect:** agent invokes a missing wrapper; must fall back to system python + self-install numpy/scipy.
- **Fix (task-data):** stage the wrapper, or correct the prompt to say "use system python + uv". Content (problem_spec) is fine.

---

## B. Network / external-service dependencies — NOT a blocker (sandboxes are networked by default)

UPDATE: ALE sandboxes have outbound network by default, so install scripts may fetch and these tasks CAN
complete — there is no env/install problem here. This section is retained only as a reference of which tasks
reach external services, and to flag the TWO with a real *determinism* caveat (live data can drift vs the
hidden reference) — a task-design concern, not an environment one:
  - **healthcare_tcga_luad_survival_kras** — pulls live data from GDC (api.gdc.cancer.gov); results can drift.
  - **ff5_public_reconstruction** — yfinance/FRED with a time-sensitive reference window.
Everything else below just needs the (default-on) network. No action required from an env standpoint.

| Task | External dependency | Effect if offline |
|---|---|---|
| health_medicine/Clinical_Variant_Annotation | gnomAD GraphQL + Ensembl VEP REST + NCBI E-utilities | 4/5 checkpoints unreachable; only the count==200 check is offline. Also GRCh38 build-mismatch risk. Non-deterministic even online. |
| health_medicine/healthcare_tcga_luad_survival_kras | api.gdc.cancer.gov (TCGA data NOT staged) | unsolvable offline; live-data drift vs hidden reference. |
| business_finance/ff5_public_reconstruction | yfinance / FRED / allowlisted financial hosts | inherently online; time-sensitive reference window (canonical run scored 0.0 on coverage). |
| life_sciences/gene_expression_…_functional_enrichment_analysis_1 | Enrichr (maayanlab.cloud) for KEGG_2021_Human | the 2 enrichment TSVs (0.15 weight) cannot be produced offline. |
| business_finance/bpmn_category_governance_restructuring_l3 | Docker Hub pull of `flowable/all-in-one:6.5.0` | prompted live-deploy step impossible (but the static BPMN-XML grader does not re-run Flowable, so core score survives). |
| business_finance/bpmn_supply_disruption_l3 | same Flowable image | same as above. |
| life_sciences/merfish_image_decoding_segmentation_1 | PyPI + Cellpose pretrained model download + uv-managed py3.11 | env install + segmentation fail offline unless caches pre-warmed. |
| life_sciences/rgi_mcr1_colistin_v2 | PyPI/GitHub for official RGI ~6.0.7 | RGI install fails offline (BLAST + CARD DB are staged/prebaked OK). |
| Solve-time PyPI (soft) | many uv-based tasks fetch wheels at solve time | see PROGRESS.md "uv + PyPI base contract"; offline → unsolvable unless uv cache vendored. |

**Recommendation:** decide a per-task network policy (allow egress for the API tasks; pre-warm/vendor caches
for the PyPI/Docker-Hub tasks). Do not attempt to "fix" these via env packages.

---

## C. Prompt ↔ reality mismatches (mapping/staging — I fix in cards/packages)

### computing_math/cfr_game_theory_equilibrium — MAPPING
- Card maps `python-default-3.10`, but both `software/` wrappers `exec uv … --python python3.12`.
  → add `python-default-3.12`. (Agent could bypass with 3.10, so DEGRADED not BLOCKING.)

### computing_math/synthetic_causal_structure_inference — MAPPING
- Card maps 3.10; runtime pins `requires-python>=3.11,<3.14` + numpy≥2.2. Wrapper runs `uv run` with no
  `--python`, so uv must provision a 3.11+ interpreter at solve time. → add `python-default-3.12`
  (or rely on uv interpreter download; document the network need).

### legal/legal_dr_fees_01 — MAPPING
- `software/python3.12` execs `/usr/bin/python3.12` but card maps `python-default-3.10`. → add `python-default-3.12`.

### life_sciences/tms_marrow_cell_type_annotation_instance_1 — MAPPING (minor)
- README mandates `/usr/bin/python3.12`; card maps `python-default-3.10`. Cosmetic under uv, but remap to 3.12 for correctness.

### computing_math/cost_optimization_1 — DEGRADED
- `software/python_cost_optimization.sh` hard-errors unless a Stage-4 `.runtime/python-3.10.12-pandas-2.3.3/`
  tree + `scripts/install_software.sh` exist; neither is in the public bucket. Agent-recoverable via pip/uv.
  Also: 3 of the findings are only present in a PNG dashboard → require image/vision reading (agent-capability, not env).

### computing_math/branch_bound_atsp — DEGRADED (minor)
- `software/python` advertises a `software/.venv` (numpy/scipy) that is not shipped → silent fallback to bare
  system python (no numpy/scipy). Agent-recoverable via uv/pip.

### computing_math/ranking_node_feature_parity_recovery_instance_1 — MAPPING
- Grader hard-requires `uv` AND **passwordless sudo** on PATH (raises otherwise); neither is declared in
  `requiredSystemPackages`. Card also has no `inputFiles` + a `repoRecoveryNote`. Functionally OK if the base
  contract (python3 + uv + passwordless sudo) holds.

### life_sciences/WGS_Variant_Calling — DEGRADED
- Prompt says `conda activate wf1-env`, but tools are installed on PATH (no such conda env) → that command fails.
- The RTG `vcfeval` benchmark checkpoint (0.6 of the score) needs `rtg-tools`, which is not provisioned
  (see DEV_VM_MISSING_ENV.md). Reaches ~0.4 with PATH tools + manual bcftools metrics; full 1.0 needs rtg or a hand-rolled metric.

### health_medicine/scene3_skullstrip_qc — DEGRADED (prompt inconsistency)
- Prompt names "3D Slicer 5.0.3"; `agentMustDo` + `launch_gui.sh` target **FSLeyes**. Inconsistent tool.
  Also a GUI computer-use task that needs a desktop/X session on an otherwise headless snapshot.
  (Bundle availability is a separate dev-VM-missing issue.)

---

## D. Cosmetic / non-blocking

- **business_finance/digital_marketing_ab_test_analysis_1** — card `software` lists "GrowthBook"; never used by inputs/steps/grader.
- **life_sciences/cell_translocation_analysis** — card overstates "CellProfiler" as required; the provisioning plan says scoring does not require it (scikit-image stack suffices).
- **business_finance/pe_screening_memo_1** — prompt references a `software/README.txt` not staged; Constraints list skips item "3". No functional impact.
- **engineering/abb_irb6700_asset_to_urdf_instance_1** — docs mention a nonexistent `gold_fk_samples.json`; the scorer reads the present `reference/joint_manifest.json`, so FK check is intact.
- **engineering/humanoid_wbc_policy_evaluation** — 0.3 "evidence" sub-score is gameable keyword-matching; the 0.7 verdict is the real signal.
- **health_medicine/crf_sdtm_mapping_1 & _4** — optional `input/runtime_env` pins py`>=3.11,<3.13` with no wheelhouse vs the 3.10 runtime; harmless (deps optional, grader is non-executing stdlib CSV compare).
- **life_sciences/idp_ensemble_scoring** — data lives under bucket variant `default/` (not `base/`); exact pins `biopython==1.74` + `scikit-learn==0.22` are load-bearing and the ~7GB model bundle + precompiled bins (need chmod +x, x86_64) must stage intact, else accuracy degrades.
- **computing_math/go_game_reconstruction_1** — documented uniqueness reservation: final board image + 5 opening moves may not uniquely determine the single hidden 168-move trajectory (checkpoints may be unreachable). Separately: GUI-only on a headless snapshot + grader needs sgfmill (see PROGRESS.md / DEV_VM_MISSING_ENV.md).
- **computing_math/dit_pipeline_cfg_alignment_fid_256_001** — grader imports torch+numpy not in the mapped env; agent task itself is fine (see DEV_VM_MISSING_ENV.md for torch).
