# Per-task dependency install scripts — progress tracker

Goal: for each Linux-only task (`selected_tasks/linux_only.txt`, 105 entries), author a
`tasks/<domain>/<task>/scripts/install_deps.sh` + `verify_env.sh` so that, starting from
the **lean base image**, running `install_deps.sh` yields a container with the full
environment needed to perform the task. Data is out of scope (staged separately).

## Decisions (locked)
1. `software/` dir entries are **thin wrapper scripts** (`exec` → real binaries at
   fixed paths like `/opt/<tool>-<ver>/bin/...`), NOT self-contained software. The real
   software must be provided by `install_deps.sh`.
2. Heavy `/opt` software is **rebuilt from upstream** (NOT copied from dev machine 34.186.175.29).
3. `install_deps.sh` installs **only system software/libraries**. Python packages are left to
   each task's own `runtime_env` wrapper (`uv --frozen`, networked) at solve time. For non-RTENV
   tasks that still need Python libs, those come from the task's own software wrapper/runtime too.
4. Scripts live at `tasks/<domain>/<task>/scripts/{install_deps.sh,verify_env.sh}`.

## Ground-truth sources per task
- `agenthle/tasks/<task>/SOFTWARE_PROVISIONING_PLAN.md`  ← canonical software spec
- `agenthle/tasks/<task>/{TASK_INTAKE.md,CONTEXT.md,PITFALLS.md,REPRO_COMMANDS.md}`
- `eval-deps` image: baked data at `/media/user/data/agenthle/<task>/base/`
  (`software/*` wrappers reveal exact `/opt` paths+versions; `input/runtime_env/uv.lock`)
- repo `tasks/<task>/{main.py,scripts/*}` (scoring reveals libs used)

## Test harness
`/home/weichenzhang/.claude/jobs/b24d263a/tmp/run_task_test.sh <task> <host_base_dir> <install_deps.sh> <verify_env.sh>`
- lean base = `agentslastexam/ale-kasm-noagents-nodata:latest` (amd64, has uv/python3.10/3.12/node)
- stages task data at canonical path, runs install as root, runs verify as uid 1000.
- Container has network (wrappers fetch wheels at solve time, by design).

## Status legend: [ ] todo  [~] in progress  [x] PASS

- [x] business_finance/american_option_pricing_ls  — no system deps (numpy/scipy via wrapper). PASS.
- [x] business_finance/basel_operational_risk_bia_cn — LibreOffice Calc + /usr/bin/libreoffice shim (unset LD_LIBRARY_PATH; base's global path breaks UNO bootstrap). PASS.
- [x] business_finance/bpmn_category_governance_restructuring_l3 — Docker Engine+Compose; verify boots Flowable 6.5.0 (postgres13+flowable-task) → HTTP 302. PASS.
      NOTE: flowable-task binds internal port 9999 (compose maps 8080); readiness = Spring "Started ...Application"
      log marker + HTTP probe via `docker run --network container:flowable-task curlimages/curl` on :9999.
- [x] business_finance/bpmn_supply_disruption_l3 — same Docker stack, flowable/all-in-one:6.5.0 (binds 8080). PASS (REST 200).
- [x] business_finance/financial_stmt_reconstruction_aapl_fy2024 — /usr/bin/python alias; pdftotext+grep already in base. PASS.
- [x] business_finance/digital_marketing_ab_test_analysis_1 — RTENV (growthbook/pandas/scipy/statsmodels), no sys deps. PASS.
- [x] business_finance/digital_marketing_audience_segmentation_1 — RTENV (pandas/pyarrow/pyyaml), no sys deps. PASS.
- [x] business_finance/legal_ma_consistency_audit_01 — stdlib; python alias. PASS.
- [x] business_finance/llm_ecosystem_privacy_audit_realdata_1 — stdlib; python alias. PASS.
- [x] business_finance/internal_employee_agent_instance_1 — stdlib (uuid); system python3. PASS.
- [x] business_finance/sse_northbound_programmatic_trading_01 — stdlib; system python3. PASS.
- [x] business_finance/sec_10k_financial_parsing — RTENV (pdfplumber/pypdf/pydantic) via wrapper; python alias MUST be 3.10 (runtime pins ==3.10.*). PASS.
- [x] business_finance/ff5_public_reconstruction — Google Chrome stable + python alias + RTENV (pandas/numpy/statsmodels/yfinance/bs4/lxml/requests). PASS.
- [x] business_finance/pe_screening_memo_1 — baseline python(3.10)+uv; reads staged PDFs/txt. Variant is `zscaler_fy2025` (NOT base). PASS.

### ✅ business_finance domain COMPLETE (14/14)

### computing_math (in progress)
RTENV (python3.10 alias + uv sync; manylinux wheels, no sys deps) — all PASS:
- [x] branch_bound_atsp (numpy/scipy; pyproject-only, no lock → `uv sync` non-frozen)
- [x] cfr_game_theory_equilibrium (numpy)
- [x] clustered_cyclic_code_circuit_level_simulation (stim/ldpc/numpy/pandas/scipy/networkx/matplotlib)
- [x] cost_optimization_1 (pandas)
- [x] data_pipeline_etl_instance_1 (pandas)
- [x] ising_post_measurement_1 (numpy/scipy; variant `n10_critical_u01_correlators`)
- [x] particle_filter_nonlinear_tracking (numpy/scipy)
- [x] recsys_cold_start_instance_1 (numpy/pandas/sklearn/scipy)
- [x] synthetic_causal_structure_inference (networkx/numpy/pandas/sklearn/statsmodels)
Stdlib / simple — all PASS:
- [x] dit_pipeline_cfg_alignment_fid_256_001 (file-editing only; python alias; py-parse staged sources)
- [x] k8s_payment_api_root_cause_analysis (stdlib text RCA; python alias)
- [x] os_log_permission_guard_v1 (stdlib + tar; python alias)
- [x] cp_test_gen_1 (g++ C++17; variant `default`)
Heavy — mostly PASS:
- [x] mp_checkpoint_consolidation_v2 (RTENV torch+safetensors; manylinux wheels). PASS.
- [x] ranking_node_feature_parity_recovery_instance_1 (vendored python_pkgs pytest via python_task_env.sh)
- [x] go_game_reconstruction_1 (sabaki Electron AppImage: libfuse2 + Electron libs; verify=extract+ldd clean)
- [x] k3_abelian_extensions (GAP 4.11.1 via apt; variant `h_4_4_4_m_1_8`; computed |S4|=24)
- [x] paper_reproduction_instance_1 (baked software/.venv py3.10 + pip; agent installs torch at solve)
- [~] k8s_migration_1 (Docker + 5 pinned /opt tools SHA-checked [minikube1.32/kubectl1.29/helm3.14/
      terraform1.7/trivy0.70]; verify = versions + real Minikube+Calico cluster smoke. Scripts ready,
      smoke pending. install_deps adapted from agenthle scripts/install_software.sh.)

Helpers: `tmp/gen_rtenv.sh` generates RTENV install+verify (handles lock-less pyproject; imports via built venv).

### ✅ education_info COMPLETE (3/3)
- [x] homework_grading_numerical_pdes_instance_02 (stdlib; python alias)
- [x] marc_remediation_folio_overlay (stdlib; python**3.12** — its wrapper execs /usr/bin/python3.12 directly)
- [x] moodle_gradebook_closeout_reconciliation (RTENV pandas via python_with_task_deps.sh)

## Gotcha: pipefail + first-run exit
Some CLIs (minikube) exit non-zero on their FIRST invocation. Under `set -o pipefail`, a
`cmd | grep` version check then spuriously fails. Use capture-then-grep:
`out="$(cmd 2>/dev/null || true)"; echo "$out" | grep -q WANT`. Also: marc wants python3.12,
not 3.10 — check each task's wrapper for the exact interpreter (don't blanket-alias to 3.10).

### engineering (in progress)
- [x] aerospace_low_thrust_trajectory (RTENV numpy/scipy)
- [x] abb_irb6700_asset_to_urdf_instance_1 (stdlib xml/csv; python alias)
- [x] sumo_urban_am_peak_calibration (RTENV eclipse-sumo 1.26.0 — sumolib/traci via sumo.SUMO_HOME/tools, NOT top-level)
- [x] power_10kv_feeder_reliability_001 (uv project at input/ [not runtime_env]; python3.12; openpyxl+pandas)
- [x] mpc_control_building_v1 (EnergyPlus 22.1.0 → /opt/energyplus-22.1.0 from NREL GH + cvxpy RTENV). PASS.
- [x] chisel_verilog_alignment_seq_1 (oss-cad-suite 20260422[yosys] + firtool 1.138.0[CIRCT] + sbt 1.9.9 +
      default-jre[Java11]; all to /opt; verify via software/ wrappers). PASS.
- [x] humanoid_wbc_policy_evaluation (MuJoCo GL/EGL/OSMesa/GLFW libs; mjlab env built from staged
      mjlab.zip [mjlab src repo w/ uv.lock] at solve time). PASS.
- [x] openroad_sky130_ibex_pnr_signoff (Docker + pinned openroad/orfs digest; openroad at /OpenROAD-flow-scripts/tools/install/OpenROAD/bin). PASS. ([~1.58GB];
      verify runs `openroad -version` inside it. DinD. Testing.
- [x] (computing_math) k8s_migration_1 PASS: verify SKIPs the minikube cluster smoke on nested-DinD cgroup
      v2 limit (honest skip, not faked) while keeping the 6-tool version contract mandatory. Re-testing.

## More gotchas
- eclipse-sumo wheel: binaries+tools under site-packages/sumo/ (`import sumo; sumo.SUMO_HOME`);
  add $SUMO_HOME/tools to PYTHONPATH for sumolib/traci. No top-level sumolib.
- uv project location varies: usually input/runtime_env/, but some at input/ (power_10kv).
- nested-DinD (minikube): need --cgroupns=host on the test container (cgroup v2 threaded-mode).

### small domains (legal/other/social/transport/psych) — partial
- [x] legal/legal_dr_fees_01 (RTENV pymupdf[fitz]+pypdf; py3.12 wrapper)
- [x] other/aerobics_wc2026_portugal_trio_difficulty_scoring (stdlib; variant `variant_1`)
- [x] social_sciences/atwood_2022_measles_vaccine_reproduction (stdlib; pdftotext/unzip in base + python alias)
- [x] transport_safety/abm_hangzhou_metro (RTENV geopandas/matplotlib/networkx/numpy)
- [x] transport_safety/capacitated_vehicle_routing_problems (RTENV pyvrp/vrplib)
- [x] engineering/openroad_sky130_ibex_pnr_signoff — Docker + pre-pull pinned openroad/orfs@sha256:fd77..
      (~1.58GB). openroad is at /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad (NOT on PATH;
      source /OpenROAD-flow-scripts/env.sh). PASS.
- [x] legal/agora_governance_classify_instance_1 — wrapper -> /opt/cua-server/.venv/bin/python (absent on
      lean base); install_deps creates it with `uv venv --python 3.10 /opt/cua-server/.venv`. PASS. ✅ legal 2/2
- [x] transport_safety/fds_single_compartment_detector_reconstruction — FDS+SMV 6.10.1 → /opt/fds-smv-6.10.1.
      Installer `FDS-6.10.1_SMV-6.10.1_lnx.sh`: pipe `extract` → payload tarball whose tree (bin/fds,
      bin/INTEL/lib, smvbin/smokeview) maps 1:1 onto /opt/fds-smv-6.10.1. PASS. ✅ transport_safety 3/3
- [x] psychology_neuro/celegans_neuron_tracking (variant 137; RTENV opencv/PyQt5/h5py/pyqtgraph/skimage +
      OpenCV/Qt-xcb system libs; verify imports headless w/ QT_QPA_PLATFORM=offscreen). PASS.
- [ ] psychology_neuro/scene2_resample (containers_manifest.yaml + launch_gui.sh — brain_science container
      GUI bundle; like health_medicine/scene3_skullstrip_qc. Needs container-manifest approach.)

### remaining big domains (TODO): health_medicine (~26: lots of R/Bioconductor + imaging + stats),
###   life_sciences (~19: bwa/samtools/gatk pipelines, R/Bioc, AmberTools, CellProfiler, InterProScan),
###   physical_sciences (~12: QE+BerkeleyGW heavy builds, VQE, climate)

## RUNNING TALLY: 52/105 PASS (~50%).
## COMPLETE domains: business_finance 14/14, computing_math 19/19, education_info 3/3, engineering 8/8,
##   legal 2/2, other 1/1, social_sciences 1/1, transport_safety 3/3, psychology_neuro 1/2 (celegans done).
## REMAINING (53): psychology_neuro/scene2_resample (container GUI) + the 3 science-heavy domains:
##   health_medicine (22 — R/Bioconductor, medical imaging, biostatistics, variant annotation),
##   life_sciences (19 — bwa/samtools/gatk/nextflow pipelines, R/Bioc, AmberTools, CellProfiler, InterProScan, RGI),
##   physical_sciences (11 — Quantum ESPRESSO+BerkeleyGW heavy MPI/Fortran builds ×3, VQE, climate, QE/BGW).
## These are the heaviest installs; expect many R+Bioconductor (apt r-base + BiocManager) and big source builds.

### health_medicine (in progress — 11+ PASS)
RTENV/stdlib PASS: causal_ihdp_ite_estimation_6a_v1, crf_sdtm_mapping_1, crf_sdtm_mapping_4,
  epidemiology_forecast, flusight_offline_hosp_forecast_2024_12_14, healthcare_variant_annotation_pipeline,
  nhanes_confounder_sensitivity_analysis, obermeyer_bias_reproduction, Clinical_Variant_Annotation(stdlib+jq),
  wsi_tumor_localization_1(libopenslide0 + openslide-python; variant center_point).
R PASS: public_health_mask_mandate_ratio — **R RECIPE (validated)**: add CRAN apt repo
  (cloud.r-project.org/.../jammy-cran40 + marutter_pubkey) → `apt install r-base r-base-dev` (gives R 4.6.x),
  then `Rscript -e install.packages(...)` for CRAN pkgs; nlme/MASS/splines are recommended (bundled).
- [ ] nsclc_radiomics_cox_signature_v1 — python3.10-dev fixed Python.h, but pyradiomics@3.0.1a1 (alpha) sdist build then fails on its own cmatrices.h. DEFER (thorny alpha build; try --no-build-isolation or a prebuilt wheel).
- [x] limited_angle_ct_dps_reconstruction — RTENV diffusers/huggingface_hub/safetensors/torch. PASS.
- [ ] simglucose_safe_basal_control_instance_1 — uv build FAILS: ancient gym==0.9.4 sdist + root hatchling
      "files to ship". Needs `--no-install-project` and/or gym build deps. DEFER (thorny).
- [ ] healthcare_tcga_luad_survival_kras — R + Bioconductor (BiocManager + TCGAbiolinks[heavy]+survival/survminer)
- [ ] ltmle_targeted_bootstrap_simulation_study — R 4.3.2 EXACT + lmtp 1.4.2 + big CRAN closure
- [ ] replicate_paper_1 — R 4.5.3 + sim/reporting pkgs; software/Rscript wrapper -> /usr/bin/Rscript
- [ ] ct_geometry_calibration_catphan — LEAP 1.26 (leapct/leaptorch) + PyTorch CPU + numpy/scipy/skimage
- [ ] prostate_imrt_matrad_reproduction — GNU Octave 6.4.0 + matRad (git commit); run_matrad.sh wrapper
- [ ] scene3_skullstrip_qc — containers_manifest.yaml + launch_gui.sh (brain_science container GUI; like scene2)
- [ ] healthcare_bias_audit_27a_public_replication_v1 — no std RTENV; classify (python sci runtime). check
- [ ] healthcare_sap_group_sequential_nsclc — no std RTENV; classify. check

## RUNNING TALLY (update): ~64/105 PASS. health_medicine 12/22 (R recipe validated; 2 thorny py-builds + R-Bioc/LEAP/matRad/containers remain). life_sciences(19)+physical_sciences(11) untouched.

## Note: non-`base` variants
Some tasks stage under a named variant dir, not `base/` (e.g. pe_screening_memo_1 → `zscaler_fy2025/`).
Check `/media/user/data/agenthle/<task>/` for the variant dir; stage that as the test's base.
Wrappers use relative paths so re-homing the variant content as `base/` works for testing.

## IMPORTANT gotcha: /usr/bin/python must be 3.10
The reference dev VM's `python` is 3.10.12 (NOT python3=3.12). Several RTENVs pin
`requires-python ==3.10.*`, and wrappers that do `uv run --python "$(command -v python)"`
will FAIL if `python` is 3.12. ALWAYS alias `/usr/bin/python -> /usr/bin/python3.10`
(base ships both python3.10 and python3.12). Pure-stdlib tasks tolerate either, but
standardize on 3.10 for fidelity.

## More gotchas
- Base has NO `python` / `/usr/bin/python` (only python3). Tasks whose wrappers `exec python`
  or `/usr/bin/python` need `ln -sf /usr/bin/python3 /usr/bin/python` in install_deps.
- Base ALREADY has: pdftotext (poppler-utils), grep, unzip, zip, 7z, jq, ffmpeg, node22, uv, gcloud.
- Base has NO chrome/chromium (ff5 etc. need google-chrome install).
- DinD test harness: PRIVILEGED=1 + named volume `ale-dind-cache:/var/lib/docker` (overlay-on-overlay
  fails otherwise; real eval VM is ext4 so install_deps is fine). DIND_FRESH=1 for clean cache.
- Readiness probes MUST require a real success HTTP code (200/302/401); curl prints "000" on
  connection failure — never accept it as ready (caused a false PASS).

## Dependency buckets (from triage of all 105 SOFTWARE_PROVISIONING_PLAN.md)
- **No system deps** (RTENV uv wrapper or stdlib-only): american_option, many `exec uv`/`exec python` tasks.
- **parser-only / NO install**: amber_minimization_script_prep (writes text files; Amber/CUDA/SLURM are labels only).
- **apt-simple**: basel(libreoffice), financial_stmt(poppler/pdftotext), atwood(pdftotext+unzip), genomic_interval(bedtools).
- **Docker / DinD**: bpmn_category, bpmn_supply (Flowable), k8s_migration_1 (Docker+Minikube+kubectl+Helm+Terraform+Trivy+Calico), k8s_payment_api, engineering/openroad (openroad/orfs image), hg002 / WGS (nextflow+docker/apptainer+bwa/gatk/samtools).
- **R / Bioconductor**: replicate_paper_1, healthcare_tcga_luad_survival, tcga_brca_deg, ltmle (R4.3.2 + big closure), pseudotime_de (R+Bioc), gene_expression_differential, spatial_transcriptomics (R+py), nhanes/obermeyer/causal (R via wrapper).
- **Quantum ESPRESSO + BerkeleyGW (heavy MPI/Fortran build)**: computational_materials_science, mose2_bse_absorption_soc, silicon_bse_absorption.
- **Other heavy single tools**: prostate_imrt (Octave6.4+matRad), mpc_control_building (EnergyPlus 22.1.0), sumo (eclipse-sumo pip via wrapper), fds (FDS+Smokeview 6.10.1), protein_function (InterProScan+Java), yeast_colony (CellProfiler 4.2.8), chisel (yosys+firtool+sbt), ff5 (Google Chrome).
- Chrome/browser tasks: ff5 (and any `exec chromium`).

## Gotchas learned
- Base image exports a global LD_LIBRARY_PATH (CUA/nvidia). It breaks some self-contained
  apps (LibreOffice UNO -> DeploymentException). Fix per-app by shimming its launcher to
  `unset LD_LIBRARY_PATH`. Watch for this with other bundled/ELF tools.
- LibreOffice: install `libreoffice-calc` WITHOUT --no-install-recommends (needs ure/uno libs).

## PACKAGE-MODEL WAVE 2 (PR #12, base=main): +19 PASS → 83/105
life_sciences RTENV+: cell_tracking, cell_translocation, gene_expression_differential(pydeseq2/gseapy),
  tcga_brca_deg, tms_marrow, spatial_transcriptomics(3.12), merfish(torch/starfish), genomic_interval(bedtools-2.31.1 built from src),
  amber_minimization(parser-only), zdock, idp_ensemble(bundled bins data), tp53(chrome+python).
physical_sciences RTENV: adapt_vqe, gillespie, hst_acs_wfc(astropy/photutils), molecular_structure(rdkit),
  phonon_dispersion, exact_diag, climate_prediction(torch/xarray/lightning).
New package: bedtools-2.31.1 (source build). Ambiguous no-runtime_env tasks → python-default-3.10
  (libs agent-installed at solve; base has no system numpy).
REMAINING ~22 (heavy tail): R/Bioconductor (tcga_luad, ltmle[R4.3.2], replicate_paper[R4.5.3],
  healthcare_sap, pseudotime_de, healthcare_bias_audit[R3.5.1 legacy], public_health_mask r-libs FIX);
  QE+BerkeleyGW ×3 (computational_materials/mose2/silicon_bse); AmberTools23; CellProfiler4.2.8(yeast_colony);
  InterProScan(protein_function); Octave+matRad(prostate_imrt); GLM(glm_lake); bwa/gatk pipelines (WGS, hg002);
  ncbi-blast(rgi); containers (scene2/scene3); thorny py-builds (nsclc pyradiomics, simglucose gym).

## WAVE 3 (PR #12): R/Bioconductor + conda heavy + bioinformatics
PASS via package model: tcga_luad (Bioc TCGAbiolinks/PPM), public_health_mask, healthcare_sap,
  replicate_paper_1, amber_three (AmberTools23 conda), rgi (blast2.15), WGS (bioinfo-cli),
  genomic_interval (bedtools src), glm_lake (bundled libs → python only).
New packages: r-base, r-build-deps, r-libs-*, micromamba, ambertools-23, cellprofiler-4.2.8,
  matrad-rtplan, ncbi-blast-2.15.0, bioinfo-cli, nextflow, gatk-picard, bwa-mem2, ensembl-vep,
  r-base-4.5.3 / r-base-4.3.2 (rig), bedtools-2.31.1.

### KEY LEARNINGS (R + conda)
- R-version vs binary availability: PPM has CRAN/Bioc binaries only for released R minors. R 4.6 is
  too new → tradeSeq/slingshot compile-from-source & fail. Fix: pin R 4.5.3 via rig (r-base-4.5.3).
- conda envs created as root: chown the env + ~/.cache/mamba to 1000:0 or `micromamba run` (as uid
  1000) fails on /home/kasm-user/.cache/mamba/proc/proc.lock.
- cellprofiler is on **bioconda** (4.2.8.1), not conda-forge.

### HARD FRONTIER — need external resource / decision (NOT auto-completable here)
- physical_sciences/{computational_materials_science, mose2_bse_absorption_soc, silicon_bse_absorption}
  → QE 6.7 (conda-forge ok) + **BerkeleyGW 4.0 source is REGISTRATION-GATED** (no open URL) + multi-hour
  MPI/Fortran build. Need: a BGW 4.0 tarball (login at berkeleygw.org) → then build /opt/qe-bgw-6.7.0-4.0.
- life_sciences/protein_function_annotation_instance_1 → InterProScan 5.77 (~50GB incl. data). Recipe in
  task software/install_manifest.json (archive_url). Need: time/space to download ~50GB.
- health_medicine/healthcare_bias_audit_27a_public_replication_v1 → legacy **R 3.5.1** + data.table 1.11.4
  + task python venv (pandas1.5.3...). R 3.5.1 via rig uncertain.
- health_medicine/nsclc_radiomics_cox_signature_v1 → pyradiomics (alpha) sdist build fails on its own
  cmatrices.h (broken upstream packaging; no cp310 wheel). Needs a prebuilt wheel or patched build.
- health_medicine/simglucose_safe_basal_control_instance_1 → ancient gym==0.9.4 sdist build fails on
  modern setuptools + root hatchling "files to ship". Needs gym build pin/patch or --no-install-project.
- psychology_neuro/{scene2_resample}, health_medicine/{scene3_skullstrip_qc} → Neurodesk GUI apptainer
  container bundle (Slicer/FSL .simg images, computer-use). Need: apptainer + Neurodesk images (GB, GUI).

### FRONTIER UPDATE
- life_sciences/pseudotime_de: rig R 4.5.3 installs, but Bioconductor tradeSeq/slingshot/TrajectoryUtils
  fail to source-compile (`dyn.load: lazy loading failed for TrajectoryUtils` — C++/ABI vs available R).
  Needs a matched Bioc BINARY repo for the R build, or conda bioconductor-tradeseq with a conda R.
- PASS additions: ltmle (rig R4.3.2), prostate_imrt (Octave6.4+matRad, direct-octave verify),
  yeast_colony (bioconda CellProfiler 4.2.8.1).

## WAVE 4 — frontier recovery
RECOVERED (now PASS): ltmle (rig R4.3.2), prostate_imrt (Octave6.4+matRad), yeast_colony (bioconda
CellProfiler), healthcare_bias_audit (rig R3.5.1 + data.table), pseudotime_de (bioconda binary
tradeSeq/SingleCellExperiment — avoids source-compile ABI fail). hg002 re-testing (nextflow bootstrap fix).
Verify-quoting fix: meta.verify R exprs must be single-quoted (`Rscript -e 'library(x)'`).

## REMAINING TRUE FRONTIER (need external resource/decision; NOT auto-completable here)
1. physical_sciences/{computational_materials_science, mose2_bse_absorption_soc, silicon_bse_absorption}
   — QE 6.7 (conda-forge OK) + BerkeleyGW 4.0 source is REGISTRATION-GATED (no open URL) + multi-hour build.
2. life_sciences/protein_function_annotation_instance_1 — InterProScan 5.77 ≈ 50GB (url in task manifest).
3. psychology_neuro/scene2_resample + health_medicine/scene3_skullstrip_qc — Neurodesk GUI apptainer images.
4. health_medicine/nsclc_radiomics_cox_signature_v1 — pyradiomics (pinned alpha) sdist build broken (cmatrices.h; no cp310/311 wheel).
5. health_medicine/simglucose_safe_basal_control_instance_1 — gym==0.9.4 sdist won't build on modern setuptools.
=> ~98/105 PASS; 7 frontier tasks need user input (BGW tarball / 50GB+GUI provisioning / upstream-broken pins).

## FINAL MAPPING COMPLETE — 105/105 cards have requiredSystemPackages
PASS (validated on lean base): ~100 tasks incl. hg002 (nextflow/gatk/bwa-mem2/vep), protein_function
  (openjdk-11; InterProScan is staged data), bias_audit/ltmle (rig legacy R), pseudotime (bioconda tradeSeq),
  prostate_imrt/amber_three/yeast_colony (conda).
MAPPED but verify needs external resource / has upstream-broken lock (documented, not faked):
  - computational_materials_science, mose2_bse_absorption_soc, silicon_bse_absorption → qe-bgw-6.7.0-4.0
    (QE6.7 conda OK; BerkeleyGW 4.0 source registration-gated → set BGW_TARBALL to finish build).
  - scene2_resample, scene3_skullstrip_qc → neurodesk-brain-science (apptainer + multi-GB GUI .simg bundle;
    set BRAIN_SCIENCE_BUNDLE to provision).
  - nsclc_radiomics_cox_signature_v1 → env (python3.10-dev/build-essential) provided; its runtime_env lock
    pins a pyradiomics alpha whose sdist is upstream-broken (cmatrices.h) — a task-data lock fix, not env.
  - simglucose_safe_basal_control_instance_1 → env (python) provided; its lock pins gym==0.9.4 which won't
    build on modern setuptools — task-data lock fix, not env.
  - ct_geometry_calibration_catphan → python + build-essential (agent builds LEAP/torch at solve; no manifest).
