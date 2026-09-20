# LTMLE Cumulative Targeting Contract v2 (2026-09-10)

This is a new reproducible scientific benchmark, not a reconstruction of the
unknown April configuration. `study_plan_public.csv` and its JSON mirror are
the current run settings. Legacy numbers are preserved administratively but
are not grading targets. The authentic DGP and bundled package are unchanged.

## Scientific design

Use the supplied nonlinear, reversible binary-treatment, non-survival DGP.
Estimate E[Y(always treat)] - E[Y(never treat)] on the **same subjects**.
Use all supplied baseline and time-varying nodes, full history (`k=Inf`),
`mtp=FALSE`, binomial outcome, no censoring, unit subject weights, and
`obs_id=1:n`. Do not substitute a different DGP, oracle propensity, or outcome
model. Learner vectors are encoded as pipe-separated names in plan CSVs.

The method is based on sections 4.1-4.2 of Tran et al.,
[Robust variance estimation and inference for causal effect estimation](https://arxiv.org/abs/1810.03030v1),
and the author's full-data/targeted-bootstrap study architecture. The four
required method labels have these definitions:

1. **Standard LMTP Contrast (EIF)**: fit each arm with the bundled
   `lmtp_tmle(..., boot=FALSE)` and use the additive paired contrast. Point is
   target theta minus control theta; SE is `sd(EIF_target - EIF_control)/sqrt(n)`.
   Summing the two marginal variances is incorrect.
2. **Full-Data Targeting Contrast (EIF)**: for each policy create a fresh
   `lmtp_task`, fit density ratios using `cf_r` first, then initial regressions
   using `cf_sub(task, "tmp_lmtp_scaled_outcome", ...)`. These are held-out
   predictions recombined in original subject order. Do not use a standard
   already-targeted fit as the initial Q. Fit once per policy per replication.
   Target these recombined arrays on the entire dataset as specified below.
3. **Targeted Bootstrap (Wald)**: retain exactly the full-data point from (2).
   Draw B subject-index vectors jointly for both policies, holding original
   initial Q/g fixed. Re-target only, starting from those initial arrays on
   every draw. SE is the sample SD of the B target-minus-control contrasts.
4. **Targeted Bootstrap (Quantile)**: identical point, SE, and B draws to (3),
   but use type-7 empirical quantiles at .025 and .975 for interval endpoints.

Standard and full-data points need not coincide. Full-data EIF and both
bootstrap rows **must** share their point exactly on each replication.
All Wald intervals use `qnorm(.975)` and are not clipped to [-1,1].
The empirical EIF method is not the paper's separate robust-variance TMLE.

### Targeting definition and numerical boundaries

Let `mn` and `ms` be initial natural and shifted n-by-tau matrices. Let
`r` be the already trimmed non-cumulative ratios from `cf_r`, with bundled
`.trim=.999`. Compute `H[i,t]=prod(r[i,1:t])` **before** resampling.
There is no additional cumulative-weight trimming. Initialize shifted
Qstar[,tau+1] to the observed binary Y. For t=tau,...,1:

- Bound only initial mn[,t] and ms[,t] to [1e-5,1-1e-5] before taking logits.
- Solve the monotone intercept score
  `sum(H[,t]*(Qstar_shifted[,t+1]-plogis(qlogis(mn[,t])+epsilon)))=0`.
  Search epsilon in [-80,80], absolute epsilon tolerance 1e-12. Use the lower
  endpoint if its score is <=0, upper if its score is >=0. If total H is zero,
  epsilon=0, retaining initial predictions for that step. These are explicit
  support/boundary conventions, not discarded simulations.
- Apply the same epsilon to initial natural and shifted logits. **Do not
  floor/ceiling the targeted probabilities afterward**; keeping a 1e-5 floor
  would prevent solving a zero-outcome score. This numerical refinement is
  intentional, unlike the recovered author's unchecked IRLS plus final bound.
- Abort nonfinite fits and record the error; never drop, retry different seeds,
  or replace a failed bootstrap draw with an untargeted estimate. Preserve
  all attempted draws. Record boundary/zero-weight counts and numerical warnings
  in the report. Full-data targeting mean weighted residual at each step must
  have absolute value <1e-6.

Arm point is mean(Qstar_shifted[,1]). Its uncentered EIF is
`Qstar_shifted[i,1] + sum_t H[i,t]*(Qstar_shifted[i,t+1]-Qstar_natural[i,t])`.
Use the paired arm EIF difference with sample variance divided by n.
The bundled `boot=TRUE` path uses raw ratios for targeting and is **not** this
method. Implement the cumulative targeting step in your six scripts without
changing the package namespace or public inputs.

## Plan and RNG protocol

Public settings are newly selected: n=200 and 1000, tau=3, moderate effect and
positivity, 100 replications each, B=499, 5 outer folds and 5 inner folds,
SL.glm in both libraries. Main-effects misspecification is deliberate. The
coverage Monte Carlo SE is at most .05 at 100 replications; avoid claiming
precise nominal coverage or nominal inference with poor empirical support.

`05_run_full_simulation_longitudinal.R` runs the public CSV when no local
`fixture_smoke_plan.csv` exists. When that file exists, it replaces the entire
public plan. Plan header:

```text
scenario,n,tau,treatment_effect,positivity_violation,replications,base_seed,bootstrap_B,folds,learners_outcome,learners_trt,tau_true_seed
```

Scenario strings are opaque keys. Obtain sample sizes from plan/raw metadata,
not by parsing scenario text. Reruns use tau>=2, folds>=2, replications>=2, B>=2.
Only six scripts are copied; no extra helper scripts or old outputs accompany
them. Do not hardcode public settings in helpers or require a previous session.

Use `/opt/R/4.3.2/bin/Rscript` (R 4.3.2) for every R session, including analysis,
not unqualified `Rscript` or `/usr/bin/Rscript`. Use the installed task-controlled
package closure described in
`runtime_manifest.json`, not current CRAN/GitHub releases. Sequential futures,
one BLAS/OpenMP thread, `RNGkind("Mersenne-Twister","Inversion","Rejection")`,
locale C, UTC. Parallel independent replications are permitted with identical
per-replication seeds, maximum four processes; no parallel nuisance fits.

For replication r=1,...,replications, let s=base_seed+r-1:

- Set seed s immediately before `generate_longitudinal_data`.
- Standard fits: reset seed s+100 before **each** policy, control first.
- Full-data initial fits: reset seed s+200 before **each** policy, control
  first; create task, then cf_r, then cf_sub, without intermediate RNG resets.
- Bootstrap: reset seed s+5000 once. For each draw call
  `sample.int(n,n,replace=TRUE)` once, use those indices for both arms. Targeting
  must not consume RNG. Do not resample Q and g independently or refit learners.
- `folds` controls both outer and inner folds. Other package controls remain
  defaults, including bound 1e-5, trim .999, return_full_fits FALSE.
- Truth: call supplied `compute_true_effect_mc(n_mc=100000,tau,beta_psi,beta_p)`
  under `tau_true_seed`, preserving surrounding RNG state. Cache, if desired,
  by (tau,treatment_effect,positivity_violation,tau_true_seed). The DGP's ambient
  cache must not decide reported truth. Moderate maps to .5 and -1 respectively.

## Public interface and deliverables

Working directory is output/. DGP path is environment variable
`LTMLE_HIDDEN_SMOKE_DGP_SOURCE` when set, otherwise
`../input/LTMLE_Targeted_Bootstrap_Task_INPUT/01_data_generation_longitudinal.R`.
Write all solver-created files under output/. Required files are the six R
scripts named in the task prompt, `raw_results.csv`, `summary.csv`, `report.pdf`.
Your pipeline generates raw data; the evaluator does not supply it.

From `output/`, run these commands in separate fresh R sessions:

```bash
/opt/R/4.3.2/bin/Rscript 05_run_full_simulation_longitudinal.R
/opt/R/4.3.2/bin/Rscript -e 'source("06_analyze_part2_results_longitudinal.R"); analyze_part2_results_longitudinal(output_dir = ".")'
/opt/R/4.3.2/bin/Rscript -e 'source("06b_analyze_by_sample_size.R"); summary_df <- read.csv("summary.csv", stringsAsFactors = FALSE); grouped <- analyze_results_by_sample_size(summary_df); stopifnot(length(grouped) >= 1L)'
```

The first postprocessor must recompute summary from raw, not return an existing
summary. The second returns sample-size groups, using raw/plan scenario metadata.
The report must present results and discuss bias, variance calibration, coverage,
interval width, model misspecification, Monte Carlo precision, and diagnostics.

Exact raw header, with one row per (method,scenario,replicate_id):

```text
method,scenario,replicate_id,seed,bootstrap_seed,n,tau,bootstrap_B,folds,learners_outcome,learners_trt,treatment_effect,positivity_violation,reference_policy,target_policy,estimate,std_error,conf_low,conf_high,tau_true
```

Use `reference_policy=never_treat`, `target_policy=always_treat`. All numeric
values must be finite, SE>=0, low<=high. Keep raw numeric precision (at least
12 significant digits); don't round raw draws to six decimals.
Exact summary header:

```text
method,scenario,bias,empirical_se,estimated_se,se_ratio,coverage,ci_width
```

Per method/scenario use mean(estimate-truth), sample SD(estimate), mean(SE),
mean(SE)/sample SD(estimate) (0 if denominator=0), inclusive coverage, and mean
interval width. Round only final metrics to six decimals. No `na.rm=TRUE`.
Summary derivation tolerance is 1.1e-6; shared raw point/SE tolerance is 1e-8;
Wald endpoint tolerance is 1e-7. Numerical-reference absolute tolerances are
bias .03, empirical SE .05, estimated SE .05, ratio .2, coverage .12, width .12.
These do not relax mathematical identities or permit copying public answers.

The deterministic hidden rerun also compares each replication's estimate and
SE to its independently computed reference within 1e-5, and each CI endpoint
within 1e-4. This catches covariance omission or wrong targeting weights even
when aggregate summary differences happen to be small. The runtime, seed
schedule and numerical specification above apply equally to this check.
