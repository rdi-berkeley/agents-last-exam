# Compliance and replay design: release blocked

## Validation scope clarification, 2026-09-30

The source requires the five support heights 0..4 m in one consistently chosen
model family. It does not require two complete five-case mesh families or a
match to unavailable historical author responses. Earlier proposals for ten
full native cases were evaluator-development suggestions, not source conditions;
they are withdrawn as a mandatory release gate. Validate one complete compliant
five-case family against independent replay. Demonstrate acceptance of other
legitimate discretizations with a bounded alternate native control that changes
the mesh or numbering and exercises physical-point interpolation and effective
input reading. Its scope and observed results must be explicit; it cannot stand
in for the original five histories. Escalate to further full-case comparisons
only if an observed numerical or modeling issue calls for them. Mesh/domain and
profile adequacy still need proportionate engineering evidence, without a fixed
extra study count or a hidden prescribed model.

R14 currently continues the corrected source-registered pos0 family from full
native checkpoints. The earlier r09 Stage9 receipt below belongs to a different
preparation and does not validate this corrected model. The restart continuation
and saved-field controls are described in `native_replay.md`; setup/evaluation
remain guarded pending actual full-family validation and trusted integration.

## Implemented boundary

The candidate has source-fact transcription, original-input SHA256 checks,
strict report syntax checks, a bounded native preparation reader and staged
runner, and unconditional setup/evaluation errors. The unchanged fourteen-layer
preparation now completes all nine logical stages in actual Kratos execution:
r09 `pos_0m-full-amgcl-gmres-4` exits 0 after 1399.694 seconds, peak RSS 444672 KiB,
with 36 converged native substeps and exported final model/nodal/stress fields.
This is one preparation history, not accepted source geometry, a monitored
full case or full-family replay. Its 92 by 32 m inner-plan registration, traced
bracing review, physical wall/crown coverage to +7.00 m and convergence checks remain
outstanding. R10 now registers all DBC/CX symbols for this preparation and reads
their displacements independently from the retained native MDPA, with activity,
installation-offset, baseline and coverage controls. Strict source wall-height
coverage correctly rejects the unrepresented +6.50 to +7.00 m shell segment. This partial
diagnostic is not accepted full-source monitoring. Positions 1-4 are not yet
supported by this runner.
`compliance_requirements.json` is a declarative list of validator obligations,
with every native check marked pending. It is not a schema that a candidate
can fill with true/false assertions to obtain credit. The native reader resolves
actual MDPA/material assignments and checks the fourteen strata. Full-source
model compliance, trusted full replay and a numerical scorer remain
unimplemented. See `native_result_reader.md` for the bounded independent
displacement reader, geometry findings and executable saved-field controls,
and `scripts/NATIVE_FAMILY.md` for the preparation scope and native receipts.

The r11 source-only crosscheck in `source_geometry_audit.md` distinguishes
the trial's chosen quadrilateral/calibration from a dimensioned source polygon.
Its 80.11 by 30.61 m enclosure does not establish an original-input contradiction
with the approximately 92 by 32 m text. The source also distinguishes the
+7.00 m crown crest and +6.50 m first-level axis; existing native beams have
finite section properties. Review the physical wall/crown representation and
support its native kinematics rather than universally requiring one extra
shell strip. No additional mandatory public number or unique monitor map is
introduced by this audit.

The previous fixed-model scorer and uploaded-NPZ extrema gate are removed from
the executable path. A successful syntax check is only a successful syntax
check. The draft remains inspectable through task discovery, but setup fails
before preparing a solver task, and evaluation fails before reading candidate
files, references or credentials. Neither failure returns a score of zero.

The structural probe in `scripts/probe_structural_selfweight.py` checks actual
Kratos beam body-load assembly, including zero-density/acceleration controls.
Native stiffness and residuals are equilibrated with NumPy for two gravity
increments and a newly installed beam, using an explicit incremental connection.
Analytical cantilever deflections, retained displacement, the installation
imbalance and reaction/weight balance are checked. The coupon dimensions are
test fixtures, not required excavation sections. This proves a narrow native
beam calculation. The follow-up `scripts/native_components.py` now implements
a bounded native wall/MC-soil/prescribed-pressure stage runner. Its five native
tests validate shell selfweight/bending, pressure-to-effective-stress coupling,
plastic history, excavation removal, and loaded wall/beam installation in
the current configuration. The compatible eight-strain shell law and native
scheme lifecycle exit cleanly, resolving the earlier exploratory shell crash
for this supported path. See `scripts/NATIVE_COMPONENTS.md` and the captured
native receipt for controls and limitations.

These component tests do not implement the native input/result adapters or
full replay described below. The coupled excavation fixtures remain on the
elastic branch of MC; plastic history is tested separately. Gravity equilibrium
is not the source K0 state, and prescribed water pressure is not a seepage
solution. Full native compliance obligations remain pending.

## Native adapter and evaluator-owned model

1. Preserve all original inputs in `source_manifest.json`, including six PNGs
   and the original answer template. Verify their bytes during preparation.
   Install the runtime separately. Add the corrected public notes without
   replacing original files or importing any reference model or answers.
2. Freeze the submitted native inputs outside the writable solver environment.
   Resolve paths under that snapshot, reject escapes/symlinks, enforce parser
   resource limits, and hash inputs and runtime for the audit record.
3. Build a versioned adapter for actual Kratos mesh, materials and analysis
   parameters. Parse native data with tested readers. Resolve the effective
   element-property bindings, submodelpart membership, conditions, constraints
   and process settings. Unused correct material declarations are no evidence.
   Run neither candidate Python nor arbitrary process/module names, expressions,
   plugins or callbacks in the trusted evaluator.
4. Construct an evaluator-owned representation containing coordinates and unit
   transforms, mesh connectivity, resolved properties and constitutive laws,
   supports/connections, gravity and hydraulic data, ordered stage operations
   and physical monitoring definitions. Generate allowed solver processes from
   this representation with evaluator-owned code. Reject unsupported constructs
   as an adapter limitation, without pretending they are engineering errors.
   Add native support and controls before advertising the construct as accepted.
5. Read native result fields using an independent result adapter. Never treat
   an uploaded manifest, CSV, NPZ, log or purported replay receipt as evaluator
   truth. Reporting artifacts are compared against independently generated
   fields, not used as their source.

This representation is internal to the evaluator. The solver delivers a native
project and its original engineering outputs, not a second manually transcribed
model or an additional proof. No full-submission parser is accepted until it can
resolve actual effective input and demonstrate that arbitrary declarations and
unused material records cannot pass compliance.

## Constraint checks

The declarative records distinguish fixed facts, explicit task clarifications,
source interpretation and open engineering choices. `model_definition.md` and
`benchmark.json` now fix the dimensioned 6.50 m inner embedment across five cases,
retain n = .40 solely as a rounded legacy label, correct inner-support ordering
and define all-DBC/all-CX extraction. These are new decisions, not recovered
historical formulas. Fixed material/depth/level values are compared after
unit conversion, with tolerance only for serialization precision. Spatial
layer assignment, member connectivity and actual thickness are checked, not
only counts and names. Structural density and applied body acceleration must
produce the specified selfweight; both a zero-density model and an unloaded
positive-density model must fail.

Geometry requires documented comparison with the original plan/mesh/monitor
figures. The approximate source drawings do not justify an exact reference
polygon, a regular beam grid or a hidden coordinate-equality test. Review the
irregular excavation, diagonal member connectivity, approximate dimensions,
outer/inner systems and datum conversions. Review source interpretation before
inspecting response values. A semantic model review alone cannot establish
mechanical compliance.

The history validator checks the nine logical source events and actual native
state transitions, allowing necessary substeps. Replay must expose active
soil/structures, integrated body loads, effective stresses/plastic state,
water conditions and support forces at each transition. Verify removed soil
ceases to load/stiffen the domain, retained state survives, and new support
activation does not suppress earlier motion. Apply the declared ordering:
Stage 7 completes outer excavation, Stage 8 constructs the inner wall and
crown/ring beam, and the sole inner support activates at its exposed case level
in Stage 8 (position 0) or Stage 9 substeps (positions 1-4), before further digging.
There is no fifth outer cross-pit bracing level or duplicate inner support.

Rock stiffness, adequate domain size and mesh refinement are reviewed against
their original performance requirements. No hidden pair of moduli or fixed
node count stands in for those requirements. Unspecified constitutive and
hydraulic choices must be reasonable and consistently held across cases.

## Observation and replay

Register all ten DBC ground stations and all ten CX outer-wall profiles against
the source figures before inspecting responses. Use the declared station
settlement maximum and full-top-to-toe horizontal-norm maximum, with the common
end-of-Stage-1 baseline and no later construction-state resets. Store physical
coordinates/curves and source correspondence, not
candidate-selected node lists. Evaluate native shape functions at those physical
locations with coverage checks; use adequate sampling along the specified
profiles and verify sampling convergence. Whole-domain maxima cannot replace
these observations. Different meshes must observe the same physical locations.

Independently replay the five histories from the same frozen compliant model,
using the evaluator's allowlisted process runner and installed solver. Recompute
the displacement/stress fields and monitoring profiles before deriving maxima,
percentages and minima. Compare submitted native fields as well as report
numbers: matching ten scalars alone cannot verify the model. Use the original
numerical tolerances in `delivery_schema.md`; measure convergence and replay
repeatability before deciding tie handling. Do not widen tolerances to rescue
the replacement solver or match the retired arbitrary reference.

Only after native compliance and replay may an evidence reviewer assess the
original delivery, images and memo. Intended weights remain 50% numerical,
50% engineering. Unsupported software, missing replay capacity, source
ambiguities and infrastructure interruption are evaluator errors, not solver
failures. Verified engineering violations and false reports can lose credit
once the evaluator can actually establish them.

## Remaining release conditions

### Concrete next implementation sequence

The complete-family interfaces below remain planned. The bounded
`native_input_adapter.py`, `native_family_runner.py`, and a completed native
pos0 preparation through Stage 9 now exist; their entry points and limitations are in
`scripts/NATIVE_FAMILY.md`. They do not yet implement the complete interfaces
listed below. The
existing executable foundation is `scripts/sparse_native_components.py`:
`NativeStageModel.step()` initializes one UPwSolver, then advances, initializes,
predicts, solves and finalizes each step; its native constraints retain
installation offsets. Its fixture-specific constructors are not a full native
input adapter. The family runner now adds two-ended native beam/wall interpolation
constraints and source K0 initialization; these are separate from the earlier
anchored fixture and gravity-only controls. Stage-4 through Stage-9 code paths
now execute successfully on the preparation. Complete source compliance,
monitoring and independent replay still require implementation and validation.

1. **Native input and source validation.** Extend `scripts/native_input_adapter.py`
   with a trusted `load_family(directory)` entry point for native mesh,
   material and analysis files. Resolve actual element/property/submodelpart
   bindings with native readers. Restrict processes to supported evaluator-owned
   stage operations; never import submitted Python. Freeze/hash files before
   parsing. Return the effective model, unit/datum transform, case schedule and
   physical monitor map. First tests read two independently authored small native
   inputs with different numbering and rock moduli, and reject wrong thickness,
   zero applied gravity, unused correct materials and an invented n-derived toe.
   Those are parser/compliance tests, not five-case FE validation. Preserve the
   fourteen source rows and explicitly declared 6.50 m embedment.
2. **Source initialization and stage driver.** Extend
   `scripts/native_family_runner.py` with `run_case(family, position, output)`.
   Reuse the maintained sparse lifecycle, extending native construction to
   connected walls, two-ended braces/walers and the actual source topology.
   Implement and validate source K0 effective-stress initialization through the
   installed native GeoMechanics path before using the full model; inspect its
   actual API and check stresses, pore pressure and equilibrium at layer
   interiors. Save the end-of-Stage-1 reporting baseline. Follow the published
   nine-stage schedule and expose/install the one support at its case level.
   Equilibrate every loaded installation, remove excavated soil loads/stiffness,
   retain plastic history, and record stage active sets, loads, reactions and
   convergence. On failure, retain the terminal and resume only from a verified
   last-converged full native checkpoint with validated constitutive-state
   restoration, or restart from frozen inputs when no such state exists. Do not
   invent rollback by resetting fields.
3. **Independent native results and monitoring.** Extend the implemented
   `scripts/native_result_adapter.py` interfaces `read_native_result`,
   `read_stage1_baseline` and `extract_monitors` beyond the r10 supported
   tetrahedron/bonded-wall preparation. Read actual native displacement, pore
   pressure, stress and activity results; candidate summaries are never the
   field source. Reconstruct total construction displacement relative to the
   Stage-1 baseline, accounting for installation coordinates/offsets. Interpolate
   all ten fixed DBC stations and ten CX top-to-toe profiles, apply the published
   formulas and convert to mm. Verify coverage and profile sampling convergence.
   Test a numbered-mesh change, rotated horizontal axes, wrong monitor subset
   and altered native displacement against independently replayed fields.
4. **First complete family, sequentially.** Start with one faithful source-plan,
   fourteen-layer pos_0m case through Stage 9 using a modest documented mesh.
   Stop on failed K0 initialization, nonconvergence or missing monitor coverage;
   do not substitute a strip or the component coupon. Record runtime, peak RSS,
   mesh/domain and sampling checks. Once that case succeeds, complete the four
   other independent histories from the same immutable inputs, changing only
   support elevation and its installation substeps. A verified common completed
   Stage7 checkpoint may supply the identical pre-inner-system history once
   native constitutive-state continuation is validated. Do not clone a final
   loaded case or move a loaded support. Account for the common prefix once and
   each independent suffix, including interrupted computation, in resource
   receipts. This is the first full-family native validation milestone.
5. **Fair replay and task integration.** Validate a bounded alternate native
   mesh/numbering control and compare each submitted compliant model to its own
   trusted replay, using the unchanged delivery tolerances and weights. A second
   complete five-case family is optional diagnostic work, not a release mandate.
   Apply native wrong-toe/thickness, omitted-stage, zero-gravity, elastic-soil,
   rigid-brace, extra-outer-support, state-reset, wrong-monitor and fabricated-field
   controls at the stage where they can be rejected; do not repeat the entire
   component suite for each control. Only after field and reporting checks pass,
   connect `main.start/evaluate` to the trusted adapters and remove their guard.
   Keep `task_card.json.taskPrompt` synchronized with `assets/instruction.md`
   using base/input and base/output; the clarification patch now does this and
   tests exact equality. Rerun `uv run --no-sync pytest -q tests/tasks/test_inner_support_oss.py`
   and the newly implemented adapter/full-family tests, reporting native passes
   separately from host-only syntax/source checks.

Complete full-family runtime remains unknown. The r09 scalar AMGCL/ILUT backend
solves the retained full-model tangent in 0.691 seconds versus SparseLU 13.508.
All four completed SparseLU states agree; maximum first-water displacement
difference is 7.8e-15 m. The full pos0 preparation completes in 1399.694 seconds
on CPU2/3 GiB with unchanged input bytes and nonlinear convergence criteria.
The Newton iteration budget is increased from 40 to 80 after observing decaying
residual at the cap. All 36 native steps converge. This resolves the r08 measured
linear-solve obstruction but does not complete source/monitoring validation.
Any future cross-process restart still requires complete native constitutive
serialization and a continuation control; result MDPA/NPZ files are not that
checkpoint. The successful r09 history stays in one process throughout.
Do not schedule five-case acceptance until a faithful full pos_0m succeeds.

### Release evidence still required

- Enforce the published embedment, monitoring and stage-order clarifications
  in the native adapter. No historical n denominator is required by the adapted
  task; approximate monitor registration remains modeling work.
- Implement and validate the effective native input/result adapters and
  trusted stage runner, including source material and gravity handling.
- Complete one source-compliant five-case family and independent replay.
  Exercise another legitimate discretization in a bounded native control;
  report its tested scope. Accept different justified responses by comparing
  each model to its own replay, without requiring an exact historical answer
  or a second complete five-case family.
- Test actual native mutations: wrong wall thickness, omitted stage, zero
  structural density, zero gravity, elastic soil, rigid braces, extra outer
  struts, altered monitoring scope, reset state and fabricated result fields.
  Each relevant violation must be detected independently of answer matching.
- Test alternate native node/element numbering, unit transforms, mesh layouts
  and permitted modeling choices, and forged compliance/replay metadata.
- Measure complete replay runtime/storage and numerical/profile convergence
  in the final packaged environment. A component coupon or the retired fixed
  model's positive scores do not satisfy these conditions.

No alternative FE solver is declared supported until its adapter and controls
pass the same checks. No solver Agent, new VM, publication or reference-answer
replacement is required or authorized by this bounded implementation.
