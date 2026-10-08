# Native replay integration (unreleased)

R15 adds the executable family coordinator and normal task hooks, documented
in `native_interface.md` and `native_task_integration.md`. Actual formal setup,
common-Stage7 native reading and saved-pos0 comparison pass under CPU2/2GiB;
the complete five-case native entrypoint is still unvalidated. Private
boundary/refinement diagnostics first reached3570s caps at22/16 completed
substeps. The deeper-domain continuation now completes Stage9/audit/reading
in1967.65s and passes its fixed final-response comparison. Refinement still
awaits continuation from the saved16-step state, so full adequacy is undecided.
They are not a second five-case family or an extra Agent assignment.
R16 implements two independent suffix workers with2GiB limits,5GiB aggregate
memory and two CPUs, separate writable state and an immutable common prefix.
Native isolation/accounting controls pass without another FE history; complete
five-case execution and normal timed scoring remain pending.
The following dated R14 snapshot is historical.

## Current corrected-family terminal, 2026-09-30 05:29 UTC

`support-r14-corrected-resume4-pos0` has finished Stage9 with41 successful
substeps, native/systemd exit0 and MainPID0. Its Stage8-to-final continuation
takes447.563845 s. Source-fact auditing and independent physical monitoring
(10 DBC stations,510 CX rows) pass. See
`scripts/evidence/r14/corrected-resume4/pos0-terminal.json` and
`pos0-independent_monitors.json`. No support FE unit is running at this check.

Resume3's earlier exit-11/systemd245 was a new-shell initialization SIGSEGV,
not its05:34 runtime ceiling. GDB and matching native source identify the MITC
shell's empty constitutive-law vector after restart suppressed initialization.
The runner now initializes only new shells with IS_RESTARTED temporarily false,
restoring the flag before continuing the existing model. The repaired Stage8
probe exits0 in24.16 s/5 iterations and preserves all four checked old-field
hashes. No existing constitutive state, physics or tolerances were reset.

Common Stage7, corrected first-water and final native checkpoints remain
retained. Original positions1..4 must branch before inner-system installation;
they have not been run. The source-review hook now executes in `main.evaluate`;
the live review leaves mesh/domain adequacy unresolved. At that R14 snapshot,
normal setup and the numerical/delivery scoring connection were unfinished;
R15 now implements them with full five-case native controls still pending. This native
positive does not award source compliance or a task score. The historical r13
failure and restart evidence below are not the latest case status.

## Supported native replay and retained history

The installed Kratos runner now accepts `--position 0` through `--position 4`
with independent initial stress and construction histories. It installs the
inner crown after the wall, and installs the sole case brace only after its
level is exposed. Native state persists through the case. Saved displacement
or MDPA output is not a constitutive restart.

R13 adds `--max-iterations` and `--restart CHECKPOINT` to the developer
entry point. Increasing the Newton iteration budget changes neither the
declared convergence criteria nor physical inputs. Restart loads a native
`FileSerializer` Model, with constitutive history, solution-step buffers,
active sets and installed-member constraints, plus checked runner metadata.
It verifies native-state and input digests. Changing case position is rejected
after inner-system installation. A common pre-inner-installation state may
be cloned into separate processes only after changing-load continuation
equivalence has been validated; cached displacement equality is insufficient.
The retained r13 continuation control is the evidence for that decision, not
a candidate-authored restart assertion. Restart status and inherited prefix
time remain explicit in each receipt.

Each successful substep replaces `restart/latest` with a complete serialized
native Model. Newly frozen runners retain the first-water or Stage7 branch with
hard links to that immutable saved inode; later `latest` replacement preserves
the branch while avoiding a duplicate full state allocation. The last Stage7
branch must still be checked for completed outer excavation and absence of
inner installation before independent case suffixes use it. This storage change
does not change solver state, model inputs or convergence settings, and does not
modify an already running frozen script.

The r13 changing-load control exits0 after599.681 s. It repeats the second
and third Stage4 drawdowns from the native first-drawdown checkpoint; final
node/element identities, displacement, rotation, pore pressure and effective
stress match the uninterrupted full-model trajectory exactly. This supports
native state continuation for that tested transition. It does not establish
completed excavation/support branching, full physical monitor coverage,
five native cases or task acceptance. Earlier failed hold controls are retained.

The registered r13 pos0 run then terminates at Stage4 excavation4 after200
Newton iterations, native exit1 at2274.067 s. Ten prior substeps completed;
the complete native excavation3 checkpoint remains available for bounded
recovery without changing physical inputs or convergence tolerances. This is
an iteration-budget failure before the one-hour ceiling, not a Stage9 positive.

`scripts/native_replay_case.py FAMILY OUTPUT REVIEW --position N` is an
evaluator-owned developer entry point. REVIEW must come from the evaluator's
independent source/model review, outside the submitted model's authority. It
pins the three native input files, the physical monitor registration and the
required outer-wall extent. Matching those hashes establishes identity, not
engineering adequacy. A candidate cannot authorize its own model by supplying
this document. The task source-review stage produces this handoff; the
numerical/delivery hook is connected, with complete five-case native validation
still pending as detailed in `native_task_integration.md`.

The entry point runs a fresh allowlisted native process, records its terminal
status, and independently extracts all physical DBC/CX observations. Original
native fields, installation offsets, active membership and the Stage1 capture
supply observations; a candidate's summary CSV does not. If the native solve
succeeds but extraction fails, `terminal.json` retains both facts separately.
Different reasonable models compare to their own replay, not to private
witness responses or another model's mesh and monitor row numbering.

Current newly frozen runs also call `native_source_audit.py` after successful
native execution and before monitor extraction. It reopens the evaluator-owned
full checkpoint without an FE step and checks effective source soil parameters,
all fourteen strata and initial volume, active excavation membership, side/bottom
fixities, actual gravity, the declared hydrostatic field, C35 properties, wall
thicknesses and toes. Native-execution success and source-audit failure remain
separate terminal facts. These checks do not certify source-plan interpretation,
mesh/domain/hydraulic adequacy or full engineering acceptance. In particular,
checking this model's declared hydraulic field does not prescribe that spatial
choice for other submissions. No private polygon, section or rock modulus is
introduced as a public requirement.

The standalone command `native_source_audit.py FAMILY CHECKPOINT REVIEW OUTPUT
--controls` performs the same read-only audit and in-memory native mutations for
wrong cohesion, omitted wall weight, wrong wall thickness, a freed bottom and
reactivated excavated soil. It restores mutated fields and never writes a changed
checkpoint. This is a diagnostic on trusted native state, not a candidate report
oracle, and it always leaves full-task acceptance false.

Use the same command with `--extract-only` to read a saved evaluator-owned
native run without starting another FE process. It requires native exit0,
completed Stage9, matching reviewed inputs and the requested case. The separate
`reader-terminal.json` records success or failure and the original terminal's
digest; it never rewrites that original native receipt. This allows result-reader
repairs to be tested on preserved fields, without repeating a successful solve.

`native_checkpoint_monitors.py FAMILY CHECKPOINT REVIEW OUTPUT` is a separate
developer diagnostic for a retained intermediate state. It loads the actual
native Model, exports through native `ModelPartIO`, reads physical DBC/CX
fields and exercises missing/fabricated/wrong-monitor reporting controls.
It does not solve another FE step and always records Stage9/task acceptance
as false. This diagnostic cannot replace the full-case completion gate above.

`native_stage_gate.py FAMILY OUTPUT REVIEW --until outer-wall --runtime 100`
runs a fresh K0/outer-wall prefix. Its first-water mode requires `--runtime 580`
or a shorter budget and `--restart` pointing to a matching Stage2 native state.
It rejects post-installation or different-family restarts, records child exit
status and verifies the committed checkpoint digest. Use CPU2/3 GiB supervisor
caps of120 and600 seconds, respectively. These are developer diagnostic
budgets, not new task engineering constraints. A successful process without
the requested verified checkpoint is a failed gate.

The result adapter supports vertical Q4 walls and reviewed horizontal native
`CrLinearBeamElement3D2N` rectangular crowns with zero effective shear area.
It reads rotations, section properties, actual initial orientation and native
installation geometry, including physical points away from the centroid.
Cubic bending and section rotation determine crown displacement; it rejects
missing rotation data and points beyond the actual section. Combined vertical
profiles cover the complete wall/crown extent. Other legitimate native
elements/interfaces require their own adapter and must not be called
engineering failures solely for being unsupported here.

Native MDPA serialization can lose installation motion when large coordinates
or displacement terms are subtracted. The replay reader binds the trusted
runner's full-precision native TSV separately, verifies the entire node set,
finite values and exact agreement of every coordinate and translation with
MDPA's six-significant-digit serialization, then restores their native precision.
Rotations, connectivity and constraints remain native MDPA fields. Loading
translations requires an explicit reader option; coordinate-only reading does
not implicitly trust other TSV columns. The capture is produced by the
evaluator's native replay, never a candidate report or candidate-supplied digest.
At shell/crown joins, initial section placement from the native frame is
separated from attachment motion for the continuity check. Actual reported
material-point displacements retain that placement term. Overlapping members
with inconsistent interpolated responses still fail; thresholds are unchanged.

Preparation splits coincident collinear trace spans at their endpoints before
deduplicating edges, just as it already splits crossing members. This prevents
one traced centerline from becoming duplicate parallel stiffness in a private
test graph. It does not prescribe a source polygon or member section. Changing
a previously installed member graph requires a separately validated native
history; do not retrofit a constitutive checkpoint to a different graph.

One test model can partition wall/crown volumes differently from another.
No shell strip, crown section, inner polygon, digitized monitor coordinates,
drawdown shape or rock modulus from private diagnostic evidence is a public
requirement. The source-preserving public clarifications remain exactly those
in `model_definition.md`; registration and the other open modeling choices
remain the Agent's engineering work. Adequacy, correct loads/history and
sampling consistency still require review and validation.

The retained-guest helper `scripts/support_replay_trial.py` uses unique frozen
directories and detached systemd units on endpoint38500, CPU2/3GiB, with a
configurable `--runtime` native child ceiling (60..3600 seconds). The systemd
ceiling is30 seconds longer, allowing terminal collection after a child timeout.
For a strict30-minute job use `--runtime 1770`, giving a1800-second supervisor
cap. Use its `status` and `collect` actions to
observe existing jobs; do not relaunch a running case. It downloads compact
receipts/observations only. Each case must finish before launching the next
on that CPU budget. `--cpu 0` requires that shared CPU to have been released
for this work; it retains the same3GiB cap and never launches another VM.
Its `upload` action requires an explicit `--family DIRECTORY`; it must not
silently reuse an obsolete private preparation. A graph correction after
member installation invalidates post-installation restart state. Use a complete
pre-installation state with matching validated inputs, or reconstruct the short
unaffected prefix if none was retained. Keep elapsed times of all failed,
timed-out and operator-stopped attempts in the cumulative execution ledger.
A native Stage9 completion is distinct from successful
monitor extraction, five-case validation, numerical sensitivity and original
delivery scoring. None is inferred from unit tests.
