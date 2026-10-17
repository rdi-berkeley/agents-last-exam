# Independent native monitoring reader: bounded r10 implementation

R12 adds physical rectangular crown-beam interpolation, native rotations and
mixed-variable connection parsing, with actual installed-Kratos controls.
See `native_replay.md` for the new independent case interface and its release
boundary. The r10/r11 observations below remain historical evidence; their
shell-only limitation does not describe the new crown reader. Full registered
five-case native validation remains pending.

The task is still unreleased. `scripts/native_result_adapter.py` now reads the
actual retained Kratos `ModelPartIO(WRITE)` export and extracts physical
monitoring observations. It does not solve FE equations or establish source
compliance. `scripts/native_result_controls.py` reproduces the r10 controls
without Kratos, a VM command, or another pos0 solve.

## Trust and supported data

An evaluator-owned execution context must supply the native output digest,
the end-of-Stage-1 capture digest, reviewed outer-wall property membership,
physical monitor registration and required top/toe elevations. None may be
taken from a candidate manifest or alleged replay receipt. The r10 control
driver pins the already captured r09 artifacts; these pins are regression
evidence, not an answer reference or a required model for submissions.

The reader parses native nodes, element connectivity, `DISPLACEMENT_X/Y/Z`,
`Active` element/constraint memberships, `Soil` nodes and actual
`LinearMasterSlaveConstraint` offsets. Missing fields are errors. Excavated
cells remaining in the root export cannot supply ground interpolation.
The Stage-1 baseline is the separately pinned evaluator-run NPZ capture,
read with pickle disabled. This use of a trusted run's capture does not make
uploaded NPZ files authoritative and does not provide a restart checkpoint.

For this runner, an installed wall node starts at `X_soil + u_install` and its
native constraint gives `u_wall = u_soil - u_install`. The reader reconstructs
`u_wall + u_install - u_Stage1`, verifies the native attachment and recovers
original geometry from the attached soil node. Tests include nonzero
installation motion and a nonzero baseline, independent of the near-zero
baseline in r09. No late displacement reset is allowed.

DBC values use the linear shape functions of actual active tetrahedron ground
faces. CX values use bilinear shape functions of reviewed, vertically extruded
Q4 shell panels along the registered physical wall profile. Every crossed
vertical interval is covered, including both endpoints and all inter-element
levels. Horizontal displacement is linear with depth on each such interval;
its norm is convex, so its maximum occurs at an endpoint. This obtains the
exact maximum of this discrete FE field without a tunable sampling step. It
does not prove mesh convergence. Missing intervals and inconsistent adjacent
values fail; there is no nearest-node fallback or extrapolation.

The supported representation is m/kN/s with upward z, planar ground sampling,
tetrahedral soil, extruded Q4 walls and the runner's bonded single-master wall
attachments. Other legitimate native elements/interfaces need their own
interpolation adapters. Unsupported data is an evaluator limitation, not an
engineering violation. Node/element numbering and rigid plan rotations and
translations do not determine the observed magnitudes; tests check these.
A separate analytic-field control changes both soil-face triangulation and wall
depth subdivision. Both meshes reproduce the same physical observations while
retaining their different native connectivity and profile row counts. Each
model is compared to its own extraction, never to another mesh's node lists.
Pore-pressure, stress/plastic-state and complete input/history compliance are
outside this displacement reader and remain separate replay obligations.

Native MDPA numbers are serialized at approximately six significant digits.
The observed largest displacement difference against the full-precision r09
TSV is 4.99648e-7 m (0.000500 mm). Current replay explicitly restores coordinates
and translations from the evaluator's native full-precision capture, verifying
its digest, exact node set, finite values and exact six-digit rounding to every
corresponding MDPA value. This prevents near-zero cancellation from becoming a
false interface failure. Submitted CSVs or submitted digests cannot authorize
this path. Connectivity, constraints and rotations still come from native MDPA.
No response or continuity tolerance was increased. A 0.002 m coordinate
budget accounts for native coordinate rounding on this domain, not survey
uncertainty. Attachment comparison propagates rounding from all three terms,
including cancellation. Internal raw-table controls use a 1 micrometre
comparison budget; this is not a new precision requirement on solver delivery.
Final scalar comparisons retain the original max(0.30 mm, 5%) tolerance.

`check_reported_monitors` validates the adapter's normalized extraction tables
and reported maxima against independently read fields. Its internal table
layout is not an extra solver deliverable or a requirement to submit a fixed
number of samples. A production delivery adapter must normalize the original
raw tables to the reviewed physical profiles before comparison. Matching
these tables alone cannot remove the setup/evaluation guard.

## Source registration and the concrete geometry decision

The source figure identifies cyan DBC1..10 ground stations and purple
crosshair CX1..10 wall profiles. Green ZQS/ZQC wall-top symbols are different
instruments. The benchmark provides approximate outer 115 m by 36..89 m and
inner 92 m by 32 m dimensions, but no surveyed monitor coordinates. The
existing public all-DBC/all-CX and Stage-1-baseline convention still controls.

`scripts/evidence/r10/monitor_registration.json` freezes symbol centres, the
original figure SHA256 and the existing trial's pixel-to-model transform
before the new extraction. DBC centres are transformed directly. CX centres
are projected to the already traced outer wall; every offset is recorded in
`registered_monitors.json` (0.0552..0.7112 m, roughly 0.4..5.2 source pixels).
The raster symbols and thick wall strokes do not define exact survey points.
These are disclosed digitization choices for this diagnostic model, not
mandatory coordinates for another submission. The Agent remains responsible
for drawing interpretation and monitor registration. A reviewer checks that
registration against the source before responses; the evaluator does not
replace it with this hand-digitized trace. Future families must keep their
reviewed physical registration fixed across cases and mesh changes.

The r09 inner quadrilateral has a minimum-area enclosing rectangle of
80.1123 by 30.6143 m and maximum vertex separation 83.4747 m. R11 independently
confirms these trial metrics, but finds that two of its closing edges are not
source boundaries and its image-axis metric calibration is not dimensioned by
the source. This is not a registered measurement of the source's full inner
region. The approximately 92 by 32 m text therefore cannot be declared
inconsistent with a second dimensioned source input on this evidence. See
`source_geometry_audit.md` and its annotated unscaled raster. Interpretation
and registration remain Agent engineering work; do not stretch the trial to
92 m or impose a replacement polygon or an exact coordinate-equality test.

The r09 outer wall stops at +6.50 m. Requiring the section's +7.00 m top and
-26.50 m toe produces the exact rejection `Wall profile gap at CX1: 6.5 to 7 m`.
Diagnostic extraction explicitly uses only the existing -26.50..+6.50 m shell
and never calls it full-source coverage. The source also draws a crown beam
and distinguishes its crest from the first support reference axis. R11 verifies
native finite-section first-level beams; the shell-only gap is an adapter and
idealization issue, not proof that all physical crown structure is absent.
It returns ten DBC stations and ten
CX profiles (400 element-endpoint records). Their values are retained in
evidence, not published as task input answers or used to set geometry.

## Bounded next work and release boundary

1. Review the Agent's source region, plan/brace registration and wall/crown
   idealization before another full run. Preserve the +7.00 m crest and source
   section semantics while allowing legitimate shell/beam partitions. Extend
   the reader to reviewed free-wall vertices or finite-section beam kinematics
   as appropriate. Do not attach vertices to nonexistent soil, zero-fill
   observations, double-count the crown, or require a separate 0.50 m shell
   strip solely because the present shell-only reader lacks that coverage.
2. Use one common family mesh exposing all five inner-support elevations
   -10.10, -11.10, -12.10, -13.10, -14.10 m and the original layer boundaries.
   Extend case-dependent Stage-9 installation without relocating loaded beams.
   Reuse the measured AMGCL backend and preserved construction semantics.
3. Run that corrected pos0 through Stage9 with full monitor coverage. Then run
   the remaining four independent histories, each checked against its own
   trusted replay. Test a different legitimate discretization in a bounded
   native control; another complete five-case family is not mandatory. Exercise
   remaining native compliance mutations and proportionate engineering checks
   of mesh/domain/hydraulic adequacy without prescribing an extra study count.
4. Integrate trusted input freezing, source review, replay, reader and original
   delivery scoring in `main` only after those controls pass. Keep original
   weights and tolerances. Until then missing replay remains an evaluator error.

No extra public scalar is needed for monitor digitization, rock stiffness,
undimensioned sections or hydraulic spatial treatment. The already disclosed
6.50 m inner embedment remains the dimensioned-section clarification. This
round changes neither that clarification nor original input bytes, mesh, soil,
stages, support positions or existing native evidence.

Reproduce the saved-field controls from the authoritative checkout:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python tasks/engineering/inner_support_elevation_optimization/scripts/native_result_controls.py
uv run --no-sync pytest -q tests/tasks/test_inner_support_results.py
```
