# Deterministic A-zone model contract, 2026-10-06

This adaptation keeps the original input bytes and source manifest unchanged.
The fourteen strata, material constants, excavation depths, water milestones,
structural selfweight and five support positions retain their original values.
The following explicit task decisions control ambiguous or open source wording.
They are benchmark modeling decisions, not recovered historical definitions.
Evaluation uses no language model and no subjective source interpretation gate.

## Physics and construction

Use the supplied Kratos 10.4.3 support-native-family-1 interface in m-kN-s,
absolute upward z, with ground and initial water at +6.50 m. Soil uses native
GeoMohrCoulombLaw3D. Preserve all fourteen thicknesses, bulk weights, c, phi,
K0, nu and listed Es. Both unlisted rock moduli must exceed 46 MPa. Preserve
the fourteen native layer/property bindings and positive element volumes.
Do not cross stratum boundaries with elements. An explicitly declared
deep_rock_continuation may extend only the deepest source rock to the bottom.

C35 has E=30 GPa, nu=0.20 and weight=25 kN/m³, including actual gravity loads.
Outer/inner shell thicknesses are 1.00/0.62 m. Use bonded soil/shell translations,
native linear beams and the supplied rigid interpolation connections.
Prescribed spatially uniform hydrostatic pressure, no suction above water,
is the hydraulic model of this adaptation. One positive rectangular
engineering_choices.beam_section_m defines all non-crown members. Crowns
may have their own positive rectangular sections. Prose about alternatives
does not request a different executable operation. Other engineering_choices
text is documentation, not an executable override.

The outer base is -10.10 m, inner base -15.10 m, outer toe -26.50 m and
inner toe -21.60 m. Inner embedment is fixed at 6.50 m. Keep beta=0.30 and
n=0.40 as a rounded legacy study label; no exact n denominator is imposed.
Outer support axes are +6.50, +0.70, -3.30, -6.60 m. The one inner support
is at -10.10 minus position (0..4) m. Keep all other inputs fixed.

The supplied runner preserves the nine stages. After K0 and outer wall
construction, install the first outer support. Successive water/excavation
depths below +6.50 m are 6.3/5.8, 10.3/9.8, 13.6/13.1 and 17.1/16.6 m;
install outer supports 2..4 at the corresponding exposed levels. Stage 7
completes outer excavation. Stage 8 constructs the inner wall and crown.
Install pos_0m then; install positions 1..4 during Stage 9 when their levels
are exposed, before further excavation. Final water/excavation depths are
22.1/21.6 m. There is no fifth outer brace or second inner support.
Retain material state and installation offsets. Only a completed common
Stage 7 checkpoint may branch into independent alternatives.

## Deterministic admissibility

The figures guide reconstruction. To make acceptance explicit, this adaptation
replaces visual judgments of exact digitization, member layout and monitor
symbol matching with the following admissible family. These geometric limits
are new public tolerances, not surveyed coordinates or recovered source rules.
Run SOFTWARE/scripts/native_model_contract.py FAMILY before expensive solves.

- Polygons are finite, simple, have distinct vertices without a repeated closing
  vertex, and positive area. For every polygon edge direction, form the bounding
  rectangle; choose the least-area rectangle. Outer shorter/longer dimensions
  must be 60..120 / 100..140 m; inner dimensions 25..45 / 75..105 m, inclusive.
  Outer polygon area divided by that rectangle area must be at most 0.95.
  The inner boundary lies strictly inside the outer polygon without crossings.
- Both pits lie strictly inside the rectangular domain. Its bounds match actual
  mesh nodes, top is +6.50 m and bottom contains all source strata. Recommended
  230x185x60 m extents are not exact requirements.
- Wall faces are vertical Q4 rectangles on their excavation perimeter, covering
  every edge from the source toe to the declared shell top without gaps or
  overlapping panels. A declared crown attachment lies within its section below
  its top; signed eccentricity is permitted. Outer crown top is +7 m. Its axis
  must coincide with the first outer support (+6.50 m); inner crown axis is
  -10.10 m. Without crowns, shell tops are +7 / -10.10 m.
- Exactly one outer and one inner brace graph is supplied. Each graph is connected
  by shared endpoint coordinates, includes perimeter waler members and contains
  a diagonal cross-pit member and at least two wall attachments. All endpoints
  lie inside or on the respective pit. Inner crown segments cover only its
  perimeter (2 micrometre coverage tolerance). Collinear subdivisions are
  allowed; join member intersections. Continuous outer-waler coverage between
  registered profiles is not an additional gate in this adaptation.
- Exactly ten DBC and ten CX registrations are finite and at least 1 m apart
  within each group. DBC points are at +6.50 m, strictly outside the outer pit,
  within 15 m of its perimeter and inside the domain. CX points are on that
  perimeter with a marker elevation within -26.50..+7 m; the entire profile
  is extracted independently of that marker elevation. Their approximate drawing correspondence is documented;
  no hidden coordinate registration is graded. The frozen mapping is identical
  in all five replays.
- The local-refinement requirement remains mandatory. Compute each soil element's
  maximum horizontal vertex-pair distance and its centroid's horizontal distance
  to the nearest inner/outer perimeter. Both near (<=10 m) and far (>=20 m)
  groups must exist; near median diameter must be <=0.95 times far median.
  This explicit geometric refinement test supports tetrahedra and hexahedra.
  It is not a numerical error bound or a prescribed element count.
- Declared relative residual/displacement tolerances may not exceed 1e-6;
  absolute residual tolerance may not exceed 1e-5 and absolute displacement
  tolerance may not exceed 1e-7. Values must be finite and positive.
  Iteration count is only a work allowance. Native convergence, volumes,
  material bindings, gravity, boundaries, activation and retained history
  are checked separately on evaluator-owned native execution.

## Observations and evaluation

Use all DBC1..10 and CX1..10, with displacement u relative to equilibrated
Stage 1, never resetting construction movement. Report in mm:

- settlement: max_i(max(0, -u_z(DBC_i)))
- wall movement: max_i,max_depth(sqrt(u_x(CX_i,depth)^2 + u_y(CX_i,depth)^2))

DBC interpolation uses native tetrahedral triangles or horizontal convex
hexahedral Q4 faces, including bilinear cross terms. CX profiles cover -26.50
through +7.00 m. The reader checks every represented interval endpoint.
At a shell/crown junction it verifies actual endpoint attachment equations,
not equality of unrelated linear shell and cubic beam interior traces. Both
traces contribute to the maximum. Same-owner discontinuities, ownership
overlaps, uncovered intervals and invalid endpoint coupling remain errors.

Own-model replay determines the reference responses; there is no prescribed
curve or optimum. Response tolerance remains max(0.30 mm, 5%); percentage
tolerance remains max(0.30 percentage point, 8%). Additional mesh/domain
studies are further numerical validation, not another acceptance prerequisite.
Observed errors and invalid extraction remain failures to establish native
results. Contract violations receive zero with a concrete check failure;
runtime/infrastructure failure or incomplete independent replay raises
evaluator-unavailable. Passing preflight is not physical certification.
See delivery_schema.md for the deterministic 50/50 scoring and memo fields.
