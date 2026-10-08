# Source geometry necessity audit, 2026-09-30

This source-only audit supersedes the interpretation of the two geometry gaps
in r10. It changes no original input, trial model, FE result, task constraint or
solver engineering choice. The reproducible audit is
`scripts/audit_source_geometry.py`; receipts and annotated source crops are in
`scripts/evidence/r11/`. No native analysis is needed to reproduce it.

## The 92 by 32 m statement and the 80.11 by 30.61 m trial

The original `benchmark_model_spec_en.md`, geometry section, says the inner
plan is **approximately 92 m by 32 m**. It does not name measurement endpoints,
principal axes or a coordinate polygon. The six original raster figures do not
add a metric plan scale or a dimension line registering these endpoints.
`figure_index_en.md` prioritizes the section and numerical geometry view for
reconstruction and uses the monitoring figure for approximate instruments.
Although its combined-plan/section filename suggests otherwise, figure 2.4/2.5
in the preserved bundle contains the section only. The zone and monitoring
figures still show the irregular plan and member topology.

The **80.1123 by 30.6143 m** number is independently reproduced as the
minimum-area enclosing rectangle of our r09 four-vertex trial. It is not a
dimension transcribed from a source polygon. That trial selected monitoring
image pixels (397,251), (649,251), (944,523), (640,546), then assumed
`x=(pixel_x-145)*115/844`, `y=(722-pixel_y)*89/644`. No original dimension line
assigns 115 and 89 m to those image bounding-box axes. The trial's largest
vertex separation is 83.4747 m, which proves only that this selected region
under this selected calibration cannot have a 92 m straight length.

Independent red-pixel extraction finds the two dashed source cuts. Their
orthogonal line-fit residuals are 0.885 and 0.923 pixels after excluding the
central text. The trial's two diagonal sides follow those cuts. Its two other
sides, the top horizontal closure and the lower oblique closure, cut across
open drawing areas; they are not source wall lines or red boundaries. See
`trial_closure_on_source.png`, with these trial closures marked orange on the
unscaled original. Thus the four vertices do not establish the complete
source inner-pit perimeter.

Following the existing outer contour at the open ends between those red cuts
is a different plausible region interpretation. As a diagnostic only, using
the same old calibration without any stretch gives an enclosing rectangle
106.1053 by 32.9258 m, rather than 80.1123 by 30.6143 m. This alternative is
not a proposed replacement polygon and is not exported as native input. The
numerical geometry view also shows a long lowered central branch, but its
perspective view has no metric scale with which to settle the 92 m endpoints.

**Finding:** a contradiction between two independently dimensioned source
inputs has not been demonstrated. Region selection and metric registration
were not independently established in the preparation. Neither 80.11 nor
106.11 m may override the approximately 92 m source statement. The precise
historical dimension endpoints cannot be recovered uniquely from this bundle;
this does not prevent a documented engineering reconstruction under the
original approximate-geometry task. Do not stretch the trial to 92 m, publish
our traced coordinates as the solution, or require a surveyed CAD file.

## The +6.50 to +7.00 m gap

The section explicitly shows crest +7.00 m, the first support/crown reference
at +6.50 m, and the 500 mm offset between them. Its 33500 mm outer-wall length
terminates at -26.50 m. These annotations are mutually consistent. The crop
`section_crest_and_first_support.png` retains the crown beam (圈梁), member
outline and centreline, making the distinction visible. The +6.50 m depth datum
also follows from the source depths and signed excavation-base elevations.

The retained native outer shell really ends at +6.50 m; the generator explicitly
caps soil levels and wall faces there. This is a preparation representation
choice, not an OCR mistake or merely a missed profile sampling point. The
native file also contains 205 first-level beam elements at the +6.50 m axis,
with finite section properties: area 1.2 m², I22 0.144 m⁴ and I33 0.1 m⁴. The
trial declares a 1.0 by 1.2 m section. These are trial choices, not evidence
that the source crown geometry or its coupling has already been validated.

R10's `Wall profile gap at CX1: 6.5 to 7 m` is therefore a correct rejection by
its **shell-only reader** when asked to sample up to +7.00 m. It does not,
on its own, prove that the physical top structure is absent: a line beam's
axis is not its section envelope. The source does not mandate a particular
shell/beam partition. A compliant representation might extend a wall shell,
or account for a finite crown section and offsets in its native kinematics.
Review must check geometry, stiffness, selfweight and coupling without double
counting the crown. Sampling physical points from a beam representation needs
native section orientation, translation and rotation; zero extrapolation is
not an acceptable substitute.

**Finding:** there is no missing source elevation here. The residual issue is
the prepared wall/crown idealization and the reader's supported representation.
R10's proposal to add exactly 0.50 m of separate shell is one implementation
option, not a new engineering requirement imposed on every Agent model.

## Necessary supplement audit and final migration direction

No fourth mandatory numerical supplement is established by this audit. Retain
the three already explicit decisions: dimensioned embedment rather than an
undefined exact n ratio, dependency-correct inner-system activation, and the
all-DBC/all-CX observation convention. Original approximate dimensions, material
data, depth/water milestones and five positions remain unchanged.

The Agent interprets the drawings, registers physical monitors and chooses
reasonable mesh/domain, unspecified rock stiffness, undimensioned sections and
hydraulic spatial treatment. Source review occurs before response inspection.
It checks those choices against the original region, dimensions and topology,
without substituting a hidden polygon, coordinate table or numerical answer.
Different adequate meshes and equivalent native idealizations must be accepted.

The concrete migration design is to keep installed native Kratos as the FE
engine; freeze each submitted declarative native model; review source/monitor
registration; run its five independent histories with the allowlisted stage
driver; independently read its own native fields and physical monitor outputs;
then apply the original delivery checks, tolerances and weights. The existing
pos0 completion and reader controls demonstrate parts of this path, not task
acceptance. Validate the integrated path on one complete five-case family and
exercise a different legitimate discretization with a bounded native control.
A second complete five-case family is not mandatory. Do not require identical
responses between different engineering models.

The only precise source-bound limitation remaining here is the absence of
surveyed plan coordinates and explicitly registered 92/32 m dimension endpoints.
Those would be needed for exact historical-geometry equality, which this task
does not require. They are not grounds to request additional mandatory input
or replace Agent judgment. Current release blockers are unfinished native
adapter/replay integration and validation, not a newly proved source-data
contradiction. Keep the existing release guard; no further full replay is
needed to settle this bounded design audit.
