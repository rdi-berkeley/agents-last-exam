You are a geotechnical engineer comparing inner support elevations for the
A-zone staged three-dimensional pit-in-pit excavation. Build the model from the
original inputs, calculate the five alternative construction histories, and
explain how support elevation affects settlement and outer-wall movement.

Read the original files in `{input_dir}`: `benchmark_model_spec_en.md`,
`engineering_assumptions_en.md`, `A_zone_structural_materials_en.md`,
`task_specific_brief_en.md`, `figure_index_en.md`, `answer_template.json`, and
all six figures listed in the index. They remain the engineering sources.
`model_definition.md` states the explicit task clarifications adopted for this
adaptation, including the deterministic admissibility rules; they control open
or ambiguous source wording. The original
files and their hashes are unchanged. `benchmark.json` separates source facts
from these new decisions, without prescribing a solved model.

Kratos Multiphysics 10.4.3 with GeoMechanics, StructuralMechanics and LinearSolvers
is the intended runtime at `/opt/ale-kratos/bin/python`. GUI use is optional.
Use the supplied support-native-family-1 interface, uniform hydrostatic pressure,
bonded walls and a common declared non-crown rectangular beam section.

Preserve the fourteen Mohr-Coulomb strata and their supplied parameters; elastic
C35 walls, supports, ring beams and walers, including their 25 kN/m³ selfweight;
the 1.00 m outer and 0.62 m equivalent inner wall thicknesses; the original
staged excavation, groundwater levels and construction history; and four outer
bracing levels plus the one inner support. Interpret the irregular plan and
diagonal members from the figures. Distinguish the Stage 7 ring beam from an
additional layer of cross-pit outer struts.

Keep beta = 0.30 as the source's rounded depth-ratio label and retain the baseline
excavation depths. For all five cases, fix inner-wall embedment at 6.50 m below
the inner base: the section's absolute base -15.10 m and toe -21.60 m control.
Retain n = 0.40 as a rounded legacy study label only; no exact denominator or
historical n formula is recovered or imposed. This precedence is an explicit
new task clarification, not a claim that the original study defined it.
Compare `pos_0m` through `pos_4m`, measured down from the inner-pit top at absolute
-10.10 m, through each case's final excavation state. Keep all other
modeling choices consistent across the cases. Document coordinate datums and
the interpretation of source ambiguities. Rock stiffness, mesh, sufficiently
remote boundaries and other details left open by the sources remain your
engineering choices. Preserve state through the stages and install each support
when the excavation reaches its installation level. Stage 7 completes outer
excavation; Stage 8 constructs the inner wall and crown/ring beam. Install the
single inner support after its wall exists and its case level is exposed, before
further inner excavation. Position 0 can be installed before inner digging;
positions 1 through 4 require Stage 9 substeps. This explicitly corrects the
source's fifth ring/support event ordering, without adding a fifth outer brace
or duplicating the inner support. All water and final excavation milestones stay.

Use all DBC1 through DBC10 ground stations and all CX1 through CX10 outer-wall
profiles, each from wall top to toe. With upward z and displacement u measured
from the end of Stage 1, report settlement as max_i(max(0, -u_z(DBC_i))) and wall
movement as max_i,max_depth(sqrt(u_x(CX_i,depth)^2 + u_y(CX_i,depth)^2)), in mm.
This new public convention resolves the ambiguous ground-section wording to
DBC station sampling; it does not invent continuous transects or instrument
azimuths. Register the approximate source locations before viewing results,
interpolate native fields there, and check profile sampling convergence. Keep
that mapping and baseline for all five cases, without resetting construction
displacements. Whole-domain maxima and ZQS/ZQC do not replace these observations.

Put all work products under `{output_dir}`:

- `answer.json` with the fourteen fields in the original template;
- `case_summary.csv` with all five cases;
- `model_family_support_position/` containing editable native model inputs,
  meshes, materials, stages, calculation logs, native results for all five cases,
  and the raw monitoring extraction tables;
- `pos_0m_output_view.png`, `pos_2m_output_view.png`, `pos_4m_output_view.png`;
- `settlement_vs_support_position.png`, `lateral_disp_vs_support_position.png`;
- `support_position_engineering_memo.md` explaining results, percentages,
  minima, installation tradeoffs, assumptions, limitations and recommendation.

Include instructions for rerunning and inspecting the native project. There is
no additional mandatory NPZ export, handwritten project manifest or independent
quantitative-proof assignment. See `delivery_schema.md` for the delivery details
and `native_interface.md` for the supplied construction and result-file interface.

The intended weighting remains 50% numerical and 50% engineering delivery.
Numerical checking must use independent replay of your source-compliant model,
with the original response tolerance max(0.30 mm, 5%), rather than numbers from
an arbitrarily fixed alternative model. No reference optimum or response-curve
shape is prescribed. An unavailable replay is an evaluator error, not a zero
score or evidence that your engineering solution failed.

Evaluation is entirely deterministic. Run the supplied native_model_contract.py
preflight before solving. model_definition.md publishes geometric size ranges,
wall/brace coverage, monitor admissibility, local refinement and solver tolerance
checks. These explicit adaptation rules replace subjective source interpretation.
A violated public check receives zero with its failure recorded. Incomplete
independent replay remains evaluator-unavailable, not a solver score.

Include the support-decision JSON block specified in delivery_schema.md inside
the engineering memo. PNG checks cover decoding, minimum dimensions and pixel
variation only; they do not grade image content, labels or physical accuracy.
Memo checks cover native percentages/minima, a valid recommendation position
and minimum text lengths, without semantic assessment of the narrative.
Images and memo contribute 10% and 6% of the total score; the remaining delivery
34% covers schema, completeness, native results and summary/table agreement.
All credit requires passing the public model checks and completed independent
native replay. No model service grades prose, images, water or member choices.
Additional mesh/domain studies are further numerical validation, not mandatory
extra runs. No hidden optimum, registration, mesh or second five-case study is
required. Document uncertainty and installation tradeoffs in the memo.
