You are a geotechnical engineer comparing inner support elevations for the
A-zone staged three-dimensional pit-in-pit excavation. Build the model from
the supplied engineering sources, calculate the five construction histories,
and explain how support elevation affects settlement and outer-wall movement.

## Inputs and environment

The engineering sources are in `{input_dir}`: `benchmark_model_spec_en.md`,
`engineering_assumptions_en.md`, `A_zone_structural_materials_en.md`,
`task_specific_brief_en.md`, `figure_index_en.md`, its six figures, and
`answer_template.json`. These original files are frozen.

`model_definition.md` specifies the adaptation's modeling requirements and
resolves ambiguous source wording. `benchmark.json` records the source facts
and these explicit decisions. `native_interface.md` defines the supported
model inputs and result formats; `delivery_schema.md` defines the deliverables.

Kratos Multiphysics 10.4.3 with GeoMechanics, StructuralMechanics and
LinearSolvers is installed at `/opt/ale-kratos/bin/python`. GUI use is optional.
The native model interface is support-native-family-1.

## Task and resources

Compare `pos_0m` through `pos_4m`, measured downward from the inner-pit top at
absolute elevation -10.10 m. Preserve the source materials, geometry,
construction history, groundwater milestones and support system under the
clarifications in `model_definition.md`. All other modeling choices and the
monitor registration must be consistent across the five cases. Document
coordinate datums and assumptions. Geometry, mesh, rock stiffness and other
choices left open by the sources remain your engineering responsibility.

Report the largest downward settlement over DBC1 through DBC10 and the largest
horizontal displacement magnitude over the full CX1 through CX10 wall profiles,
in millimeters, relative to the end of Stage 1.

The task VM has 8 vCPU and 30 GiB RAM. The solving window is 18000 seconds
(5 hours). Independent native replay has a shared 6600-second (110-minute)
wall-clock budget, including the common Stage 7 prefix, five parallel case
suffixes, native state audits and result extraction. Each single-threaded
worker has one CPU and a 4 GiB memory cap. Five suffix workers and a 4 GiB
coordinator allowance share 24 GiB with swap disabled. There is no element-count
cap. Geometry, local refinement, physical and convergence requirements remain
in force. Exceeding the published replay time or confirmed worker memory cap
receives zero. Infrastructure/code faults or inability to provide this
allocation are evaluation errors, not solver failures.

## Deliverables

Put all work products under `{output_dir}`:

- `answer.json`, with the fourteen fields in the original template;
- `case_summary.csv`, containing all five cases;
- `model_family_support_position/`, with editable native model inputs, meshes,
  materials, stages, logs, native results and raw monitoring tables for all cases;
- `pos_0m_output_view.png`, `pos_2m_output_view.png`, `pos_4m_output_view.png`;
- `settlement_vs_support_position.png`, `lateral_disp_vs_support_position.png`;
- `support_position_engineering_memo.md`, explaining the comparison, percentages,
  minima, assumptions, limitations, installation tradeoffs and recommendation,
  with rerun instructions and the `support-decision` JSON block specified in
  `delivery_schema.md`.

The model and reported results must be independently reproducible within the
published resource budget. Additional mesh/domain studies are optional.
