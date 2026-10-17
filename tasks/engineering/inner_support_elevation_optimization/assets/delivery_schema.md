# Delivery schema

Required root files are `answer.json`, `case_summary.csv`,
`pos_0m_output_view.png`, `pos_2m_output_view.png`, `pos_4m_output_view.png`,
`settlement_vs_support_position.png`, `lateral_disp_vs_support_position.png`,
and `support_position_engineering_memo.md`.

The editable `model_family_support_position/` directory contains native inputs,
logs, saved results and raw DBC/CX monitor tables for all five cases. Native
file formats are specified in `native_interface.md`.

`answer.json` uses the fourteen original template fields with finite JSON
numbers and positive response magnitudes. `case_summary.csv` has columns
`case_label,support_position_m,max_settlement_mm,max_disp_mm`, with one row per
`pos_0m` through `pos_4m`. Order is immaterial. JSON, CSV, raw tables and native
results must represent the same calculations.

Report the percentage reduction from position 0 to 2 relative to position 0,
and the percentage increase from position 2 to 4 relative to position 2,
each rounded to one decimal place. Report a tested position attaining each
minimum; an exact tie permits either minimizing position.

The five images must be PNGs of at least 320x240 pixels, showing labeled native
output views and the two support-position comparisons.

The memo must explain the engineering recommendation and contain exactly one
fenced JSON block tagged `support-decision`, with these fields:

- `reduction_0m_to_2m_percent`, `increase_2m_to_4m_percent`: finite numbers using
  the percentage definitions above;
- `best_support_position_for_minimum_settlement_m`,
  `best_support_position_for_minimum_disp_m`: integer minimizing positions;
- `recommended_position_m`: integer 0 through 4;
- `recommendation_reason`, `assumptions`, `limitations`, `installation_tradeoff`,
  `rerun_instructions`: descriptive strings of at least 40 characters each after
  trimming surrounding whitespace. Explain the computed comparison, adopted
  assumptions, limitations, installation implications and executable native
  rerun/inspection instructions.

A tradeoff recommendation may differ from either single-objective minimum.
