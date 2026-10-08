# Deterministic delivery and evaluation contract

Full five-case native entrypoint validation remains pending. No LLM or evaluator
API credential is used. Public model checks are in model_definition.md.
Passing preflight permits independent replay, not a physical certificate.
All numerical and delivery credit requires passing the public model checks
and successful independent native replay of all five cases.

All original root deliverables remain required: answer.json, case_summary.csv,
three pos_0m/pos_2m/pos_4m output-view PNGs, the settlement and lateral-displacement
comparison PNGs, and support_position_engineering_memo.md. The editable
model_family_support_position directory includes native inputs, logs, saved
results and raw DBC/CX monitor tables for all five cases. Use native_interface.md
for filenames and commands. Missing a root deliverable or raw monitor table
receives zero; native execution/runtime failures remain evaluator-unavailable.

answer.json contains the fourteen original fields with finite JSON numbers and
positive response magnitudes. case_summary.csv columns are case_label,
support_position_m,max_settlement_mm,max_disp_mm, with one row per pos_0m..pos_4m.
Order is immaterial; duplicates, nonfinite values and CSV/JSON disagreement fail.

The numerical half is unchanged: thirteen fields at 0.0357 each and
best_support_position_for_minimum_disp_m at 0.0359. Independently replayed
responses use max(0.30 mm, 5%) tolerance. Signed percentages are
(S0-S2)/S0*100 and (S4-S2)/S2*100, rounded to one decimal place, with
max(0.30 percentage point, 8%) tolerance. Reported percentage arithmetic must
also agree within 0.1 point with the reported responses. Minima are tested
positions with the smallest native response; exact native ties are allowed.
There is no preferred response curve or invariant optimum across models.

The delivery half retains schema 0.08, completeness 0.12, native project 0.06,
summary 0.08, images 0.10 and memo 0.06. Native project/summary credit depends
on independent native fields and physical DBC/CX table agreement. Duplicate
shell/crown boundary rows may retain either valid trace; the reported profile
must still retain the native maximum within response tolerance.
Native-result disagreement loses the 0.06 native-project component; raw-table
disagreement or CSV/JSON disagreement loses the 0.08 summary component.
These are partial-credit checks after replay, distinct from the zero-score
missing-deliverable rule above.

## Explicit presentation checks

Each of the five PNGs earns 0.02 when it decodes as PNG, is at least 320x240
pixels, and at least one RGB channel has a decoded range of 16 or more.
These are deliberately limited presentation checks. They do not certify
image semantics, readable labels or physical accuracy. Native files and numeric
tables supply that authority. Provide informative labeled views and plots.

The memo may include arbitrary narrative, but must contain exactly one fenced
JSON block tagged support-decision. The following six checks each earn 0.01:

1. reduction_0m_to_2m_percent and increase_2m_to_4m_percent are finite numbers
   agreeing with native percentages within the numerical tolerance above.
2. best_support_position_for_minimum_settlement_m and
   best_support_position_for_minimum_disp_m are integer native minimizers.
3. recommended_position_m is an integer 0..4 and recommendation_reason is a
   string of at least 40 characters after stripping leading/trailing whitespace.
   A tradeoff recommendation need not equal either single-objective minimum.
4. assumptions is a string of at least 40 characters after stripping.
5. limitations is a string of at least 40 characters after stripping.
6. installation_tradeoff and rerun_instructions are each strings of at least
   40 characters after stripping. Give the case-dependent installation order
   and actual native rerun/inspection commands.

Example structure (replace every value with your own computed result/text):

```support-decision
{
  "reduction_0m_to_2m_percent": 0.0,
  "increase_2m_to_4m_percent": 0.0,
  "best_support_position_for_minimum_settlement_m": 0,
  "best_support_position_for_minimum_disp_m": 0,
  "recommended_position_m": 0,
  "recommendation_reason": "Explain the computed comparison and your engineering recommendation.",
  "assumptions": "Describe rock stiffness, sections, groundwater, interfaces and other model choices.",
  "limitations": "Explain model dependence, sampling, mesh and boundary uncertainty without claiming universal accuracy.",
  "installation_tradeoff": "Explain excavation access and the time at which each alternative support becomes active.",
  "rerun_instructions": "Give executable native model commands and identify the saved result and monitoring files."
}
```

Invalid JSON or an absent/duplicate block earns no memo credit. Fields earn
credit independently. Prose length is a coverage check, not a claim that a
semantic engineering assessment was performed. No extra simulations, force
accuracy certificate or independent proof assignment is required.
