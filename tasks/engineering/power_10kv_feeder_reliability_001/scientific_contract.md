## Scientific model: normal-affine-v2.1.0

Compute normal-configuration reliability for the entire selected feeder under
independent rare single-contingency events, selective isolation, and additive
annual expected interruption counts and durations. No alternate supply, tie
closing, temporary bypass, overlapping outages or backup-protection failure is
part of this planning model.

The top-level numeric indices represent recorded-line-exposure subtotals.
Missing line lengths remain unknown physical quantities. The complete model
consists of those subtotals and the per-kilometre coefficients requested below;
no imputed length or completed point estimate is requested.

### Source and topology conventions

XML connectivity is authoritative; the SVG is schematic. Equipment identity
is its RDF ID. Connectivity is defined by Terminal.ConductingEquipment and
Terminal.ConnectivityNode. Select the unique Feeder with
Feeder.IsCurrentFeeder=true. Membership includes either
Equipment.EquipmentContainer or Equipment.EquipmentBelongtoFeeder references.
The output feeder is IdentifiedObject.name.

The feeder voltage is the common ConductingEquipment.BaseVoltage of its assigned
ACLineSegment objects. Conducting equipment comprises ACLineSegment, Breaker,
Disconnector, Fuse, BusbarSection and Junction at that voltage, including the
energizing substation's equipment through Equipment.EquipmentContainer,
VoltageLevel.Substation and Bay.VoltageLevel. Other feeders are not sources.

Each assigned PowerTransformer is one equally weighted load point. Its feeder
connection is the TransformerEnd.Terminal of the unique PowerTransformerEnd
with matching TransformerEnd.BaseVoltage. Transformer low-voltage terminals
are outside this feeder's conducting graph.

Normal states are Switch.normalOpen for Breaker, Disconnector and Fuse.
Closed conductors connect their terminals. The source is
Feeder.NormalHeadTerminal when present, otherwise the unique feeder-voltage
BusbarSection in Feeder.NormalEnergizingSubstation. Include the station feeder
breaker. All assigned lines and loads must be source-connected; ambiguous or
missing connectivity/state records are input errors.

### Sections and outage assumptions

Sections are the connected conducting regions separated by Breakers, within
the source-connected normal-state conducting graph only. Closed
disconnectors and fuses remain within a section and have no separate failure
rates. Include source-busbar, zero-line and zero-load sections, including
connectivity-node-only regions. Each line and transformer belongs to one section.

Use a section's lexicographically smallest connectivity-node RDF ID without '#'
as its canonical name. The separate section field may use any consistent,
locally unique identifier.

A line fault or scheduled outage isolates its entire section. All loads that
lose source connection are affected, including downstream loads. Other loads
remain uninterrupted. A breaker fault means loss of conduction with successful
selective isolation. A transformer fault interrupts only that transformer.

Include one switch row per Breaker, including open ties and the station feeder
breaker, named by RDF ID with device_count=1. Attribute it to the energized
endpoint section farthest from the source in shortest-path closed-breaker hop count; ties use
the lexicographically smallest canonical section name. Include one transformer
row per section with transformers, with its local count and affected_users=1.
The device type strings are exactly switch and transformer.

### Parameters and reported quantities

ACLineSegment IdentifiedObject.Length is in metres. Output lengths are in
kilometres. PSRType PD_10100000 means overhead; PD_20100000 means cable.
Retain both types within mixed sections. A missing length does not remove the
line's connectivity. Numeric lengths must be finite and nonnegative.

Emit one fault row and one scheduled row per section, including zero-exposure
or zero-affected-load sections. Use the type-specific rates and durations in
params.json. Rates are annual; line rates are per kilometre-year. Manual
isolation applies to line faults and breaker faults in addition to their repair
duration. Transformer faults use transformer repair duration only. Scheduled
outages use their own type-specific duration without extra isolation time.

In fault_rows, length_km is recorded length, lambda_i the section's annual
line-fault rate, perm_users the affected-load count, t_iso_h manual isolation
time and r_i_h total restoration duration. lambda_N is annual interrupted-load
count; lambda_N_r is annual interrupted-load hours. scheduled_rows uses the
same interrupted-load quantities and users for affected-load count.
device_fault_rows reports device_count, lambda_per_device, affected_users and
t_repair_h. For switch rows, t_repair_h is the elapsed restoration time,
including manual isolation. For transformer rows, it is the repair duration
only.

SAIFI and SAIDI use the equally weighted transformer population N_T. Suffixes
F, D and S designate line faults, device faults and scheduled outages.
Report hours in _h fields, minutes in _min fields, and ASAI as an availability
fraction on an 8760-hour year. CAIDI is zero when expected interruption count
is zero. Retain the exact top-level and row fields from the task prompt.

### Data quality interface

Include data_quality with exactly these fields:

- assessment_scope: the string recorded_line_exposure_subtotal;
- line_exposure_complete: true exactly when no assigned line length is missing
  or whitespace-only; a recorded numeric zero is not missing;
- missing_length_line_ids: all affected ACLineSegment RDF IDs without '#',
  each exactly once;
- missing_length_count: the integer size of that set;
- recorded_length_km_by_type: exactly overhead and cable, with recorded
  kilometres including zero for an absent type;
- full_feeder_point_estimate_available: true exactly when all missing-line
  coefficients are zero, including the case of no missing lines.

### Unknown-length coefficient interface

Include missing_exposure_terms with one row per missing-length line, including
zero-coefficient rows, using exactly:
line_id, line_type, SAIFI_F_per_km, SAIDI_F_h_per_km, SAIFI_S_per_km,
SAIDI_S_h_per_km.

line_id is the RDF ID without '#'; line_type is overhead or cable. The four
finite nonnegative coefficients describe the contribution of one unknown
kilometre to the corresponding population-normalized index, under the section
outage model above. Device-fault contributions have no line-length dependence.
No assumed length or expression-string fields are requested.

Object-key and row order are immaterial. IDs and semantic rows are unique.
Integral JSON numbers are accepted for counts. Strings may have surrounding
whitespace. Boolean, NaN and infinity are not numerical measurements.
