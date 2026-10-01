## Scientific Model: normal-affine-v2.1.0

Compute the whole-feeder normal-configuration, selective-isolation model as
recorded-exposure intercepts plus exact affine terms for unknown line lengths.
The existing top-level numeric indices are recorded-exposure subtotals, NOT
complete feeder point estimates when missing lengths have nonzero coefficients.
The intercepts together with the mandatory missing_exposure_terms represent the
complete model without imputing physical measurements. This is a specified
planning model, not a reconstruction of historical outages.
Use independent, rare, single-contingency events and additive annual expected
interruption counts and hours; do not model overlapping outages.
Blank line lengths are genuinely unknown. Their exposure is excluded from the
reported subtotal, not inferred to be physically zero. Do not estimate lengths
from schematic SVG coordinates, neighboring spans, or equipment names.

### Topology and Source

- Use RDF IDs and Terminal.ConductingEquipment / Terminal.ConnectivityNode links
  in the XML, not names, drawing proximity, or terminal ordering. The SVG is a
  schematic cross-check; XML connectivity takes precedence. Do not add edges from
  SVG GLink_Ref metadata. Equal names do not identify equal equipment.
- Select the unique Feeder with Feeder.IsCurrentFeeder=true. An equipment object
  is assigned if either Equipment.EquipmentContainer or
  Equipment.EquipmentBelongtoFeeder references that feeder. N_T counts every
  assigned PowerTransformer once, with equal weight; EnergyConsumer is not a
  second load population. Set the output feeder to IdentifiedObject.name.
- The feeder voltage is the common ConductingEquipment.BaseVoltage reference of
  its assigned ACLineSegment objects. Include assigned ACLineSegment, Breaker,
  Disconnector, Fuse, BusbarSection and Junction objects in the conducting graph.
  Also include these classes at that voltage in Feeder.NormalEnergizingSubstation,
  following Equipment.EquipmentContainer, VoltageLevel.Substation and
  Bay.VoltageLevel containment transitively. Do not include other feeders as
  alternate sources. Ignore other equipment classes for connectivity and failure
  exposure except for the transformer loads described below.
- A transformer is a terminal load, not a through-connection between voltage
  levels. Select its unique PowerTransformerEnd whose TransformerEnd.BaseVoltage
  equals the feeder voltage; attach one load to that end's TransformerEnd.Terminal
  connectivity node. Never connect its low-voltage terminal into the feeder graph.
- Use Switch.normalOpen for all Breaker, Disconnector and Fuse states, even when
  Switch.open disagrees. Normal-open devices do not connect their terminals.
  Closed conductors join all their terminals. No tie closing or transfer is
  modeled. The supplied normal states, not an invented spanning tree, define
  which paths conduct.
- If Feeder.NormalHeadTerminal exists, its connectivity node is the source.
  Otherwise use the unique feeder-voltage BusbarSection in the energizing
  substation. Include the station feeder breaker in device exposure. Do not
  discard the first feeder-side section or its loads.
- Retain the source-connected normal-state graph. Every assigned line and
  transformer must be reachable. Disconnected assigned lines/loads, ambiguous
  source/transformer ends, missing normal switch states, and unsupported line
  types are data errors, not grounds to invent edges or silently drop loads.

### Sections and Attribution

- Form a bipartite graph of conducting-equipment IDs and connectivity-node IDs,
  with transformer IDs attached only at their feeder-voltage end. Remove all
  Breaker vertices from the energized graph. Each remaining connected component
  is one section. Closed disconnectors and fuses stay inside sections; they have
  no separate failure-rate parameters. Include zero-line, zero-load and source
  busbar sections, including components consisting solely of connectivity nodes.
- Each line and transformer belongs to exactly one section. A section's canonical
  name is its lexicographically smallest connectivity-node RDF ID, without '#'.
  Use this name in fault_rows, scheduled_rows and transformer device rows. The
  section field may instead use any consistent, locally unique identifier.
- For a section outage, isolate the entire section: remove all its vertices from
  the normal graph. Affected loads are all N_T loads no longer reachable from the
  source, including loads inside that section. If the source section is removed,
  all loads are affected. Compute reachability, not traversal-dependent ownership
  or duplicate path counts. This also defines the result for branched or meshed
  graphs. Ideal selective protection leaves all other loads uninterrupted; no
  feeder-wide transient, backup-protection or capacity-limited restoration event
  is added.
- Model breaker faults as loss of conduction with successful selective isolation,
  not stuck-breaker or backup-protection failures. Remove only that Breaker vertex.
  Count loads thereby losing source reachability. A normal-open breaker therefore
  has zero affected loads.
  Make one switch device row per included Breaker (including open ties and the
  station feeder breaker), with name equal to its RDF ID and device_count=1.
- Assign each breaker row to its energized endpoint section farthest from the
  source section in shortest-path closed-breaker hop count on the section graph.
  If distances tie, choose the lexicographically smallest canonical section name.
  For open ties consider only energized endpoints. This reporting attribution
  does not change which loads its failure interrupts.
- Make one transformer device row per section containing transformers, with
  device_count equal to its local transformer count and affected_users=1. Each
  transformer fault interrupts only itself. Omit transformer rows with count zero.
  Device type strings are exactly 'switch' and 'transformer'.

### Exposure and Formulas

- IdentifiedObject.Length on ACLineSegment is interpreted in metres. Divide by
  1000 for kilometres. Map PSRType PD_10100000 to overhead and PD_20100000 to cable
  (the SVG distinguishes solid dxd and dashed 0201 line representations).
  Sum each assigned line exactly once, retaining both types within mixed sections.
  A blank/absent length contributes no recorded exposure, but the line still
  conducts and belongs to a section. Reject negative or non-finite numeric lengths.
- For every section emit one fault row and one scheduled row, even when length
  or affected-load count is zero. Let L_o and L_c be its recorded overhead/cable
  kilometres, and A its affected-load count from section removal.
- Fault row: length_km=L_o+L_c;
  lambda_i=L_o*line_fault.lambda_overhead+L_c*line_fault.lambda_cable;
  perm_users=A; t_iso_h=line_fault.t_manual_isolation_h;
  r_i_h=t_iso_h+line_fault.t_repair_h; lambda_N=lambda_i*A;
  lambda_N_r=lambda_N*r_i_h. Use manual isolation throughout; FTU deployment does
  not establish a different restoration scheme in this assessment.
- Scheduled work isolates the same entire section, without an alternate supply
  or temporary bypass, so users=A, not just the local load count. For each line
  type k, let q_k=scheduled_outage.lambda_scheduled_k and
  d_k=scheduled_outage.t_scheduled_k_h. Then length_km=L_o+L_c,
  lambda_N=A*sum(L_k*q_k), and lambda_N_r=A*sum(L_k*q_k*d_k).
  Do not apply the overhead duration to cable exposure or add isolation time to
  the scheduled duration.
- Device row: lambda_per_device comes from device_fault.lambda_breaker or
  device_fault.lambda_transformer; lambda_N=device_count*lambda_per_device*
  affected_users; lambda_N_r=lambda_N*t_repair_h. For switches, t_repair_h is
  device_fault.t_repair_breaker_h plus the manual isolation time. For transformers
  it is device_fault.t_repair_transformer_h, without additional isolation time.
- For fault, device and scheduled tables respectively (suffixes F, D, S),
  SAIFI_suffix=sum(lambda_N)/N_T and SAIDI_suffix_h=sum(lambda_N_r)/N_T.
  Add the three components to obtain SAIFI and SAIDI_h; CAIDI_h=SAIDI_h/SAIFI
  (zero if SAIFI is zero); ASAI=1-SAIDI_h/8760. Every corresponding _min field
  is 60 times its hours field. Retain precision until serialization.
- Rates are annual; line rates are per kilometre-year. The index definitions
  retain their usual meaning within this restricted model. Missing positive
  line exposure can only increase its SAIFI/SAIDI and decrease its ASAI;
  CAIDI is not a bound. No numerical completion of missing physical lengths is
  requested or scientifically justified by the supplied schematic.

### Mandatory Data Quality

The output must include a `data_quality` object with exactly these fields:

- `assessment_scope`: the string `recorded_line_exposure_subtotal`, describing
  the existing top-level numeric fields even when all lengths are recorded.
- `line_exposure_complete`: JSON boolean true exactly when no assigned line has
  an absent or whitespace-only length. A recorded numeric zero is not missing.
- `missing_length_line_ids`: a list of all such assigned ACLineSegment RDF IDs,
  without '#', each exactly once. Compute the IDs from the input; order is ignored.
- `missing_length_count`: the exact integer count of those IDs.
- `recorded_length_km_by_type`: an object with exactly `overhead` and `cable`,
  giving each type's total recorded kilometres, including zero for an absent type.
- `full_feeder_point_estimate_available`: JSON boolean true exactly when every
  missing-line coefficient defined below is zero, including the case of no
  missing lines. Otherwise false: the complete model is identified as a function
  of unknown lengths, but its numeric point estimates are not identified.

When all lengths are recorded, both flags are true, the ID list and coefficient
table are empty, and the scalar intercepts equal the complete model's indices.
When lengths are missing but all their coefficients vanish, line_exposure_complete
remains false while full_feeder_point_estimate_available is true.

### Full-Model Affine Terms

Include `missing_exposure_terms`, a list with exactly one row per missing-length
line, including lines with zero affected loads or zero rates. Rows have exactly:
`line_id, line_type, SAIFI_F_per_km, SAIDI_F_h_per_km, SAIFI_S_per_km, SAIDI_S_h_per_km`.
`line_id` is the line's RDF ID without '#'; `line_type` is `overhead` or `cable`.
All four coefficients are finite nonnegative JSON numbers. Do not include an
imputed length, instance assignment, or additional field in these rows.

For missing line j of type k in a section with A_j affected loads, define:

```text
a_Fj = SAIFI_F_per_km   = A_j * line_fault.lambda_k / N_T
b_Fj = SAIDI_F_h_per_km = a_Fj * (line_fault.t_manual_isolation_h + line_fault.t_repair_h)
a_Sj = SAIFI_S_per_km   = A_j * scheduled_outage.lambda_scheduled_k / N_T
b_Sj = SAIDI_S_h_per_km = a_Sj * scheduled_outage.t_scheduled_k_h
```

Here k selects the actual overhead/cable parameter suffix. The population A_j
comes from section isolation and source reachability, not merely local loads.
The coefficients are per unknown kilometre, already normalized by N_T, not row
numerators. Coefficient rows are matched by line ID, independent of row ordering.
Neither the public prompt nor the SVG provides a precomputed coefficient table.

Let x_j be the unknown nonnegative physical length of line j in **kilometres**.
Use a subscript 0 below for the reported recorded-exposure scalar intercept.
The required output represents these exact full-model functions:

```text
SAIFI_F(x)   = SAIFI_F0   + sum_j(a_Fj * x_j)
SAIDI_F_h(x) = SAIDI_F_h0 + sum_j(b_Fj * x_j)
SAIFI_S(x)   = SAIFI_S0   + sum_j(a_Sj * x_j)
SAIDI_S_h(x) = SAIDI_S_h0 + sum_j(b_Sj * x_j)
SAIFI_D(x)   = SAIFI_D0
SAIDI_D_h(x) = SAIDI_D_h0
SAIFI(x)     = SAIFI0    + sum_j((a_Fj + a_Sj) * x_j)
SAIDI_h(x)   = SAIDI_h0  + sum_j((b_Fj + b_Sj) * x_j)
CAIDI_h(x)   = SAIDI_h(x) / SAIFI(x), or zero when SAIFI(x) is zero
ASAI(x)      = 1 - SAIDI_h(x) / 8760
```

Every hours-to-minutes conversion, including component SAIDI and CAIDI, is a
factor of 60 applied to the corresponding full function. Device contributions
are unchanged because this model gives them no line-length dependence. Full
CAIDI is a ratio of affine functions, not the subtotal CAIDI plus linear terms.
Do not choose numeric x_j values or report fabricated full-model scalar totals.
No separate expression-string fields are required: intercepts, coefficient rows,
units and these generic equations completely specify the full model.

### Output and Scoring

Use the exact top-level and row fields listed in the task. No extra version field
is required in the answer. Row ordering and the literal section identifier are
ignored in one-to-one content matching. Each scalar/cell has equal weight. Numeric
comparisons retain 5% relative tolerance (absolute 1e-9 near zero); ASAI uses
absolute 1e-4. These tolerances do not authorize a different topology, partition,
load population or outage model. Scope, ID sets, counts and completeness flags
are exact semantic checks; counts may use equivalent integral JSON number
formats. Numeric coefficients and recorded-length subtotals use the same 5%
relative / 1e-9 absolute tolerance. No boolean, NaN, infinity or negative number
is a valid coefficient or length. Object-key order, list order and surrounding
whitespace in strings do not matter; duplicate IDs, missing or extra rows, and
extra fields in data_quality, its length object, or coefficient rows are invalid.

The original scalar/section-cell leaf score retains its weights and denominator;
data_quality and missing_exposure_terms add no scored leaves. They instead form
a mandatory scientific gate. The returned score is the original leaf score when
the gate passes and zero when it fails, with the ungated leaf score retained in
evaluation diagnostics. Full pass requires both the scientific gate and all
original leaves to pass. Thus a numerically correct subtotal without its scope,
complete missing-data inventory and correct affine completion cannot pass.
