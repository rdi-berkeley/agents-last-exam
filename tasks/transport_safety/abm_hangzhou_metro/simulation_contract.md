# Hangzhou Simulation Contract v2

Implement the following stochastic, capacity-constrained, one-day passenger
simulation. This is a newly explicit benchmark model, not a reconstruction
of an undocumented historical simulator. Use the unchanged AFC demand,
directional station sequences, GIS and operation parameters. Do not fit to
observed end times. Those labels are used only for the two observed CSV
columns and the validation report. No hidden passenger table is an oracle.

## Geometry and Time

Use `line_id` as a separate directional service, including parallel services
between the same station pair. Order each service by `station_order`.
Match GIS features by `linename = line_name_cn + "(" + direction + ")"`,
and station points additionally by `stationnames`. Use geometry coordinates.

Project each station point onto its matching line polyline in raw longitude,
latitude degrees: for each segment a,b, let f be dot(point-a,b-a)/|b-a|^2
clamped to [0,1] (zero for a zero-length segment). Choose the projection
with minimum squared coordinate distance, breaking ties by smaller chainage.
Chainage is the sum of preceding segment lengths plus f times that segment's
length. Segment lengths use the haversine formula on a sphere of radius
6,371,008.8 meters. Successive stations must have increasing chainage.
This is an explicit local-map approximation, not an empirical speed fit.

Use the configured one-minute grid. Running time on each adjacent segment
is max(1, ceil(chainage_difference / (metro_speed_kmh * 1000 / 60))) minutes.
Trains dwell for `station_stop_time_minutes` at intermediate stations.
There is no dwell before a terminal's initial departure or after final arrival.
Access from entry admission to platform takes one minute, and final egress
takes one minute. A transfer takes `transfer_time_minutes` from arrival to
readiness on the next platform, without adding access or egress again.

## Routes and Randomness

Enumerate at most `k_shortest_paths` node-simple paths using **NetworkX 3.6.1**
`shortest_simple_paths(..., weight="weight")`. The directed routing graph
has physical station names as string nodes, plus one tuple node
`(line_id, zero_based_segment_index)` per directed segment. First insert all
station nodes in Unicode lexicographic order; then all tuple nodes in sorted
tuple order. In sorted tuple order, insert origin-station -> tuple-node with
weight running-time plus dwell, then tuple-node -> destination-station with
weight zero. This preserves parallel services while prohibiting station loops.
Use the first k paths in the generator's order, including its deterministic
tie order. Origin equal to destination instead has one empty route.

Merge consecutive segments of the same directional service into a ride leg.
The route's free-flow time T is the sum of running times plus intermediate
dwell within legs plus transfer-time times (number of legs minus one).
There is no dwell at the final station of a leg. Waiting, access and egress
are not in T. Transfers F equal max(0, number of legs minus one).
Compute utilities with the provided multinomial-logit coefficients:
U = logit_epsilon + logit_beta_travel_time*T + logit_beta_transfer_times*F.
Normalize exp(U-max(U)); the common epsilon cancels. Do not select only the
modal route or recalibrate coefficients on AFC outcomes.

Choose any unsigned 64-bit integer seed and declare it in the manifest.
For each input data row, indexed from zero in original CSV order, SHA-256
hash the ASCII string `hangzhou-discrete-event-v2:<seed>:<row_index>`.
Interpret its first eight bytes as an unsigned big-endian integer and divide
by 2^64 to get u. Select the first route whose cumulative normalized
probability is strictly greater than u (last route on floating-point overflow
at the upper endpoint). Use ordinary binary64 arithmetic. Row order and seed
are the complete RNG contract; no global random state or card-id hashing.
Different seeds are valid and may produce different route choices and outputs.

## Trains, Entry, and Queues

Interpret all operation times on the extended day, never modulo 1440.
Each directional service starts with no passengers and its first nominal
terminal departure at operation start. Recursively add the headway applicable
at the previous **unrounded nominal departure**. Peak-period intervals are
left-closed/right-open, tested in JSON order; otherwise use the default.
Include nominal departures through operation end inclusive. Round each actual
terminal departure upward to the next integer minute, without resetting phase
at a period boundary. Trains continue to their terminal after service closes;
there are no previous-day or extra next-day trains. Intermediate arrivals and
departures follow the running times and dwell above.

The supplied timetable must have successive departures separated by at least
dwell + max(1, ceil(block_length_meters / (metro_speed_kmh*1000/60))). With
identical segment running times and no overtaking, this conservative clearance
bound holds along each directional service. Parallel services use separate
tracks. There is no train delay from boarding volume in this fixed-dwell model.

Per physical station, admit entrants FIFO by (AFC start_time, input row index).
Admission starts at operation start; earlier entrants wait outside. Admit at
most `station_entry_flow_limit_per_minute` in each integer-minute bucket
(zero means unlimited). Carry excess entrants forward without skipping them.
Stop new admissions after operation end inclusive. Access starts on admission.

Every platform queue is specific to (directional service, station). Its FIFO
priority is (platform-ready minute, input row index). Assigned routes do not
change. At a train arrival, alight all passengers whose current leg ends there;
they either exit or become transfer-ready after the configured walk. At each
departure, board ready passengers in FIFO order until the queue is exhausted
or the configured train capacity is reached. Never skip a ready passenger
when a seat exists. Alighting frees capacity before boarding. Boarding is
allowed when readiness equals departure time.

Process all arrivals at a minute before any departures at that minute; within
either phase use (line_id, zero-based terminal departure number, station index)
order. Initial entrants may be placed in queues in advance with their readiness
times. A same-station trip finishes two minutes after admission without a train.
After all trains finish, queued and never-admitted demand is unserved. Do not
invent times for those trips. Every admitted passenger must be completed or
still queued, every boarding must have an alighting, and loads must remain
between zero and capacity.

## Deliverables and Evaluation

Write `simulation_manifest.json` containing exactly:
`{"contract_version":"hangzhou-discrete-event-v2","seed":42}`
(42 is an example; any unsigned 64-bit seed is allowed).

Write `passenger_records.csv` with exactly the existing nine columns, one row
for **every completed trip and no other trip** under your declared seed.
CSV row order is free; identities must be unique. Times and counts are integers.
end_time_simulate is the extended-day completion minute including egress;
duration_simulation = end_time_simulate - start_time; transfer_times = F.
Echo AFC end_time as end_time_real. duration_real = end_time - start_time,
adding 1440 once if negative. Never wrap simulated completion modulo 1440.

Write the existing four `validation_report.txt` metrics on the completed rows:
R2 = 1 - sum((sim-real)^2)/sum((real-mean(real))^2), or zero when the denominator
is zero; RMSE = sqrt(mean((sim-real)^2)); Total passengers = completed row count;
std(sim-real) uses population variance (ddof=0). All metrics must be finite.
Use the original report line names and units. Tolerances are 0.001 for R2,
0.02 minutes for RMSE and standard deviation, and exact passenger count.
The scatter plot remains optional and unscored.

Evaluation recomputes this model from the public inputs and your declared seed.
It requires exact integer dynamics and conservation, not distance to a hidden
answer or distance from observed labels. The original 80% minimum input coverage
remains a dataset feasibility check; omitting completed passengers is invalid
even above 80% or 95%. Close agreement with observations is not proof of copying;
copying or fitting labels is prohibited because it does not implement this model.
