# Memory experiment

Implement the simulation described by `cc_codes_quits_extraction.tex` for all
rows of `simulation_grid.csv`. The notes supply the code definitions and
scientific meaning of the reported rates. The computational workflow is yours
to implement. The bundled `fd_wrapper.FastBpOsd` is a low-level decoder, not a
simulation driver.

Use QUITS v1.0.0 (commit `1ccfccf58340ec5e1dce222c2cbee9effae9de6f`), Stim
1.15.0 and NetworkX 3.6.1. The runtime manifest supplies the remaining versions.
The experiment is Z-basis memory with the QUITS cardinal layout, construction
seed 1, its default qubit/detector/observable labeling, noisy zeroth round,
noiseless final data measurement and Z-basis detectors. Each of the four
QUITS ErrorModel probabilities (`idle_error`, `sqgate_error`, `tqgate_error`,
and `spam_error`) equals the row's `p_physical`.

The direction order is E, N, S, W. Within each direction, apply NetworkX's
DSATUR greedy coloring to the CNOT conflict graph, with vertices enumerated
in QUITS's edge-list order and colors scheduled in ascending order. Two
vertices conflict when their gates share a qubit. These conditions define
the physical gate schedule and idle-noise exposure; the construction is not
provided as an executable helper.

Decode the full, undecomposed detector error model with parallel min-sum BP,
scaling factor 0.625, at most 200 iterations, and OSD-CS order 5. The supplied
decoder can be used directly. A shot fails when any decoded logical observable
differs from the sampled logical observable. Use all logical observables.

The decoder representation is one binary fault variable per `error` instruction
in `circuit.detector_error_model(decompose_errors=False).flattened()`, in that
iteration order. Each variable supplies one column of the detector matrix and
one column of the logical-observable matrix, with all of that instruction's
detector and logical targets and its unchanged error probability as the prior.
Retain faults with no detector targets. Do not merge columns, even when their
detector targets or their complete detector/logical effects coincide: the
variable representation affects approximate BP+OSD decoding.

Run 5,000 independent shots for each row. Choose and record separate sampling
seeds for each row. Sampling may be batched or parallelized; seed values and
batch sizes are not prescribed. A sample's integer failure count is an
observation, not an invariant of the experiment.

The CSV's three rates describe, respectively, total shot-failure probability,
the equivalent independent per-round probability, and the equivalent
independent per-logical-qubit per-round probability, as defined in the notes.

Keep your implementation in the delivered `source/` directory with
`source/simulate.py` as its entry point. Save each actual
Stim circuit used in `circuits/row_000.stim` through `circuits/row_059.stim`,
where the zero-based index follows the input grid. Retain QUITS's detector,
observable and qubit labels; comments, repeat blocks, and instruction grouping
may differ without changing the experiment. Record sampling seeds in
`simulation_manifest.json` as an object with `sampling_seeds`, a list of 60
distinct nonnegative integers less than 2**64 in grid order.
