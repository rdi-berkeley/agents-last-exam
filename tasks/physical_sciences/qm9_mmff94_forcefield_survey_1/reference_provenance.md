# QM9 survey reference provenance

The original QM9 XYZ archive contains 133,885 entries. The authors' list of
3,054 geometries failing their consistency checks defines the 130,831-entry
survey population. `assets/uncharacterized_ids.json` contains those published
IDs, not a task-specific selection of favorable molecules.

- Dataset: <https://doi.org/10.6084/m9.figshare.c.978904.v5>
- Exclusion list: <https://doi.org/10.6084/m9.figshare.978904_D10>
- Source file: Figshare article 1057644, file 3195404, `uncharacterized.txt`.
- Source SHA-256: `3aa5115d540b356de94791d4a74c3bf1ed91c469ecf52a4f5d7cc0506fe02e24`.

`assets/survey_protocol.json` fixes the software version, input SMILES order,
finite conformer search, numerical failure policy, group definitions, PES
initialization and energy convention. These are required because the grader
compares numerical results against a fixed reference. The file contains methods
and schemas, without computed answers or reference conformers.

The corrected reference uses the original absolute-discrepancy criterion in
both directions. All downstream results are regenerated from that candidate
population. Energy differences are computed from the reported energies, with
the artificial restraint contribution excluded consistently. Group labels use
the public SMARTS definitions, and every requested minimum is included.

The scientific work remains the five-phase survey with 200 conformers per
candidate, five PES scans, the original thresholds, and the original phase
weights and numeric tolerances. Parallel processing is allowed, while molecule
seeds and per-molecule numerical settings remain fixed.
