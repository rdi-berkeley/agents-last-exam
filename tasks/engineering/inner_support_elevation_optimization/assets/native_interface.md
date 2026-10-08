# Supported native model interface

Setup and evaluation run natively on Linux. The evaluator independently runs
all five support heights using the submitted model inputs and compares its
results with the submitted native fields and monitoring tables.

The current adapter supports Kratos10.4.3 UPw 4-node tetrahedral or 8-node
hexahedral soil, native Mohr-Coulomb layers, elastic C35 Q4 shells and native
linear beams with rigid interpolation connections. Its implemented hydraulic
choice is prescribed uniform hydrostatic drawdown. This is now an explicit
modeling restriction of the deterministic adaptation in model_definition.md.

The installed runtime is `/opt/ale-kratos/bin/python`. Preparation can create
a Python3.10 virtual environment and install the legitimate packages:

```
python3.10 -m venv /opt/ale-kratos
/opt/ale-kratos/bin/pip install KratosMultiphysics==10.4.3 KratosGeoMechanicsApplication==10.4.3 KratosStructuralMechanicsApplication==10.4.3 KratosLinearSolversApplication==10.4.3 numpy scipy
```

Setup verifies original source hashes and the installed runtime; it does not
launch a VM or modify the original sources. It stages the native construction
and result-reading tools under the task software directory. No private witness
mesh, digitization, monitor coordinates, material choices or answers are staged.

Place these native construction inputs directly under
`model_family_support_position/`:

- `model.mdpa`: native soil nodes, positive element connectivity, properties
  and fourteen `Layer01` through `Layer14` submodelparts.
- `Materials.json`: native `Support.LayerNN` material bindings using
  `GeoMohrCoulombLaw3D` and the original material values.
- `family.json`: `support-native-family-1` construction data in m-kN-s and the
  source absolute upward-z datum. Fields are `format`, `units`,
  `ground_elevation`, `domain` (xmin,xmax,ymin,ymax,zmin,zmax), `layers`
  (original id/top/bottom), `outer_polygon`, `inner_polygon`, wall-face
  connectivity in `outer_wall_faces` and `inner_wall_faces`, `braces`,
  `monitors`, `source_hashes`, and `engineering_choices`.

Brace definitions contain `system`, `segments_xy_m` and inner
`crown_segments_xy_m`. `engineering_choices.beam_section_m` declares width and
height for this adapter's common rectangular non-crown section; optional
`crowns` declares each system's axis elevation, shell attachment elevation and
section. Separate non-crown member sections are outside this adaptation.
At a declared crown elevation, members on the submitted wall perimeter use
the crown section; cross-pit members keep the common support section. Lower
perimeter support members also keep the common support section. Collinear
subdivision of the perimeter or beam graph does not change this assignment.
The native audit checks the actual installed properties against these choices.
States computed with different effective member properties cannot be reused
by changing checkpoint metadata; resume before the affected installation.
Physical `monitors` maps all DBC/CX names to your registered native xyz points.
Record your registration and engineering reasoning in `engineering_choices`.
The exact accepted fields and material API are in `native_input_adapter.py`
and `native_family_runner.py`; these inputs drive actual native construction,
not a separate claim of correctness. Choose your own justified mesh, geometry,
rock stiffness and undimensioned sections consistently across all five cases.
There is no private family hash requirement. A deeper-domain continuation of
the last listed rock can be explicitly declared; it is not recovered geology
and no60m depth is universally required.

First run the inexpensive deterministic preflight (no FE solve):

```
/opt/ale-kratos/bin/python SOFTWARE/scripts/native_model_contract.py FAMILY
```

Then run the staged native model for each position0..4, for example:

```
OMP_NUM_THREADS=1 /opt/ale-kratos/bin/python SOFTWARE/scripts/native_family_runner.py FAMILY FAMILY/results/pos_0m --position 0 --until full --linear-backend amgcl-gmres --max-iterations 400 --write-checkpoints
```

Here `SOFTWARE` and `FAMILY` denote the task software and delivered family
directories. Iteration budgets affect work allowance, not convergence criteria.
Retain each successful native `FileSerializer` checkpoint to continue after a
runtime limit. Only the completed common Stage7 state, before any inner member
installation, may branch the five heights. Final displacements alone are not
a valid constitutive restart.

The runner saves native editable/results files, a precision nodal-field export,
calculation logs and the Stage1 baseline. Retain these generated products in
`results/pos_Nm/` (or `pos_Nm/`). They are the supported native project outputs,
not a requirement to invent extra archive formats. The evaluator independently
replays your frozen inputs, reads your saved `native_model.mdpa` and
`native_nodal_fields.tsv`, and compares physical monitors with its own results.
It does not accept your summary CSV as numerical authority.

Keep the originally required raw monitor table as `monitoring.csv` or
`monitoring.tsv` in each case directory, with columns
`monitor,x_m,y_m,z_m,ux_mm,uy_mm,uz_mm`. The native reader's equivalent
`independent_monitors.json` is also supported. Duplicate shell/crown endpoint
rows are permitted. Different profile row counts are accepted, with full
wall-top-to-toe coverage and sampling adequate to retain the physical maximum.
These are supported serialization choices for the original raw extraction,
not an additional engineering deliverable or prescribed monitor coordinates.

Evaluation first checks the published deterministic model contract, then independently
executes one common prefix and five native suffixes, audits actual source facts,
and reads physical DBC/CX results. It applies the original50/50 rubric and
max(0.30mm,5%) response tolerance; percentages retain max(0.30point,8%).
No language-model service or evaluator credential is needed. Native replay
unavailability produces an evaluator error; preflight violations identify
the failed public check. Horizontal native hex faces use bilinear Q4 sampling.
Shell/crown interior traces may differ when the endpoint coupling is valid;
both traces are retained, with gaps, overlaps and invalid constraints rejected.
