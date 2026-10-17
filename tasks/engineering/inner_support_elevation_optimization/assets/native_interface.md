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
a Python3.10 virtual environment and install the runtime packages:

```sh
python3.10 -m venv /opt/ale-kratos
/opt/ale-kratos/bin/pip install KratosMultiphysics==10.4.3 KratosGeoMechanicsApplication==10.4.3 KratosStructuralMechanicsApplication==10.4.3 KratosLinearSolversApplication==10.4.3 numpy scipy
```

The task software directory contains the native model runner, input adapter,
member-section adapter and raw result reader. Mesh, geometry, monitoring
locations and the engineering choices are supplied by you.

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

## Runtime interface

`SOFTWARE/scripts/native_family_runner.py` is the native model entry point,
using the installed `/opt/ale-kratos/bin/python`. Its positional arguments
are the family input directory and case result directory. `--position` selects
0 through 4; `--until full` requests the complete construction history.
Its command-line interface includes native checkpoint and restart options.
A native FileSerializer checkpoint preserves constitutive and construction
state; nodal arrays alone are not a native restart format.

For example, with `SOFTWARE` and `FAMILY` set to the task software directory
and your family input directory:

```sh
OMP_NUM_THREADS=1 /opt/ale-kratos/bin/python "$SOFTWARE/scripts/native_family_runner.py" "$FAMILY" "$FAMILY/results/pos_0m" --position 0 --until full
```

The task VM and independent replay resource limits are specified in
model_definition.md. Additional mesh/domain studies are optional.

The runner saves native editable/results files, a precision nodal-field export,
calculation logs and the Stage1 baseline. Retain these generated products in
`results/pos_Nm/` (or `pos_Nm/`). They are the supported native project outputs,
not a requirement to invent extra archive formats. The evaluator independently
replays your frozen inputs, reads your saved `native_model.mdpa` and
`native_nodal_fields.tsv`, and compares physical monitors with its own results.
It does not accept your summary CSV as numerical authority.

## Raw result reader

The public `SOFTWARE/scripts/native_result_adapter.py` provides
`read_native_result`, `read_native_coordinates`, `read_native_translations`
and `read_stage1_baseline`. Add `SOFTWARE/scripts` to your Python import path.
`read_native_result` returns a `NativeResult` containing node coordinates in
meters, element types/properties/connectivity, displacement fields in meters,
rotations, scalar properties, constraints and active/soil membership.
`result.displacement(node_id)` returns the three stored displacement components.

```python
import hashlib
from pathlib import Path
from native_result_adapter import read_native_result, read_stage1_baseline

case = Path("FAMILY/results/pos_0m")
mdpa = case / "native_model.mdpa"
fields = case / "native_nodal_fields.tsv"
baseline_path = case / "stage_1_k0_equilibrium_hold.npz"
result = read_native_result(
    mdpa,
    expected_sha256=hashlib.sha256(mdpa.read_bytes()).hexdigest(),
    coordinates_path=fields,
    coordinates_sha256=hashlib.sha256(fields.read_bytes()).hexdigest(),
    precise_translations=True,
)
baseline = read_stage1_baseline(
    baseline_path,
    expected_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    soil_nodes=result.soil_nodes,
)
```

Digests pin the bytes being read; they do not establish physical correctness.
The reader checks format, finite values, node coverage and agreement with
MDPA's six-significant-digit serialization before restoring precision from
the TSV. Stage1 arrays are returned separately. Displacements and constraint
offsets retain their native values. The reader does not apply installation or
baseline corrections, interpolate monitor locations, produce monitor tables,
calculate response maxima, compare cases or write deliverables. Those
postprocessing calculations are your work, using the observation definitions
in `model_definition.md`.

Keep the originally required raw monitor table as `monitoring.csv` or
`monitoring.tsv` in each case directory, with columns
`monitor,x_m,y_m,z_m,ux_mm,uy_mm,uz_mm`. An equivalent
`independent_monitors.json` is also supported: its `stations` and `profiles`
arrays contain rows with `monitor`, `xyz_m` (three coordinates in meters) and
`u_m` (three displacement components in meters). Duplicate shell/crown endpoint
rows are permitted. Different profile row counts are accepted, with full
wall-top-to-toe coverage and sampling adequate to retain the physical maximum.
These are supported serialization choices for the original raw extraction,
not an additional engineering deliverable or prescribed monitor coordinates.
