# Cailian Road on Linux

Release status: native controls passed; awaiting Agent review and rollout.
The final compatibility patch updates the native alignment Shape before sampling
terrain crossings. Both routes now pass ordinary reopen, Terrain relinking,
editing, restoration, save and independent evaluation, with maximum sampled
profile errors of 0.000951265 m and 0.002035161 m. The unchanged tolerance is
0.2 m. Correct routes score 1.0, shifted TSV elevations score 0.6, and a wrong
native endpoint scores 0.0. A separate control retains an actual 0.1 m route
edit, saves/reopens it and scores 1.0. These are native controls, not Agent runs.
The explicit refresh macro remains optional; these controls use automatic
recomputation and do not rely on that fallback.

`load()` supplies the Linux configuration. `start()` stages only task-local source
assets, the application launcher, public compatibility helpers and native verifier.
It does not obtain private route fixtures or use bucket reference outputs.

The runtime must already contain FreeCAD 1.1.3, Road commit
`1ee9e1081d6bfc93d64a765154ea845697f2d312`, and its NumPy/SciPy/pyproj dependencies.
Set `CAILIAN_ROAD_RUNTIME` in the guest to the directory containing
`FreeCAD.AppImage`, `Road/` and `road-python/`, or install it in
`/opt/cailian-road` or `~/.local/opt/cailian-road`. Setup also discovers an existing
user runtime directory. It creates a task-local symlink and never patches the
installed addon files. This task does not install software or replace an image.

Existing FreeCAD addon links in `~/.local/share/FreeCAD/Mod/Road` and
`~/.local/share/FreeCAD/v1-1/Mod/Road` must point to the same permanent Road
installation. FreeCAD scans these links during startup even when a task-specific
configuration file is supplied.

Start the solver application with `software/open_road.sh`. The bootstrap installs
the compatibility callbacks once in that FreeCAD process. Subsequent ordinary
document opens use those callbacks; no special file-open wrapper is required.
The patch defers Terrain-link resampling until recompute after restoration, uses
native double-precision Points/Faces, refreshes ProfileFrames after Model edits,
and fixes the inspected group icon `Icon`/`icon` serialization mismatch.
The objects remain upstream Road Alignment, Terrain, Profile and ProfileFrame.

The evaluator loads the compatibility patch in read-only mode: it must not repair
a submitted profile before checking it. It validates saved native objects and
compares against the frozen source mesh, with the original rubric and thresholds.
Evaluation requires systemd, passwordless task-runtime sudo, Xvfb and four visible
CPUs; its process is pinned to CPU 3 with a 3 GiB memory limit and no swap.

The compatibility patch covers this fixed survey terrain workflow. Editing
terrain operations, station equations and optional-spiral editing have not been
validated here. The public source reconstruction disclosure is in
`assets/TERRAIN_PROVENANCE.md`; `assets/original_contract.json` preserves the
original task description and grading contract. Assets contain no solved route.
