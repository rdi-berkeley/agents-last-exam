You are a civil engineer using FreeCAD with the Road workbench on Linux.

Design a horizontal alignment for Cailian Road and generate an unedited existing-ground profile along your own alignment.

Control points (metres; X is northing and Y is easting):
- Start: X = -52093.6660, Y = -5836.2683, listed Z = 5.5.
- End: X = -50855.6202, Y = -4142.4687, listed Z = 5.3.
- Connect both endpoints in XY. Profile Z comes from the supplied surface; the listed endpoint Z values are not additional surface-height constraints.

Design constraints:
- Minimum circular curve radius: 85.0 m.
- Minimum spiral length, if spirals are used: 25.0 m. Spirals are optional.
- Design speed: 30 km/h.
- Total alignment length: 1800.0 m to 2400.0 m.
- Alignment length divided by the straight-line distance between the specified control points must be at least 1.05.
- Include at least 2 horizontal curves.

Environment and delivery:
- Run `{software_dir}/open_road.sh` to start FreeCAD/Road with the task's compatibility patch and open `{input_dir}/declared-survey-terrain.FCStd`. Create your Road Alignment using the workbench tools or its existing Python API.
- Road geometry uses metres; FreeCAD shapes and terrain Points/Faces use millimetres. The OBJ uses easting, northing, elevation in metres. Native terrain geolocation is configured.
- Save `{output_dir}/alignment.FCStd` with the unchanged supplied terrain, one Road Alignment and its associated surface-derived ProfileFrame. Profile elevations must be unedited ground heights.
- To create the raw surface profile, use Road's Create Profile Frame command, select your alignment and the supplied terrain, and recompute. The compatibility patch restores documents and refreshes linked surface profiles on recompute.
- Save `{output_dir}/alignment_metrics.tsv` with columns exactly `Station`, `X`, `Y`, `Z`, all in metres. Include stations 0, 20, 40, ... and the actual alignment endpoint; the final interval may be shorter than 20 m. Z is ground elevation at the corresponding alignment point. The native table-export adapter is `{software_dir}/export_metrics.FCMacro`.
- After saving, run `{software_dir}/export_metrics.FCMacro` from FreeCAD's Macro dialog. It exports `alignment_metrics.tsv` beside the saved drawing.

The public surface is a declared survey reconstruction, provided as input repair. You do not need to decode DWG or reconstruct it. Read `{input_dir}/TERRAIN_PROVENANCE.md` for source selection, units and boundaries. It is not claimed to be the recovered original Civil TIN. No solved route is supplied, and grading uses your own route against the provided fixed ground.

Keep work in `{output_dir}`.
