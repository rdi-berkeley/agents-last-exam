You are a civil engineer using FreeCAD with the Road workbench on Linux.

Design a horizontal alignment for Cailian Road and generate an unedited existing-ground profile along your own alignment.

Control points (metres; X is northing and Y is easting):
- Start: X = -52093.6660, Y = -5836.2683, listed Z = 5.5.
- End: X = -50855.6202, Y = -4142.4687, listed Z = 5.3.
- Both endpoints must connect within 0.5 m in XY. Profile Z comes from the supplied surface; the listed endpoint Z values are not additional surface-height constraints.

Design constraints:
- Minimum circular curve radius: 85.0 m.
- Minimum spiral length, if spirals are used: 25.0 m. Spirals are optional.
- Design speed: 30 km/h.
- Total alignment length: 1800.0 m to 2400.0 m.
- Alignment length divided by the straight-line distance between the specified control points must be at least 1.05. The chord is 2098.026332 m, so this requires a length of at least 2202.927648 m within the total-length range above.
- Include at least 2 horizontal curves.

Workflow:
1. Run `{software_dir}/open_road.sh`. This starts the installed FreeCAD/Road application with the task's compatibility patch and opens `{input_dir}/declared-survey-terrain.FCStd`.
2. Create your Road Alignment using the workbench tools or its existing Python API. Keep the provided terrain unchanged. Road geometry uses metres; FreeCAD shapes and terrain Points/Faces use millimetres. The OBJ uses easting, northing, elevation in metres. Native terrain geolocation is already configured.
3. Use Road's Create Profile Frame command, select your alignment and the supplied terrain, and recompute. This creates the raw surface profile. Do not manually edit its elevations. The compatibility patch safely restores documents and refreshes linked surface profiles when the document is recomputed. Save your work with one submitted alignment and its linked profile.
4. Save `{output_dir}/alignment.FCStd`. The native file must include the supplied terrain, the alignment, and the associated ProfileFrame.
5. Run `{software_dir}/export_metrics.FCMacro` using FreeCAD's Macro dialog after saving. It exports `alignment_metrics.tsv` beside the saved drawing. Include stations 0, 20, 40, ... and the actual alignment endpoint; the last interval may be shorter than 20 m. Columns must be exactly `Station`, `X`, `Y`, `Z`. Z must equal ground elevation at the corresponding point on your alignment.

The public surface is a declared survey reconstruction, provided as input repair. You do not need to decode DWG or reconstruct it. Read `{input_dir}/TERRAIN_PROVENANCE.md` for source selection, units and boundaries. It is not claimed to be the recovered original Civil TIN. No solved route is supplied, and grading uses your own route against the provided fixed ground.

Keep work in `{output_dir}`. Grading preserves the original geometric gates and 20/20/40/20 curve/spiral/elevation/formatting weights. Ground tolerance is 0.2 m and the passing score is 70/100.
