You are a CAM engineer. Design roughing and finishing toolpaths for workpiece
**{variant_name}** using FreeCAD CAM and the supplied original tool library.

## Inputs

- `{input_dir}/blank.FCStd`: original machining surfaces, stock and toolcontrollers,
  with no machining operations.
- `{input_dir}/tool-library/`: the original cutters, shanks and holders.
- `{input_dir}/geometry/`: converted original geometry, face display metadata and
  source coordinate correspondence. The machining surfaces in the CAM Job are
  the target; the separately labeled external geometry is a reference. Open
  machining surfaces are intentional and need not be joined into a closed solid.
- `{input_dir}/{variant_name}.jpg`: original workpiece image.

## Work and delivery

1. Copy `blank.FCStd` to `{output_dir}/machining.FCStd`. Work on this copy.
2. Open it with `python3 {software_dir}/open_cam.py {output_dir}/machining.FCStd`.
   This enables the provided bindings for the original tools and native CAM editing.
3. Choose roughing and finishing strategies, cutting parameters and tool mounting.
   Use only the supplied original cutters, shanks and holders. Do not change the
   original target or stock, or introduce a new cutter. Preserve the original tool
   identities. Holder overhang can be adjusted through `OriginalAssembly.Mount.Overhang`.
4. Complete the machining without gouging the target, colliding with the shank or
   holder, or running rapid moves through remaining material. Keep critical surfaces
   within 0.3 mm and all modeled surfaces within 2.0 mm of the required shape.
5. Save the calculated, active operations in `{output_dir}/machining.FCStd`.
6. Check the saved project with:

   `python3 {software_dir}/preview_stock.py {output_dir}/machining.FCStd {output_dir}/preview --config {software_dir}/preview.json`

   Use a new preview directory for each attempt. The result names a remaining-stock
   STL that you can open in FreeCAD. The bundled FreeCAD voxel preview has a known
   sloped-cut error; the provided preview uses the same native replay as evaluation.

Toolpaths remain your responsibility. Optional OpenCAMLib adaptive sampling is
available through `{software_dir}/enable_adaptive.FCMacro` on a selected CAM Surface
operation; native standard sampling is also allowed. The script supplies no toolpath.

## Evaluation

The evaluator exports your saved operations with FreeCAD's LinuxCNC postprocessor,
interprets their actual movements, checks safety, and simulates cutting with the
original tool geometry. A missing/empty project or a collision/gouge earns zero.
A submitted STL never substitutes for this replay. Software/environment failures
are reported as evaluation errors rather than solver scores.

After the safety gate, the remaining stock is compared with the hidden reference.
Surface agreement uses the original 0.3 mm and 2.0 mm bands, weighted 70% and 30%.
Both missing and unwanted material count. The machining region, defined from the
original public geometry, is checked separately from the remaining outer surface;
the lower region score is used so that untouched exterior cannot hide missing cuts.
You do not need to reproduce the author's operation sequence or toolpath.
