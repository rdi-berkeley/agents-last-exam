# Betonwerk Katzenberger architectural reconstruction

Reconstruct the workshop Hall and residential Tower from the six architectural
PDFs and the supplied 3D snapshot. This is a complete architectural model at
LOD 200–300, including the building shell, structure, facade and internal
workshop modules. Furniture, annotations and surrounding landscape are excluded.

Use Blender 4.5 on Linux, installed at `/opt/blender-4.5.1/blender`. Both the
graphical editor and scripts are allowed. Import `base_model.obj` with +Y forward
and +Z up. It contains positioning lines, not the completed building. Preserve
its axis orientation and millimeter dimensions; +X is east, +Y north, +Z up.
Ground-floor elevation is z=0. A common X/Y translation of the complete building
is allowed; the renderer centers its views independently. Relative component
placement and orientation are scored, not the absolute horizontal origin.
Any legacy `.3dm` copy of this same anchor is optional and requires no Rhino.

## Required building features

- The wide workshop Hall, roughly 58 by 68 m and 11.8 m high.
- A continuous solid north Hall wall with metal-panel cladding.
- An open south structural frame, with the Hall interior visible through it.
- The Hall ceiling near 11,820 mm and partial mezzanines near 4,030, 6,130 and
  7,830 mm, following the drawings.
- Three workshop modules in the locations and orientations shown in the Hall
  ground-floor drawing.
- The tall narrow Tower at the north-eastern end, about 19 by 11 m in plan.
- Tower slabs, circulation core, curtain-wall glass and mullions, and the metal
  cap above the curtain wall. Follow the tower plans, sections and elevations.

Use the drawing dimensions to resolve details. The snapshot also shows structural
members projecting beyond the enclosed Hall; the anchor is not a requirement to
clip every component to a rectangular bounding box.

## Deliverables

Save `model.obj` and `model.blend` in the output directory specified in the task
prompt. The OBJ must contain actual polygon geometry in millimeters, +Z up.
The Blender file must reopen with the same evaluated surfaces, including
modifiers and instances. Pack external geometry dependencies. Object names,
object counts, polygon counts and triangulation are unrestricted. The two
representations must agree to within 1 mm. Do not submit DWG files.

## Evaluation

The evaluator checks the two submitted representations and renders the OBJ with
Blender on a CPU. It compares 14 architectural views: four plans, four elevations,
two sections and four corner axonometrics. A visual judge assesses eight equally
weighted criteria: Hall/Tower proportions, floor levels, Tower placement, the
solid north Hall facade, the open south structural frame, the Tower curtain-wall
mullions, workshop-module placement and elevation silhouettes. Final score is
the fraction of these criteria satisfied. Materials and textures are not scored.

This is not pixel matching. Equivalent architectural geometry and different
mesh organization are acceptable. Missing or unreadable required deliverables,
inconsistent native/OBJ geometry, or forbidden DWG output receive zero.
