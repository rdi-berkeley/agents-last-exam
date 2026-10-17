# Declared survey reconstruction

This is source-only preparation of the original Cailian input. It replaces the
unavailable Civil custom-surface decoding with an explicitly declared terrain;
the original agent assignment remains alignment design and raw ground profiling.
It contains no reference alignment or solved route.

Input DXF SHA256: `98b4a71a597847b621a5804498748c0ea0714a7f630ee0a161472ba7af6b04ad`.
Extracted survey CSV SHA256: `d3fbef586fab08cecf682b4cb33865812eaf6ef413ec540cd5258a2ee17016ca`.
Frozen OBJ SHA256: `f5b136d233c8bb62bfbb6f1cfba44a48470a6cccf4077a7e883c9286b9f356df`.

The 4,162 unique vertices represent 4,286 source records after collapsing 124
duplicates. Thirty zero-Z annotation placeholders are excluded using their paired
TEXT evidence; one ambiguous water-surface record (handle 5D8A) is excluded.
At exact duplicate XY locations, precise POINT coordinates take precedence over
rounded gc200 insert symbols. Heights are not averaged. Eight negative elevations
and distinct nearby points with sharp height changes are retained without smoothing.
See the supplied selection/exclusion CSVs for source handles and decisions.

The fixed connectivity starts with Delaunay triangulation and nine convex flips
to enforce eleven observed gully connections. Only triangles with maximum XY edge
length at most 60 m are retained: twice the maximum nearest-neighbour distance,
25.975262 m, rounded up to a multiple of 10 m. The 8,091 retained faces define five
components. No elevations are extrapolated outside that union. Unclassified map
linework was not assigned invented elevations or silently treated as a breakline.
The exact reconstruction evidence and gully connections accompany this input.

OBJ coordinates are WCS easting, northing, elevation in metres. TSV X means
northing and Y means easting. Road Points/Faces contain the same coordinates scaled
to millimetres. No EPSG identity or equivalence to the original Civil triangle
connectivity is asserted. The native display mesh has float precision; the supplied
compatibility helper samples the double-precision native Points/Faces.

The task launcher applies a narrow in-process compatibility patch to the installed
Road ProfileFrame callbacks and group-icon serialization. It does not replace
FreeCAD or Road objects. Raw samples use a 0.25 m grid, terrain intersections and
20 m export stations; the evaluator independently samples the frozen source OBJ.
