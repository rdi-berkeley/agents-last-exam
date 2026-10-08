# Native CAM migration draft

The task uses Linux and FreeCAD CAM, with LinuxCNC's native interpreter and a
pinned CAMotics cutting backend. `assets/instruction.md` is the prompt source.
`main.py` dispatches through the actual ALE setup/evaluate interface; it never
accepts a pre-submitted STL in place of saved-project replay.

## Data layout

Each registered workpiece requires its own converted original public package:

```
<task>/<variant>/input/blank.FCStd
<task>/<variant>/input/tool-library/original-tools.json
<task>/<variant>/input/tool-library/Tools/...
<task>/<variant>/input/geometry/...
<task>/<variant>/input/geometry/machining-region.json
<task>/<variant>/input/<variant>.jpg
<task>/<variant>/reference/reference_sim.stl
```

The blank project contains original machining surfaces, stock and toolcontrollers,
but no operations. Original external geometry is a separate reference, not a
closed solid imposed on the machining surfaces. The public machining-region file
is prepared from those original surfaces and stock using `prepare_region` in
`scripts/native_geometry.py`. It must not be derived from a candidate or hidden
reference. The hidden reference STL remains evaluator-only.

The default workpiece `125162_319` has native controls. Other registered workpieces
remain marked `awaiting_variant_data_validation`; registration does not certify
their conversion. None of these changes publishes task data or a runtime image.

## Runtime configuration

The operator provisions `/opt/ale-cam/runtime.json`, or supplies another location
through `ALE_CAM_RUNTIME_CONFIG` when constructing the task configuration:

```json
{
  "freecad": "/opt/ale-freecad/FreeCAD.AppImage",
  "interpreter_runtime": "/opt/ale-cam/linuxcnc",
  "collision_python": "/opt/ale-cam/python/bin/python",
  "stock_binary": "/opt/ale-cam/native-stock"
}
```

The interpreter tree must contain `usr/bin/rs274` and its matching runtime
libraries. `collision_python` needs Python 3.11 or later and
`requirements-eval.txt`. `bubblewrap` and `xvfb-run` must be available. Setup
checks these dependencies and the converted public inputs before solving.

FreeCAD can share the pinned installation used by the Road task. Its addon links
and the geometry Python environment must resolve outside temporary build or
diagnostic directories, including the Python environment's base interpreter.

`assets/linuxcnc-runtime.json` pins the five tested interpreter packages by URL,
version and SHA-256. Place those exact Debian archives in a persistent package
cache, then run Python 3.11 or newer:

```sh
python scripts/install_interpreter.py \
  --manifest assets/linuxcnc-runtime.json \
  --package-cache /persistent/package-cache \
  --destination /opt/ale-cam/linuxcnc
```

The destination must not already exist. The installer verifies every archive
before creating it, extracts the interpreter, matching private libraries and
package notices, and checks dynamic library resolution. Ubuntu still supplies
the standard system libraries, including `libedit2` and `libtirpc3`. The private
Debian libraries do not replace the host's Python or system libraries. Nine real
interpreter controls pass after fresh extraction in a separate directory, including
arcs, unit changes, drilling and invalid-program rejection. A wrong package hash
is rejected before creating the destination.

`requirements-native-lock.txt` records the exact geometry packages used by these
controls (Python 3.14.3). The broader `requirements-eval.txt` expresses supported
package ranges; it is not a reproducible environment lock.

The validated CAM GUI is FreeCAD 1.1.3. The stock backend uses CAMotics commit
`e84665f2fa9d1151f03282ac7e01320bc65e015b` and cbang commit
`f91bf30cead30c2d4c5fc28b4f1720cd09f48c50`, plus the validated sweep-geometry
compatibility changes in `scripts/camotics-compat.patch` and empty-cell storage
fix in `scripts/camotics-sparse-empty.patch`. An arbitrary CAMotics installation
is not a substitute.

`scripts/build_stock_from_sources.sh <archive-cache> <new-build-directory>`
extracts both pinned source archives into new trees, applies the empty-cell fix,
builds cbang and CAMotics, then links the driver with the geometry corrections.
Compiler temporary files remain under the persistent build directory. This path
needs Ubuntu 22.04 development packages including `build-essential`, `scons`,
`python3-dev`, `libssl-dev`, `libzstd-dev`, `libre2-dev`, `libevent-dev`,
`default-libmysqlclient-dev`, `libexpat1-dev`, `liblz4-dev`, `libbz2-dev`,
`zlib1g-dev` and `libyaml-dev`.

`scripts/build_native_stock.sh <source-root> <new-build-directory>` reproduces the
native driver link on Ubuntu 22.04. The source root contains the two pinned source
archives (named `camotics-source.tar.gz` and `cbang-source.tar.gz`) and their
extracted `CAMotics/` and `cbang/` trees, with the patched baseline static
libraries produced by the full-source command above.

The script verifies archive hashes, extracts the five affected source files into
an isolated compatibility directory, applies the exact patch, and links the
repository driver against the existing native libraries. It records library,
source, patch and binary hashes. This build step was executed in the retained VM;
its sloped-cut output has identical triangle data to the validated earlier build.

The full-source path was also executed from freshly extracted trees in 520.49
seconds, including the empty-cell correction discovered by comparing the existing
source tree with the original archive. The rebuilt driver passes the real sloped
cut in 10.08 seconds with identical triangle data. Original archives, both patches,
compiler logs, library hashes and the tested driver identify this build. The
temporary source/object trees were then removed; the tested binary and compressed
control output remain available in the private verification workspace.

Full clean-image provisioning, including FreeCAD, interpreter libraries and Python
dependencies, remains a release prerequisite. A successful driver rebuild does not
prove that entire provisioning path.

Setup uploads the public editing and preview helpers to `software/`. The preview
configuration contains only public source paths and runtime executables. It does
not include reference answers or a grading command.

## Evaluation

The evaluator stages fresh helper code in a unique `.evaluation/` directory and
starts a bounded native worker. It exports saved FreeCAD operations, interprets
their NC instructions, checks all physical motions, cuts the original stock, and
then applies the existing region-aware geometry scorer. The stock and safety
receipts must describe identical motions. Submitted stock files are ignored.
Native safety has a 3000-second budget within the existing 3600-second preview
limit; stock simulation retains its 300-second limit. The overall evaluator
still allows 6900 seconds, including geometry scoring.

Holder-free original tools retain their source `HolderOverhang` metadata when
`Mount` is absent. Export checks it against the original manifest and replay
checks it against the original document. This does not translate the cutter or
invent a holder, shank or gauge offset. A holder without its mount is rejected.

Remaining-stock safety queries clip the original stock to a box enclosing the
complete component sweep, chord error and tolerance margin. A lazy index of
completed feeds selects only intersecting history. Upper refinement visits older
completed feeds first, removing verified interiors individually and unique
cutter endpoints in bounded batches. Periodic complete-sweep clearance checks
can end refinement early. Separate batches remove enclosing feed boxes from the
lower bound. Horizontal and vertical
straight feeds use the swept interior of a cylinder only after an OCCT difference
proves that cylinder lies inside the original cutter. Curved chords and sloped
feeds retain the endpoint bound. Up to eight conservative local upper bounds can be reused
for adjacent queries. Each region records successfully subtracted feed interiors
and individual endpoints, so later queries skip those same removals. Failed
refinements remain eligible for retry, and an applied interior does not imply
either endpoint was applied. An empty upper bound or a complete native sweep-clearance
proof can finish refinement early. If OCCT raises `ValueError: Null shape` during
an upper-bound cut, the previous valid upper bound is retained and the failed
refinement is recorded. Lower-bound failures still stop certification. Current feeds
can only reduce the lower bound, so they cannot establish their own clearance.
Rapid, shank, holder, original-face gouge and canonical event checks retain their
existing tolerances and fail-closed behavior. These bounds do not replace the
separate CAMotics material-removal replay or its motion digest checks.

Missing/empty projects and verified collisions produce a completed score of zero.
Runtime failures, unavailable safety checks and missing reference data remain
evaluation errors. A terminal receipt is written atomically; the host polls that
receipt without downloading growing native logs. Cancellation/timeout terminates
the worker process group. Task registration preserves the original 18 workpiece
identities while using POSIX paths and Linux metadata.

The actual ALE setup/evaluate entrypoint was executed in the retained QEMU VM:
the constructed saved project scored 0.92314 in 714.25 seconds, missing and empty
projects both scored zero, and a missing runtime raised a setup error. Including
setup and these controls took 756.84 seconds. This is native developer control
work, not solver-Agent rollout or clean-image provisioning acceptance.
