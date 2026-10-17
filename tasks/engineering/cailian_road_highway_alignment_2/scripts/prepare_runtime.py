import json
import os
from pathlib import Path
import shlex
import sys


def main():
    software = Path(sys.argv[1]).resolve()
    candidates = [Path(os.environ.get("CAILIAN_ROAD_RUNTIME", "/opt/cailian-road")), Path.home() / ".local/opt/cailian-road"]
    candidates.extend(path.parent for path in Path.home().glob("*/runtime/FreeCAD.AppImage"))
    runtime = next((path for path in candidates if (path / "FreeCAD.AppImage").is_file() and (path / "Road/freecad/road/objects/profile_frame.py").is_file()), None)
    if runtime is None:
        raise RuntimeError("Install FreeCAD 1.1.3 and Road 1ee9e1081d6bfc93d64a765154ea845697f2d312 with pyproj; set CAILIAN_ROAD_RUNTIME to their directory")
    destination = software / "runtime"
    if destination.is_symlink():
        if destination.resolve() != runtime.resolve():
            destination.unlink()
    if not destination.exists():
        destination.symlink_to(runtime, target_is_directory=True)
    launcher = software / "open_road.sh"
    launcher.write_text("#!/bin/sh\nset -eu\nexport CAILIAN_SOFTWARE=" + shlex.quote(str(software)) + "\nexec " + shlex.join([str(destination / "FreeCAD.AppImage"), "--user-cfg", str(software / "FreeCAD-user.cfg"), "--system-cfg", str(software / "FreeCAD-system.cfg"), str(software / "open_road.FCMacro")]) + ' "$@"\n')
    launcher.chmod(0o755)
    print(json.dumps({"runtime": str(runtime), "launcher": str(launcher), "source_files_modified": False}))


if __name__ == "__main__":
    main()
