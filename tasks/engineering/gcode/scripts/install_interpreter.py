"""Extract the pinned LinuxCNC interpreter from verified cached Debian packages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from urllib.parse import urlparse


LIBRARIES = {
    "liblinuxcnchal.so.0",
    "liblinuxcncini.so.0",
    "libnml.so.0",
    "libposemath.so.0",
    "libpyplugin.so.0",
    "librs274.so.0",
    "libtooldata.so.0",
}


def install(manifest_path, package_cache, destination):
    manifest = json.loads(Path(manifest_path).read_text())
    package_cache = Path(package_cache).resolve(strict=True)
    destination = Path(destination).absolute()
    parent = destination.parent.resolve(strict=True)
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(parent)], text=True
    ).strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("Interpreter installation requires disk-backed storage")
    if shutil.disk_usage(parent).free < 256 * 1024**2:
        raise RuntimeError("Interpreter installation requires 256 MiB free space")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination = parent / destination.name
    archives = []
    for package in manifest["packages"]:
        archive = package_cache / Path(urlparse(package["url"]).path).name
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != package["sha256"]:
            raise ValueError(f"Pinned package checksum mismatch: {archive.name}")
        archives.append((package, archive))
    destination.mkdir()
    receipt = {"packages": manifest["packages"], "files": [], "omitted_system_links": []}
    for package, archive in archives:
        with subprocess.Popen(
            ["dpkg-deb", "--fsys-tarfile", str(archive)], stdout=subprocess.PIPE
        ) as process:
            with tarfile.open(fileobj=process.stdout, mode="r|") as packed:
                for member in packed:
                    name = member.name.removeprefix("./")
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Invalid package member: {name}")
                    wanted = name.startswith("usr/share/doc/")
                    if package["name"] == "linuxcnc-uspace":
                        wanted |= name == "usr/bin/rs274" or (
                            path.parent == Path("usr/lib") and path.name in LIBRARIES
                        )
                    else:
                        wanted |= name.startswith("usr/lib/")
                    if not wanted or not (member.isfile() or member.issym()):
                        continue
                    target = destination / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if member.issym():
                        link = Path(member.linkname)
                        if link.is_absolute():
                            receipt["omitted_system_links"].append(name)
                            continue
                        if not (target.parent / link).resolve().is_relative_to(destination):
                            raise ValueError(f"Escaping package symlink: {name}")
                        target.symlink_to(member.linkname)
                    else:
                        with packed.extractfile(member) as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                        target.chmod(member.mode & 0o777)
                    receipt["files"].append(name)
            if process.wait() != 0:
                raise RuntimeError(f"Cannot unpack {archive.name}")
    executable = destination / "usr/bin/rs274"
    with executable.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["rs274_sha256"]:
        raise ValueError("Installed interpreter checksum mismatch")
    environment = {
        **os.environ,
        "LD_LIBRARY_PATH": f"{destination}/usr/lib:{destination}/usr/lib/x86_64-linux-gnu",
    }
    linkage = subprocess.run(
        ["ldd", str(executable)], env=environment, text=True, capture_output=True, check=True
    )
    if "not found" in linkage.stdout:
        raise RuntimeError("Missing interpreter host libraries: " + linkage.stdout)
    receipt.update(rs274_sha256=digest, linkage=linkage.stdout)
    (destination / "installation.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--package-cache", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = install(args.manifest, args.package_cache, args.destination)
    print(json.dumps({"rs274_sha256": result["rs274_sha256"], "files": len(result["files"])}))
