"""Build the instrument-output helper against supplied local Ardour headers/libraries."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_symbols(source):
    return sorted(set(re.findall(r'"(_ZN[^"\n]+)"', source.read_text())))


def build(headers, dev_prefix, library_dir, output):
    source = Path(__file__).with_name("native_batch_export.cc")
    output = output.resolve()
    receipt_path = output.with_suffix(".build.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Use a new output path to retain previous build evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "failed", "source_sha256": sha256(source)}
    temporary = output.with_suffix(".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        library = library_dir / "libardour.so"
        symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(library)],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        ).stdout
        exported = {line.split()[-1].split("@")[0] for line in symbols.splitlines() if line}
        required = required_symbols(source)
        missing = sorted(set(required) - exported)
        receipt.update(required_symbols=required, missing_symbols=missing)
        if missing:
            raise RuntimeError("Installed Ardour lacks required batch API symbols")
        includes = [headers]
        if dev_prefix:
            includes.extend(
                [
                    dev_prefix / "usr/include/glibmm-2.4",
                    dev_prefix / "usr/include/sigc++-2.0",
                    *sorted(dev_prefix.glob("usr/lib/*/glibmm-2.4/include")),
                    *sorted(dev_prefix.glob("usr/lib/*/sigc++-2.0/include")),
                ]
            )
        else:
            includes = [headers]
        packages = ["glib-2.0", "libxml-2.0"]
        if not dev_prefix:
            packages.extend(["glibmm-2.4", "sigc++-2.0"])
        flags = subprocess.run(
            ["pkg-config", "--cflags", *packages],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        command = [
            "g++",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-O2",
            *(f"-I{directory}" for directory in includes),
            *shlex.split(flags),
            str(source),
            f"-L{library_dir}",
            "-lpbd",
            "-ldl",
            "-o",
            str(temporary),
        ]
        receipt.update(
            command=command,
            compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
            libraries={
                str(path.resolve()): sha256(path) for path in (library, library_dir / "libpbd.so")
            },
            headers={str(path): sha256(path) for path in sorted(headers.rglob("*.h"))},
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        result.check_returncode()
        temporary.replace(output)
        receipt.update(status="built", output=str(output), output_sha256=sha256(output))
        return receipt
    except Exception as exc:
        receipt["error"] = str(exc)
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--dev-prefix", type=Path)
    parser.add_argument("--library-dir", type=Path, default=Path("/usr/lib/ardour6"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.headers, args.dev_prefix, args.library_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
