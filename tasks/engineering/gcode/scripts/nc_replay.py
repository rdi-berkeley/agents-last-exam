"""Run LinuxCNC's standalone interpreter and retain its canonical event stream."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class ReplayError(RuntimeError):
    def __init__(self, message, evidence_directory):
        super().__init__(message)
        self.evidence_directory = Path(evidence_directory)


def sha256_file(path):
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_events(lines):
    """Yield native record envelopes from an iterable of canonical trace lines."""
    pattern = re.compile(r"^\s*(\d+)\s+N(\d+|\.+)\s+([A-Z][A-Z_0-9]*)\((.*)\)$")
    for line in lines:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"Unsupported native canonical trace record: {line[:512]!r}")
        ordinal, block, name, arguments = match.groups()
        yield {
            "ordinal": int(ordinal),
            "source_block": int(block) if block.isdigit() else None,
            "name": name,
            "arguments": arguments,
        }


def iter_events(events_file):
    """Yield one event at a time from an accepted replay's events_file JSONL path.

    Coordinates retain native units and frames. Iteration does not prefetch the
    file or verify its receipt hash; use sha256_file for an integrity check.
    """
    with Path(events_file).open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def replay(
    nc_file,
    tool_table,
    *,
    runtime_root,
    work_root,
    parameter_file=None,
    ini_file=None,
    timeout=30,
):
    """Return a compact receipt pointing to accepted canonical events in JSONL.

    The private runtime contains usr/bin/rs274 and its libraries. work_root must
    be a caller-selected disk directory. Parameters default to native empty-file
    initialization; supply the setup parameter file when offsets are required.
    Native execution completes before streaming trace conversion. Memory scales
    with one record, not the event count; evidence storage scales with input and
    output size. Failures retain evidence and never publish an accepted stream.
    """
    nc_file = Path(nc_file).resolve(strict=True)
    tool_table = Path(tool_table).resolve(strict=True)
    runtime_root = Path(runtime_root).resolve(strict=True)
    executable = runtime_root / "usr/bin/rs274"
    if not nc_file.is_file() or not tool_table.is_file() or not executable.is_file():
        raise ValueError("NC input, tool table and native rs274 executable must be files")
    if not 0 < timeout < float("inf"):
        raise ValueError("Replay timeout must be finite and positive")
    isolation = shutil.which("bwrap")
    if isolation is None:
        raise RuntimeError("bubblewrap is required to isolate native interpreter state")
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix="replay-", dir=work_root))
    shutil.copyfile(tool_table, evidence / "tools.tbl")
    if parameter_file is None:
        (evidence / "parameters.var").write_text("")
    else:
        shutil.copyfile(parameter_file, evidence / "parameters.var")
    shutil.copyfile(evidence / "parameters.var", evidence / "parameters.initial.var")
    shutil.copyfile(nc_file, evidence / "input.ngc")
    user_directory = Path.home()
    command = [
        isolation,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-net",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(evidence),
        str(user_directory),
        "--ro-bind",
        str(runtime_root),
        "/mnt",
        "--ro-bind",
        str(nc_file.parent),
        "/media",
        "--ro-bind",
        str(evidence / "input.ngc"),
        str(Path("/media") / nc_file.name),
        "--chdir",
        "/media",
        "--unsetenv",
        "LD_PRELOAD",
        "--unsetenv",
        "PYTHONPATH",
        "--setenv",
        "LD_LIBRARY_PATH",
        "/mnt/usr/lib:/mnt/usr/lib/x86_64-linux-gnu",
        "--setenv",
        "PYTHONHOME",
        "/mnt/usr",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "LC_ALL",
        "C",
    ]
    native_command = [
        "/mnt/usr/bin/rs274",
        "-g",
        "-n",
        "2",
        "-t",
        str(user_directory / "tools.tbl"),
        "-v",
        str(user_directory / "parameters.var"),
    ]
    if ini_file is not None:
        shutil.copyfile(ini_file, evidence / "replay.ini")
        native_command.extend(["-i", str(user_directory / "replay.ini")])
    command.extend(
        native_command + [str(Path("/media") / nc_file.name), str(user_directory / "canonical.txt")]
    )
    receipt = {
        "accepted": False,
        "command": command,
        "evidence_directory": str(evidence),
        "input_sha256": sha256_file(evidence / "input.ngc"),
        "rs274_sha256": sha256_file(executable),
        "tool_table_sha256": sha256_file(evidence / "tools.tbl"),
        "parameter_sha256": sha256_file(evidence / "parameters.initial.var"),
        "coordinate_convention": "Native canonical units and frames; consume unit/offset/plane events",
    }
    try:
        with (
            (evidence / "stdout.txt").open("w") as stdout,
            (evidence / "stderr.txt").open("w") as stderr,
        ):
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                env=os.environ.copy(),
                check=False,
            )
        receipt["returncode"] = result.returncode
        if result.returncode:
            with (evidence / "stderr.txt").open(errors="replace") as stderr:
                detail = stderr.read(4096).strip()
            raise ReplayError(
                detail or f"Native interpreter exited with {result.returncode}", evidence
            )
        event_count = 0
        events_bytes = 0
        digest = hashlib.sha256()
        partial = evidence / "events.jsonl.partial"
        with (
            (evidence / "canonical.txt").open(encoding="utf-8") as trace,
            partial.open("wb") as output,
        ):
            for event in canonical_events(trace):
                if event["name"] == "CANON_ERROR":
                    raise ReplayError("Native interpreter returned CANON_ERROR", evidence)
                record = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
                output.write(record)
                digest.update(record)
                event_count += 1
                events_bytes += len(record)
        if not event_count:
            raise ReplayError("Native interpreter returned an empty trace", evidence)
        events_file = evidence / "events.jsonl"
        partial.replace(events_file)
        receipt.update(
            events_file=str(events_file),
            event_count=event_count,
            events_bytes=events_bytes,
            events_sha256=digest.hexdigest(),
        )
        receipt["accepted"] = True
        return receipt
    except (OSError, ValueError, subprocess.TimeoutExpired, ReplayError) as error:
        receipt["error"] = str(error)
        raise ReplayError(str(error), evidence) from error
    finally:
        (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc_file", type=Path)
    parser.add_argument("--tool-table", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument("--ini-file", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    try:
        result = replay(**vars(args))
    except ReplayError as error:
        parser.exit(1, f"{error}\nEvidence: {error.evidence_directory}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
