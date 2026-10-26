#!/usr/bin/env python3
"""Build the ALE Windows runtime snapshot on Daytona.

Stages (each idempotent; the builder sandbox id is kept in ``--state``):

    create     start a builder sandbox from a stock Daytona Windows snapshot (default windows-large: 4 vCPU / 16 GiB / 50 GiB, the per-sandbox cap on this org)
    bootstrap  upload + run bootstrap_runtime.ps1 (Python, venv, Node, 7-Zip, cua-server, E:, C:\\Users\\User)
    packages   upload env/packages-win from the ALE checkout and run install.ps1 for every
               requiredSystemPackages id of the tasks in --tasks (chocolatey-bootstrap first)
    data       upload <data_root>/tasks/<task>/<variant>/{input,software} for --tasks to E:\\agenthle,
               and bake reference.7z (password from $ALE_REFERENCE_ARCHIVE_PASSWORD) when
               <data_root>/tasks/<task>/<variant>/reference exists on the host
    verify     run every package's meta.json verify command + spot-check the layout
    snapshot   stop the builder and create the Daytona snapshot --name
    all        create bootstrap packages data verify snapshot

    uv run python env/daytona/build_snapshot.py all --data <data_root> \
        --tasks selected_tasks/daytona_windows_cpu.txt --name ale-win-$(date +%Y%m%d)

The Daytona API key comes from $DAYTONA_API_KEY. Reference archives use 7z (``7z``/``7zz``/``7za``) when
present on the host, otherwise py7zr.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath

HERE = Path(__file__).resolve().parent
BOOTSTRAP = HERE / "bootstrap_runtime.ps1"
REMOTE_BOOTSTRAP = r"C:\ale-bootstrap\bootstrap_runtime.ps1"
REMOTE_PACKAGES = r"C:\ale-bootstrap\packages-win"
DATA_ROOT_REMOTE = r"C:\drive-e\agenthle"  # == E:\agenthle inside the sandbox (bootstrap maps E: onto C:\drive-e)
UPLOAD_BATCH_BYTES = 256 << 20
UPLOAD_BATCH_FILES = 200
# meta.json verify strings that cannot pass as written (microsoft-edge-os single-quotes its ${env:...}
# paths so they never expand); install.ps1 performs the same existence check itself.
NON_PORTABLE_VERIFY = {"microsoft-edge-os"}
GPU_ONLY_PACKAGES = {"davinci-resolve-21.0"}   # only the gpu-free task needs it; no Windows GPUs on Daytona


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def client():
    from daytona import Daytona, DaytonaConfig

    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        sys.exit("set DAYTONA_API_KEY")
    return Daytona(DaytonaConfig(api_key=key))


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def ps(sandbox, script: str, timeout: int = 3600):
    """Run a PowerShell command string through the Daytona toolbox."""
    cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{script}"'
    return sandbox.process.exec(cmd, timeout=timeout)


# Daytona's toolbox shell keeps the PATH it started with, so tools the bootstrap installed (choco, 7z,
# python) must be put on PATH explicitly for the package installers (they probe `Get-Command choco`).
_TOOL_PATH = r"C:\ProgramData\chocolatey\bin;C:\Program Files\7-Zip;C:\Python312;C:\Python312\Scripts"
_VERIFY_PATH = _TOOL_PATH + r";C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\Program Files\ffmpeg\bin"


# Download overrides for installers whose default sources are gone (SourceForge mirror path 404,
# gs://ale-data-public is not anonymously readable). The installers read these env vars.
PACKAGE_ENV = {
    "kicad-10.0.0": {"KICAD100_INSTALLER_URL": "https://kicad-downloads.s3.cern.ch/windows/stable/kicad-10.0.0-1-x86_64.exe"},
    "microdicom-2025.3": {"MICRODICOM_ARCHIVE_URL": 'https://www.microdicom.com/downloads/Software/microdicom-2025.3-x64.zip'},
}


# Installers replaced outright. KiCad's NSIS MultiUser installer must run in the interactive session
# with `/S /allusers`: run headless as SYSTEM it hangs (and every later launch then exits at once with
# "The installer is already running"), and the SourceForge/GCS sources in install.ps1 are gone.
# The per-user copy mirrors what install.ps1 does. Runs through run_interactive() (see below).
PACKAGE_OVERRIDE = {
    "kicad-10.0.0": r"""
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$dst = 'C:\Users\User\AppData\Local\Programs\KiCad\10.0'
$kicad = Join-Path $dst 'bin\kicad.exe'
if ((Test-Path $kicad) -and ((Get-Item $kicad).VersionInfo.ProductVersion -like '10.0.0*')) { Write-Host 'OK kicad-10.0.0 already present'; exit 0 }
$pf = 'C:\Program Files\KiCad\10.0'
if (-not (Test-Path (Join-Path $pf 'bin\kicad.exe'))) {
  $exe = 'C:\WINDOWS\TEMP\kicad-10.0.0-1-x86_64.exe'
  if (-not (Test-Path $exe) -or (Get-Item $exe).Length -lt 900MB) {
    curl.exe --fail --location --silent --show-error -o $exe 'https://kicad-downloads.s3.cern.ch/windows/stable/kicad-10.0.0-1-x86_64.exe'
  }
  Get-Process | Where-Object { $_.Name -like 'kicad-10*' } | Stop-Process -Force -ErrorAction SilentlyContinue
  & 'C:\ale-bootstrap\run_interactive.ps1' -CommandLine ('"' + $exe + '" /S /allusers') -LogName kicad
  if (-not (Test-Path (Join-Path $pf 'bin\kicad.exe'))) { throw "KiCad installer did not produce $pf" }
}
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item (Join-Path $pf '*') $dst -Recurse -Force
if (-not (Test-Path $kicad) -or ((Get-Item $kicad).VersionInfo.ProductVersion -notlike '10.0.0*')) { throw 'verify failed for kicad-10.0.0' }
Write-Host "OK kicad-10.0.0 @ $dst"
""",
    # Inno Setup "user" installer: installs into the profile of the token it runs under, so from the
    # SYSTEM toolbox it lands in systemprofile; run it in the Administrator desktop session instead.
    "vscode-1.112.0": r"""
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$code = 'C:\Users\User\AppData\Local\Programs\Microsoft VS Code\Code.exe'
if ((Test-Path $code) -and ((Get-Item $code).VersionInfo.ProductVersion -like '1.112.0*')) { Write-Host 'OK vscode-1.112.0 already present'; exit 0 }
$exe = 'C:\WINDOWS\TEMP\VSCodeUserSetup-x64-1.112.0.exe'
if (-not (Test-Path $exe)) { curl.exe --fail --location --silent --show-error -o $exe 'https://update.code.visualstudio.com/1.112.0/win32-x64-user/stable' }
& 'C:\ale-bootstrap\run_interactive.ps1' -CommandLine ('"' + $exe + '" /VERYSILENT /SUPPRESSMSGBOXES /MERGETASKS=!runcode') -LogName vscode
if (-not (Test-Path $code)) { throw 'verify failed for vscode-1.112.0' }
Write-Host "OK vscode-1.112.0 @ $(Split-Path $code -Parent)"
""",
}

# Runs one command line in the autologon Administrator desktop session (session 1) through a one-shot
# scheduled task and waits for it; the Daytona toolbox itself runs as SYSTEM in session 0, where
# some installers (NSIS MultiUser) misbehave.
RUN_INTERACTIVE_PS1 = r"""
param([Parameter(Mandatory=$true)][string]$CommandLine, [string]$LogName = 'oneshot', [int]$TimeoutSeconds = 3600)
$ErrorActionPreference = 'Stop'
$log = "C:\ale-bootstrap\$LogName.interactive.log"
Remove-Item $log -ErrorAction SilentlyContinue
$task = "ale-oneshot-$LogName"
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c "' + $CommandLine + ' > "' + $log + '" 2>&1 & echo rc=%ERRORLEVEL% >> "' + $log + '""')
$pr = New-ScheduledTaskPrincipal -UserId 'Administrator' -LogonType Interactive -RunLevel Highest
Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $task -Action $act -Principal $pr -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero)) | Out-Null
Start-ScheduledTask -TaskName $task
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do { Start-Sleep 3 } while (((Get-ScheduledTask $task).State -ne 'Ready') -and ((Get-Date) -lt $deadline))
if ((Get-ScheduledTask $task).State -ne 'Ready') { throw "interactive command timed out: $CommandLine" }
Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
$out = Get-Content $log -ErrorAction SilentlyContinue
$out | Write-Host
$rc = ($out | Select-String '^rc=(\d+)' | Select-Object -Last 1).Matches.Groups[1].Value
if ($rc -and [int]$rc -ne 0) { throw "interactive command exited ${rc}: $CommandLine" }
"""


def ps_file(sandbox, remote_path: str, args: str = "", timeout: int = 3600, env: dict[str, str] | None = None):
    # a `throw` inside the script must fail the step too (it leaves $LASTEXITCODE untouched)
    env_prefix = "".join(f"$env:{k} = '{v}'; " for k, v in (env or {}).items())
    script = (f"{env_prefix}$env:Path = '{_TOOL_PATH};' + $env:Path; "
              f"try {{ & '{remote_path}' {args} }} catch {{ Write-Host $_; exit 1 }}; "
              f"if ($LASTEXITCODE) {{ exit $LASTEXITCODE }}; exit 0")
    cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{script}"'
    return sandbox.process.exec(cmd, timeout=timeout)


def run_verify(sandbox, verify: str, timeout: int = 300):
    """Run a meta.json verify command with the bootstrap's tools on PATH. The command is written
    to a batch file on the sandbox and run from there: the strings carry nested quotes and pipes
    that an inline cmd /c wrapper mangles, and the toolbox's env parameter replaces the whole
    environment (killing $env:ProgramFiles), so neither works."""
    import hashlib

    name = hashlib.sha1(verify.encode()).hexdigest()[:10]
    script = rf"C:\ale-bootstrap\verify_{name}.cmd"
    body = f"@echo off\r\nset PATH={_TOOL_PATH};%PATH%\r\n{verify}\r\nexit /b %ERRORLEVEL%\r\n"
    sandbox.fs.upload_file(body.encode(), script)
    return sandbox.process.exec(script, timeout=timeout)


def get_sandbox(d, state: dict):
    sid = state.get("sandbox_id")
    if not sid:
        sys.exit("no builder sandbox in state; run `create` first")
    sb = d.get(sid)
    if str(sb.state).lower().endswith("stopped"):
        log(f"starting stopped builder {sid}")
        sb.start(timeout=600)
    return sb


def tasks_from_list(path: Path) -> list[str]:
    return [l.split("#", 1)[0].strip() for l in path.read_text().splitlines() if l.split("#", 1)[0].strip()]


def required_packages(ale: Path, tasks: list[str]) -> list[str]:
    seen: list[str] = []
    for t in tasks:
        card = json.loads((ale / "tasks" / t / "task_card.json").read_text())
        for p in card.get("requiredSystemPackages") or []:
            if p not in seen:
                seen.append(p)
    return seen


# ----------------------------------------------------------------------------- stages


def stage_create(d, state: dict, args) -> None:
    from daytona import CreateSandboxFromSnapshotParams

    if state.get("sandbox_id"):
        log(f"builder already exists: {state['sandbox_id']}")
        return
    log(f"creating builder from {args.base_snapshot}")
    sb = d.create(
        CreateSandboxFromSnapshotParams(
            snapshot=args.base_snapshot,
            labels={"ale": "builder", "ale-target-snapshot": args.name},
            auto_stop_interval=0,
            auto_delete_interval=-1,
            ttl_minutes=args.builder_ttl_minutes,
            target=args.target,
        ),
        timeout=900,
    )
    state.update({"sandbox_id": sb.id, "base_snapshot": args.base_snapshot, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save_state(args.state, state)
    log(f"builder {sb.id} state={sb.state} cpu={sb.cpu} mem={sb.memory} disk={sb.disk}")


def stage_bootstrap(d, state: dict, args) -> None:
    sb = get_sandbox(d, state)
    sb.fs.upload_file(BOOTSTRAP.read_bytes(), REMOTE_BOOTSTRAP)
    log("running bootstrap_runtime.ps1")
    r = ps_file(sb, REMOTE_BOOTSTRAP, timeout=3600)
    print(r.result[-4000:])
    if r.exit_code != 0:
        sys.exit(f"bootstrap failed (exit {r.exit_code})")
    state["bootstrap"] = "ok"
    save_state(args.state, state)


def stage_packages(d, state: dict, args) -> None:
    sb = get_sandbox(d, state)
    ale = Path(args.ale)
    tasks = tasks_from_list(Path(args.tasks))
    pkgs = [p for p in required_packages(ale, tasks) if p not in GPU_ONLY_PACKAGES]
    log(f"packages for {len(tasks)} tasks: {pkgs}")
    src = ale / "env" / "packages-win"
    uploads = []
    from daytona import FileUpload

    for p in ["chocolatey-bootstrap", *pkgs]:
        for f in sorted((src / p).rglob("*")):
            if f.is_file():
                rel = f.relative_to(src)
                uploads.append(FileUpload(source=str(f), destination=str(PureWindowsPath(REMOTE_PACKAGES) / PureWindowsPath(*rel.parts))))
    sb.fs.upload_files(uploads, timeout=600)
    done = set(state.get("packages_done") or [])
    for p in ["chocolatey-bootstrap", *pkgs]:
        if p in done and not args.force:
            log(f"{p}: already installed (state)")
            continue
        log(f"installing {p}")
        t0 = time.time()
        script = rf"{REMOTE_PACKAGES}\{p}\install.ps1"
        sb.fs.upload_file(RUN_INTERACTIVE_PS1.encode(), r"C:\ale-bootstrap\run_interactive.ps1")
        if p in PACKAGE_OVERRIDE:
            script = rf"{REMOTE_PACKAGES}\{p}\install_override.ps1"
            sb.fs.upload_file(PACKAGE_OVERRIDE[p].encode(), script)
        r = ps_file(sb, script, timeout=5400, env=PACKAGE_ENV.get(p))
        tail = (r.result or "")[-1500:]
        print(tail)
        if r.exit_code != 0:
            sys.exit(f"{p}: install.ps1 exit {r.exit_code} after {time.time() - t0:.0f}s")
        meta = json.loads((src / p / "meta.json").read_text())
        verify = meta.get("verify")
        if verify:
            v = run_verify(sb, verify)
            if v.exit_code != 0:
                # install.ps1 does its own checks; some meta.json verify strings are not portable
                # (microsoft-edge-os single-quotes ${env:...}), so record and go on; `verify` reports.
                log(f"{p}: WARNING verify failed (install.ps1 passed): {(v.result or '')[-300:]}")
                state.setdefault("verify_failed", [])
                if p not in state["verify_failed"]:
                    state["verify_failed"].append(p)
        log(f"{p}: ok ({time.time() - t0:.0f}s)")
        done.add(p)
        state["packages_done"] = sorted(done)
        save_state(args.state, state)


def _bake_reference(ref: Path, archive: Path, password: str) -> None:
    """Write the canonical flat, header-encrypted ``reference.7z`` (what the ALE images bake and
    ``baked_in_sandbox.stage_reference`` decrypts with 7z on the sandbox)."""
    for exe in ("7z", "7zz", "7za"):
        if shutil.which(exe):
            subprocess.run([exe, "a", "-t7z", "-mhe=on", f"-p{password}", str(archive), "."], cwd=ref, check=True, stdout=subprocess.DEVNULL)
            return
    import py7zr  # pure-Python fallback (pip install py7zr)

    with py7zr.SevenZipFile(archive, "w", password=password) as z:
        z.set_encrypted_header(True)
        z.writeall(ref, arcname=".")


def _variants(data_root: Path, task: str) -> list[Path]:
    base = data_root / "tasks" / task
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def stage_data(d, state: dict, args) -> None:
    from daytona import FileUpload

    sb = get_sandbox(d, state)
    data_root = Path(args.data)
    tasks = tasks_from_list(Path(args.tasks))
    password = os.environ.get("ALE_REFERENCE_ARCHIVE_PASSWORD")
    done = set(state.get("data_done") or [])
    missing: list[str] = []
    for t in tasks:
        variants = _variants(data_root, t)
        if not variants:
            missing.append(t)
            continue
        for v in variants:
            key = f"{t}/{v.name}"
            if key in done and not args.force:
                continue
            remote_base = PureWindowsPath(DATA_ROOT_REMOTE, *t.split("/"), v.name)
            uploads: list[FileUpload] = []
            nbytes = 0
            for sub in ("input", "software"):
                sdir = v / sub
                if not sdir.is_dir():
                    continue
                for f in sorted(sdir.rglob("*")):
                    if f.is_file() and not f.is_symlink():
                        rel = f.relative_to(v)
                        uploads.append(FileUpload(source=str(f), destination=str(remote_base / PureWindowsPath(*rel.parts))))
                        nbytes += f.stat().st_size
            ref = v / "reference"
            tmp_archive = None
            if ref.is_dir():
                if not password:
                    sys.exit(f"{key}: reference/ present but ALE_REFERENCE_ARCHIVE_PASSWORD is unset")
                tmp_archive = Path(tempfile.mkdtemp()) / "reference.7z"
                _bake_reference(ref, tmp_archive, password)
                uploads.append(FileUpload(source=str(tmp_archive), destination=str(remote_base / "reference.7z")))
                nbytes += tmp_archive.stat().st_size
            log(f"{key}: uploading {len(uploads)} files, {nbytes / 1e6:.1f} MB" + ("" if ref.is_dir() else " (no reference on host)"))
            ps(sb, f"New-Item -ItemType Directory -Force -Path '{remote_base}\\output' | Out-Null", timeout=120)
            batch: list[FileUpload] = []
            bsize = 0
            for u in uploads:
                batch.append(u)
                bsize += Path(u.source).stat().st_size
                if len(batch) >= UPLOAD_BATCH_FILES or bsize >= UPLOAD_BATCH_BYTES:
                    sb.fs.upload_files(batch, timeout=1800)
                    batch, bsize = [], 0
            if batch:
                sb.fs.upload_files(batch, timeout=1800)
            if tmp_archive is not None:
                shutil.rmtree(tmp_archive.parent, ignore_errors=True)
            done.add(key)
            state["data_done"] = sorted(done)
            save_state(args.state, state)
    if missing:
        log(f"WARNING: no data on the host for {len(missing)} tasks: {missing}")
    state["data_missing"] = missing
    save_state(args.state, state)


def stage_verify(d, state: dict, args) -> None:
    sb = get_sandbox(d, state)
    ale = Path(args.ale)
    tasks = tasks_from_list(Path(args.tasks))
    pkgs = [p for p in required_packages(ale, tasks) if p not in GPU_ONLY_PACKAGES]
    failures = []
    for p in pkgs:
        meta = json.loads((ale / "env" / "packages-win" / p / "meta.json").read_text())
        if not meta.get("verify"):
            continue
        v = run_verify(sb, meta["verify"])
        status = "ok" if v.exit_code == 0 else f"FAIL exit {v.exit_code}: {(v.result or '')[-300:]}"
        if v.exit_code != 0 and p in NON_PORTABLE_VERIFY:
            status = f"skipped (non-portable verify string; install.ps1 checked it): {status}"
        log(f"verify {p}: {status}")
        if v.exit_code != 0 and p not in NON_PORTABLE_VERIFY:
            failures.append(p)
    layout = ps(sb, (
        "Test-Path 'E:\\agenthle'; Test-Path 'C:\\Users\\User\\.ale' -PathType Any; "
        "Test-Path 'C:\\Users\\User\\node-v24.12.0-win-x64\\node.exe'; Test-Path 'C:\\ale-run\\.venv\\Scripts\\python.exe'; "
        "(Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:5000/status).Content; "
        "(Get-ChildItem 'E:\\agenthle' -Recurse -Depth 1 -Directory | Measure-Object).Count; "
        "(Get-PSDrive C | Select-Object -ExpandProperty Free)/1GB"
    ), timeout=300)
    print(layout.result)
    if failures:
        sys.exit(f"verify failed for {failures}")
    state["verify"] = "ok"
    save_state(args.state, state)


def stage_snapshot(d, state: dict, args) -> None:
    sb = get_sandbox(d, state)
    log("cleaning builder scratch (downloads, pip caches)")
    ps(sb, "Remove-Item -Recurse -Force 'C:\\ale-bootstrap\\dl','C:\\AgentHLE\\dl' -ErrorAction SilentlyContinue; "
           "Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\pip\\Cache\" -ErrorAction SilentlyContinue", timeout=600)
    log(f"stopping builder {sb.id}")
    sb.stop(timeout=600)
    sb.wait_for_sandbox_stop(timeout=600)
    log(f"creating snapshot {args.name} (filesystem only)")
    t0 = time.time()
    sb.create_snapshot(args.name, timeout=0)
    state["snapshot"] = args.name
    state["snapshot_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(args.state, state)
    log(f"snapshot {args.name} created in {time.time() - t0:.0f}s; builder {sb.id} left stopped")


STAGES = {
    "create": stage_create,
    "bootstrap": stage_bootstrap,
    "packages": stage_packages,
    "data": stage_data,
    "verify": stage_verify,
    "snapshot": stage_snapshot,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--ale", default=str(HERE.parents[1]), help="ALE checkout (tasks/, env/packages-win)")
    ap.add_argument("--data", required=True, help="host task data root (tasks/<domain>/<task>/<variant>/...)")
    ap.add_argument("--tasks", default=str(HERE.parents[1] / "selected_tasks" / "daytona_windows_cpu.txt"))
    ap.add_argument("--name", default=time.strftime("ale-win-%Y%m%d"), help="Daytona snapshot name to create")
    ap.add_argument("--base-snapshot", default="windows-large", help="stock Daytona Windows snapshot; this org allows at most 4 vCPU per sandbox")
    ap.add_argument("--target", default=None)
    ap.add_argument("--builder-ttl-minutes", type=int, default=48 * 60)
    ap.add_argument("--state", type=Path, default=HERE / "builder_state.json")
    ap.add_argument("--force", action="store_true", help="re-run packages/data even if the state says done")
    args = ap.parse_args()
    stages = list(STAGES) if "all" in args.stages else args.stages
    d = client()
    state = load_state(args.state)
    for s in stages:
        log(f"== stage {s}")
        STAGES[s](d, state, args)
    log("done")


if __name__ == "__main__":
    main()
