"""Run ALE's in-box infrastructure under renamed interpreters so agents' python-wide kills miss it.

C:\\ale-py  = copy of C:\\Python312 with alepy.exe/alepyw.exe + the sandbox worker's deps (no venv stub).
C:\\cua-py  = copy of C:\\Python312 with cuasrv.exe/cuasrvw.exe + cua-computer-server; run_cua_server.cmd re-pointed.
Then snapshot."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_snapshot import client, load_state, log, ps, save_state

STATE = Path(__file__).resolve().parent / "builder_state.json"
name = sys.argv[1] if len(sys.argv) > 1 else "ale-win-20261025d"
d = client(); state = load_state(STATE)
sb = d.get(state["sandbox_id"])
for i in range(240):
    try:
        if "stopped" in str(sb.state).lower():
            sb.start(timeout=900)
        break
    except Exception as e:
        if "limit exceeded" in str(e).lower():
            if i % 10 == 0: log("waiting for org CPU capacity to start the builder")
            time.sleep(30); sb = d.get(state["sandbox_id"]); continue
        raise
log(f"builder {sb.id} state={sb.state}")
script = r'''
$ErrorActionPreference='Continue'; $ProgressPreference='SilentlyContinue'
foreach ($spec in @(@{dst='C:\ale-py'; exe='alepy'; pkgs=@('pydantic','requests','PyYAML','pypdf')},
                    @{dst='C:\cua-py'; exe='cuasrv'; pkgs=@('cua-computer-server==0.3.39','fastapi<0.130','starlette<1.0')})) {
  $dst=$spec.dst; $exe=$spec.exe
  if (-not (Test-Path "$dst\$exe.exe")) {
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Write-Output ("copying C:\Python312 -> " + $dst)
    robocopy C:\Python312 $dst /E /NFL /NDL /NJH /NJS /NP | Out-Null
    Copy-Item "$dst\python.exe" "$dst\$exe.exe" -Force
    Copy-Item "$dst\pythonw.exe" "$dst\${exe}w.exe" -Force
  }
  Write-Output ("installing packages into " + $dst)
  # pip prints retry warnings on stderr; with ErrorActionPreference=Stop those become
  # terminating errors, so run it with Continue and judge by the exit code only.
  $out = & "$dst\$exe.exe" -m pip install --quiet --no-warn-script-location --retries 10 --timeout 60 @($spec.pkgs) 2>&1 | Out-String
  $rc = $LASTEXITCODE
  Write-Output ($out.Substring([Math]::Max(0, $out.Length - 600)))
  if ($rc -ne 0) { throw ("pip failed for " + $dst + " rc=" + $rc) }
  & "$dst\$exe.exe" -c "import sys; print(sys.executable)"
}
$chk = & C:\cua-py\cuasrv.exe -c "import computer_server, fastapi; print('cua ok', fastapi.__version__)" 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw ("cua import check failed: " + $chk) }
Write-Output $chk
$chk2 = & C:\ale-py\alepy.exe -c "import pydantic, requests, yaml, pypdf; print('worker deps ok')" 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw ("worker import check failed: " + $chk2) }
Write-Output $chk2
Set-Content -Path 'C:\cua-server\run_cua_server.cmd' -Value "@echo off`r`n:loop`r`n`"C:\cua-py\cuasrv.exe`" -m computer_server --port 5000 >> C:\cua-server\server.log 2>&1`r`ntimeout /t 5 /nobreak >nul`r`ngoto loop`r`n"
Get-Content 'C:\cua-server\run_cua_server.cmd'
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\pip\Cache" -ErrorAction SilentlyContinue
Write-Output 'HARDEN_OK'
'''
ps(sb, r"New-Item -ItemType Directory -Force 'C:\ale-bootstrap' | Out-Null; Remove-Item 'C:\ale-bootstrap\hp.done','C:\ale-bootstrap\hp.log' -ErrorAction SilentlyContinue", timeout=60)
sb.fs.upload_file(script.encode("utf-8"), r"C:\ale-bootstrap\harden.ps1")
ps(sb, r"Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command','& { try { & C:\ale-bootstrap\harden.ps1 *> C:\ale-bootstrap\hp.log; Set-Content C:\ale-bootstrap\hp.done 0 } catch { $_ | Out-File C:\ale-bootstrap\hp.log -Append; Set-Content C:\ale-bootstrap\hp.done 1 } }' -WindowStyle Hidden", timeout=120)
log("hardening started detached; polling")
for i in range(120):
    time.sleep(20)
    r = ps(sb, r"if (Test-Path 'C:\ale-bootstrap\hp.done') { Get-Content 'C:\ale-bootstrap\hp.done' } else { 'pending' }", timeout=60)
    out = (r.result or "").strip()
    if out != "pending": break
    if i % 6 == 5: log(f"still running ({(i+1)*20}s)")
else:
    sys.exit("hardening did not finish in 40 min")
r = ps(sb, r"Get-Content 'C:\ale-bootstrap\hp.log' | Select-Object -Last 25", timeout=60)
log(f"hardening rc={out}\n{(r.result or '')[-2500:]}")
if out != "0" or "HARDEN_OK" not in (r.result or ""): sys.exit("hardening failed")
# restart the cua server under the new interpreter and check it answers
ps(sb, r"Stop-ScheduledTask -TaskName CuaServer-ale -ErrorAction SilentlyContinue; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'computer_server|run_cua_server' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep 2; Start-ScheduledTask -TaskName CuaServer-ale", timeout=120)
ok = False
for i in range(30):
    time.sleep(5)
    r = sb.process.exec("curl.exe -s -m 5 http://127.0.0.1:5000/status", timeout=30)
    if r.exit_code == 0 and '"status":"ok"' in (r.result or "").replace(" ", ""): ok = True; break
log(f"cua server under cuasrv.exe answering: {ok}")
r = ps(sb, r"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'computer_server' } | Select-Object Name,ProcessId | Format-Table | Out-String", timeout=60)
log((r.result or "").strip())
if not ok: sys.exit("cua server did not come up under cuasrv.exe")
state.setdefault("packages_done", []).append("hardened-python")
log(f"stopping builder {sb.id}"); sb.stop(timeout=600); sb.wait_for_sandbox_stop(timeout=600)
t0 = time.time(); log(f"creating snapshot {name}"); sb.create_snapshot(name, timeout=0)
state["snapshot"] = name; state["snapshot_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save_state(STATE, state)
log(f"snapshot {name} created in {time.time()-t0:.0f}s")
