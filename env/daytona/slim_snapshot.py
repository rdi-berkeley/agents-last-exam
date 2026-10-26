"""Free disk on the builder (50 GB fixed; the c snapshot leaves ~7 GB) and re-snapshot."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_snapshot import client, load_state, log, ps, save_state

STATE = Path(__file__).resolve().parent / "builder_state.json"
name = sys.argv[1] if len(sys.argv) > 1 else "ale-win-20261025e"
d = client(); state = load_state(STATE)
sb = d.get(state["sandbox_id"])
for i in range(240):
    try:
        if "stopped" in str(sb.state).lower(): sb.start(timeout=900)
        break
    except Exception as e:
        if "limit exceeded" in str(e).lower():
            if i % 10 == 0: log("waiting for org CPU capacity to start the builder")
            time.sleep(30); sb = d.get(state["sandbox_id"]); continue
        raise
log(f"builder {sb.id} state={sb.state}")
def free(): 
    r = ps(sb, r"$d=Get-PSDrive C; 'free {0:N1} GB / {1:N1} GB' -f ($d.Free/1GB), (($d.Used+$d.Free)/1GB)", timeout=60); return (r.result or '').strip()
log("before: " + free())
# (usage breakdown is produced inside the detached script; a direct toolbox exec over the whole disk exceeds the gateway's request limit)
script = r'''
$ErrorActionPreference='Continue'
Get-ChildItem C:\ -Force -File -ErrorAction SilentlyContinue | Select-Object Name,@{n='GB';e={[math]::Round($_.Length/1GB,2)}} | Format-Table -AutoSize | Out-String
foreach ($p in 'C:\Windows\WinSxS','C:\Windows\SoftwareDistribution','C:\Windows\Temp','C:\Windows\Installer','C:\ProgramData\chocolatey','C:\ale-run','C:\cua-server','C:\ale-bootstrap','C:\Users\Administrator\.npm-ale','C:\Users\Administrator\AppData\Local\Temp','C:\Program Files\Blender Foundation','C:\Softwares','C:\drive-e','C:\Python312','C:\ale-py','C:\cua-py','C:\Program Files','C:\Users') { if (Test-Path $p) { $s=(Get-ChildItem $p -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum; '{0,-55} {1,8:N2} GB' -f $p, ($s/1GB) } }
powercfg /h off 2>$null
# fixed 1 GB pagefile instead of the system-managed one
$cs = Get-WmiObject Win32_ComputerSystem; $cs.AutomaticManagedPagefile = $false; $cs.Put() | Out-Null
$pf = Get-WmiObject Win32_PageFileSetting -ErrorAction SilentlyContinue
if ($pf) { $pf.InitialSize = 1024; $pf.MaximumSize = 1024; $pf.Put() | Out-Null } else { Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{Name='C:\pagefile.sys'; InitialSize=1024; MaximumSize=1024} | Out-Null }
Stop-Service wuauserv -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\Windows\SoftwareDistribution\Download\*' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\Windows\Temp\*','C:\Users\Administrator\AppData\Local\Temp\*','C:\ale-bootstrap\dl','C:\Users\Administrator\.npm-ale\_cacache','C:\ProgramData\chocolatey\lib-bad','C:\ProgramData\chocolatey\logs\*' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'C:\ale-run\.venv','C:\cua-server\.venv' -ErrorAction SilentlyContinue
Get-ChildItem 'C:\ProgramData\chocolatey\lib' -Recurse -Include *.exe,*.msi,*.zip,*.7z -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match '\\tools\\' } | Remove-Item -Force -ErrorAction SilentlyContinue
Dism.exe /Online /Cleanup-Image /StartComponentCleanup /ResetBase 2>&1 | Select-Object -Last 3
Write-Output 'SLIM_OK'
'''
sb.fs.upload_file(script.encode("utf-8"), r"C:\ale-bootstrap\slim.ps1")
ps(sb, r"Remove-Item 'C:\ale-bootstrap\slim.done','C:\ale-bootstrap\slim.log' -ErrorAction SilentlyContinue; Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command','& { try { & C:\ale-bootstrap\slim.ps1 *> C:\ale-bootstrap\slim.log; Set-Content C:\ale-bootstrap\slim.done 0 } catch { $_ | Out-File C:\ale-bootstrap\slim.log -Append; Set-Content C:\ale-bootstrap\slim.done 1 } }' -WindowStyle Hidden", timeout=120)
log("slimming started detached (Dism cleanup takes a while); polling")
for i in range(150):
    time.sleep(20)
    r = ps(sb, r"if (Test-Path 'C:\ale-bootstrap\slim.done') { Get-Content 'C:\ale-bootstrap\slim.done' } else { 'pending' }", timeout=60)
    out = (r.result or "").strip()
    if out != "pending": break
    if i % 9 == 8: log(f"still slimming ({(i+1)*20}s)")
else:
    sys.exit("slimming did not finish in 50 min")
r = ps(sb, r"Get-Content 'C:\ale-bootstrap\slim.log' | Select-Object -Last 45", timeout=60)
log(f"slim rc={out}\n{(r.result or '')[-4000:]}")
log("after (pagefile change applies after reboot): " + free())
log("restarting builder so the pagefile setting takes effect"); sb.stop(timeout=600); sb.wait_for_sandbox_stop(timeout=600)
for i in range(240):
    try: sb.start(timeout=900); break
    except Exception as e:
        if "limit exceeded" in str(e).lower(): time.sleep(30); continue
        raise
time.sleep(20)
log("after reboot: " + free())
r = sb.process.exec("curl.exe -s -m 5 http://127.0.0.1:5000/status", timeout=30); log(f"cua status rc={r.exit_code} {(r.result or '')[:60]}")
log(f"stopping builder {sb.id}"); sb.stop(timeout=600); sb.wait_for_sandbox_stop(timeout=600)
t0 = time.time(); log(f"creating snapshot {name}"); sb.create_snapshot(name, timeout=0)
state["snapshot"] = name; state["snapshot_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save_state(STATE, state)
log(f"snapshot {name} created in {time.time()-t0:.0f}s")
