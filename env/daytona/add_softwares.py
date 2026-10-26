"""Bake Godot 4.6.2 + Blender 4.5.1 LTS portable builds under C:\\Softwares (atlas_outpost task paths) and re-snapshot."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_snapshot import client, get_sandbox, load_state, log, ps, save_state

STATE = Path(__file__).resolve().parent / "builder_state.json"
name = sys.argv[1] if len(sys.argv) > 1 else "ale-win-20261024c"
d = client(); state = load_state(STATE); sb = get_sandbox(d, state)
log(f"builder {sb.id} state={sb.state}")
script = r'''
$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'
New-Item -ItemType Directory -Force 'C:\Softwares','C:\ale-bootstrap\dl' | Out-Null
$dl='C:\ale-bootstrap\dl'
$items = @(
  @{url='https://github.com/godotengine/godot/releases/download/4.6.2-stable/Godot_v4.6.2-stable_win64.exe.zip'; zip="$dl\godot.zip"; dest='C:\Softwares\Godot-4.6.2'; exe='godot.exe'; src='Godot_v4.6.2-stable_win64.exe'},
  @{url='https://download.blender.org/release/Blender4.5/blender-4.5.1-windows-x64.zip'; zip="$dl\blender451.zip"; dest='C:\Softwares\Blender-4.5.1'; exe='blender.exe'; src=''}
)
foreach ($it in $items) {
  if (Test-Path (Join-Path $it.dest $it.exe)) { Write-Output ('present ' + $it.dest); continue }
  Write-Output ('downloading ' + $it.url)
  Invoke-WebRequest -UseBasicParsing -Uri $it.url -OutFile $it.zip
  $tmp = $it.zip + '.x'; if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  Expand-Archive -LiteralPath $it.zip -DestinationPath $tmp -Force
  New-Item -ItemType Directory -Force $it.dest | Out-Null
  if ($it.src) {
    Move-Item -Force (Join-Path $tmp $it.src) (Join-Path $it.dest $it.exe)
    Get-ChildItem $tmp | Move-Item -Destination $it.dest -Force -ErrorAction SilentlyContinue
  } else {
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Get-ChildItem $inner.FullName | Move-Item -Destination $it.dest -Force
  }
  Remove-Item -Recurse -Force $tmp
  Write-Output ('installed ' + $it.dest + ' -> ' + (Test-Path (Join-Path $it.dest $it.exe)))
}
Remove-Item -Force "$dl\godot.zip","$dl\blender451.zip" -ErrorAction SilentlyContinue
Get-ChildItem C:\Softwares | Select-Object -ExpandProperty Name
'''
ps(sb, r"New-Item -ItemType Directory -Force 'C:\ale-bootstrap' | Out-Null; Remove-Item 'C:\ale-bootstrap\sw.done','C:\ale-bootstrap\sw.log' -ErrorAction SilentlyContinue", timeout=60)
sb.fs.upload_file(script.encode("utf-8"), r"C:\ale-bootstrap\add_softwares.ps1")
ps(sb, r"Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command','& { try { & C:\ale-bootstrap\add_softwares.ps1 *> C:\ale-bootstrap\sw.log; Set-Content C:\ale-bootstrap\sw.done 0 } catch { $_ | Out-File C:\ale-bootstrap\sw.log -Append; Set-Content C:\ale-bootstrap\sw.done 1 } }' -WindowStyle Hidden", timeout=120)
log("install started detached; polling")
for i in range(120):
    time.sleep(20)
    r = ps(sb, r"if (Test-Path 'C:\ale-bootstrap\sw.done') { Get-Content 'C:\ale-bootstrap\sw.done' } else { 'pending' }", timeout=60)
    out = (r.result or "").strip()
    if out != "pending": break
    if i % 6 == 5: log(f"still installing ({(i+1)*20}s)")
else:
    sys.exit("install did not finish in 40 min")
r = ps(sb, r"Get-Content 'C:\ale-bootstrap\sw.log' | Select-Object -Last 20", timeout=60)
log(f"install rc={out}\n{(r.result or '')[-2000:]}")
if out != "0": sys.exit("install failed")
r = ps(sb, r"Test-Path 'C:\Softwares\Godot-4.6.2\godot.exe'; Test-Path 'C:\Softwares\Blender-4.5.1\blender.exe'", timeout=60)
log(f"verify: {(r.result or '').split()}")
if (r.result or "").split() != ["True", "True"]: sys.exit("verify failed")
state.setdefault("packages_done", []).extend(["godot-4.6.2", "blender-4.5.1"])
save_state(STATE, state)
log(f"stopping builder {sb.id}"); sb.stop(timeout=600); sb.wait_for_sandbox_stop(timeout=600)
t0 = time.time(); log(f"creating snapshot {name}"); sb.create_snapshot(name, timeout=0)
state["snapshot"] = name; state["snapshot_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save_state(STATE, state)
log(f"snapshot {name} created in {time.time()-t0:.0f}s")
