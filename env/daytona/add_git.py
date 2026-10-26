"""Install Git for Windows (Claude Code's Bash tool needs git-bash) into the builder and re-snapshot."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_snapshot import client, get_sandbox, load_state, log, ps, save_state

STATE = Path(__file__).resolve().parent / "builder_state.json"
name = sys.argv[1] if len(sys.argv) > 1 else "ale-win-20261024b"
d = client()
state = load_state(STATE)
sb = get_sandbox(d, state)
log(f"builder {sb.id} state={sb.state}")
ps(sb, r"New-Item -ItemType Directory -Force 'C:\ale-bootstrap' | Out-Null; Remove-Item 'C:\ale-bootstrap\git.done','C:\ale-bootstrap\git.log' -ErrorAction SilentlyContinue; "
       r"$cmd = '$env:Path = ''C:\ProgramData\chocolatey\bin;'' + $env:Path; choco install git -y --no-progress --params ''/GitAndUnixToolsOnPath /NoAutoCrlf'' *> C:\ale-bootstrap\git.log; Set-Content C:\ale-bootstrap\git.done $LASTEXITCODE'; "
       r"Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',$cmd -WindowStyle Hidden", timeout=120)
log("choco install git started detached; polling")
for i in range(90):
    time.sleep(20)
    r = ps(sb, r"if (Test-Path 'C:\ale-bootstrap\git.done') { Get-Content 'C:\ale-bootstrap\git.done' } else { 'pending' }", timeout=60)
    out = (r.result or '').strip()
    if out != 'pending':
        break
    if i % 6 == 5:
        log(f"still installing ({(i+1)*20}s)")
else:
    sys.exit("choco install git did not finish in 30 min")
r = ps(sb, r"Get-Content 'C:\ale-bootstrap\git.log' | Select-Object -Last 15", timeout=60)
log(f"choco done rc={out}\n{(r.result or '')[-1500:]}")
log(f"choco exit={r.exit_code}\n{(r.result or '')[-1500:]}")
r = ps(sb, r"Get-Item 'C:\Program Files\Git\bin\bash.exe','C:\Program Files\Git\cmd\git.exe' | Select-Object -ExpandProperty FullName; & 'C:\Program Files\Git\cmd\git.exe' --version", timeout=300)
log(f"verify exit={r.exit_code}\n{(r.result or '')[-800:]}")
if r.exit_code != 0 or "git version" not in (r.result or ""):
    sys.exit("git verify failed")
state.setdefault("packages_done", []).append("git-for-windows")
save_state(STATE, state)
log("cleaning choco temp")
ps(sb, r"Remove-Item -Recurse -Force 'C:\ProgramData\chocolatey\lib-bad','C:\Users\Administrator\AppData\Local\Temp\chocolatey' -ErrorAction SilentlyContinue", timeout=600)
log(f"stopping builder {sb.id}")
sb.stop(timeout=600)
sb.wait_for_sandbox_stop(timeout=600)
t0 = time.time()
log(f"creating snapshot {name}")
sb.create_snapshot(name, timeout=0)
state["snapshot"] = name
state["snapshot_created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save_state(STATE, state)
log(f"snapshot {name} created in {time.time()-t0:.0f}s")
