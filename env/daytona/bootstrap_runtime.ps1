# ALE runtime bootstrap for a Daytona Windows sandbox (Windows Server 2025, autologon user Administrator).
#
# Reproduces the ale-win10 image conventions ale_run relies on (ale_run/environments/images/ale_win10.py):
#   C:\Users\User\.ale                      per-run scratch      -> C:\Users\User is a junction onto the Administrator profile
#   E:\agenthle                             task data root       -> E: is a persistent DOS-device mapping onto C:\agenthle
#   C:\Users\User\node-v24.12.0-win-x64     Node 24.12.0
#   C:\ale-run\.venv\Scripts\python.exe     framework venv (pydantic, requests, PyYAML)
#   C:\Users\User\cua_mcp_server            installed by the deployer at run time
#   cua computer-server on :5000            C:\cua-server\.venv, interactive scheduled task CuaServer-ale
# plus 7-Zip on PATH (reference.7z) and C:\Softwares (some graders glob it).
# Idempotent: every step checks before acting. Runs elevated (Daytona's toolbox runs as SYSTEM).
param(
  [string]$CuaServerVersion = "0.3.39",
  [string]$ProfileDir = "C:\Users\Administrator",
  [string]$DriveRoot = "C:\drive-e"     # folder the E: drive letter maps onto; task data root is E:\agenthle
)
$DataRoot = Join-Path $DriveRoot 'agenthle'
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$dl = 'C:\ale-bootstrap\dl'; New-Item -ItemType Directory -Force -Path $dl | Out-Null
function Step($msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $msg) }

# --- 0. Path conventions: C:\Users\User -> profile, E: -> $DataRoot, C:\Softwares ---
if (-not (Test-Path 'C:\Users\User')) {
  Step "junction C:\Users\User -> $ProfileDir"
  cmd /c mklink /J C:\Users\User $ProfileDir | Out-Null
}
New-Item -ItemType Directory -Force -Path $DataRoot, 'C:\Softwares' | Out-Null
# A global, boot-persistent drive letter (Session Manager DOS Devices), so every logon session sees
# E:\agenthle == $DataRoot. `subst` only covers the current (SYSTEM) session until the next boot.
$dos = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\DOS Devices'
if ((Get-ItemProperty -Path $dos -Name 'E:' -ErrorAction SilentlyContinue).'E:' -ne "\??\$DriveRoot") {
  Step "mapping E: -> $DriveRoot (persistent DOS device)"
  New-ItemProperty -Path $dos -Name 'E:' -Value "\??\$DriveRoot" -PropertyType String -Force | Out-Null
}
if (-not (Test-Path 'E:\')) { cmd /c "subst E: $DriveRoot" | Out-Null }
if (-not (Test-Path 'E:\agenthle')) { throw 'E: drive mapping failed' }

# --- 1. Python 3.12 (all users, py launcher, on PATH) ---
$py = 'C:\Python312\python.exe'
if (-not (Test-Path $py)) {
  Step 'installing Python 3.12.10'
  $exe = "$dl\python-3.12.10-amd64.exe"
  Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $exe
  $p = Start-Process $exe -ArgumentList '/quiet','InstallAllUsers=1','TargetDir=C:\Python312','PrependPath=1','Include_launcher=1','InstallLauncherAllUsers=1','Include_test=0' -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "python installer exit $($p.ExitCode)" }
}
& $py --version

# --- 2. framework venv C:\ale-run\.venv ---
$fw = 'C:\ale-run\.venv\Scripts\python.exe'
if (-not (Test-Path $fw)) {
  Step 'creating C:\ale-run\.venv'
  & $py -m venv C:\ale-run\.venv
  & $fw -m pip install -q --upgrade pip
  & $fw -m pip install -q pydantic requests PyYAML
}
& $fw -c "import pip, pydantic, requests; print('framework venv ok')"

# --- 3. Node 24.12.0 x64 portable under the profile dir (ale image convention) ---
$nodeDir = "$ProfileDir\node-v24.12.0-win-x64"
if (-not (Test-Path "$nodeDir\node.exe")) {
  Step 'installing Node 24.12.0'
  $zip = "$dl\node-v24.12.0-win-x64.zip"
  Invoke-WebRequest -UseBasicParsing 'https://nodejs.org/dist/v24.12.0/node-v24.12.0-win-x64.zip' -OutFile $zip
  Expand-Archive -Force $zip $ProfileDir
}
& "$nodeDir\node.exe" --version

# --- 4. Chocolatey + 7-Zip (env/packages-win installers assume both) ---
$choco = "$env:ProgramData\chocolatey\bin\choco.exe"
if (-not (Test-Path $choco)) {
  Step 'bootstrapping chocolatey'
  Set-ExecutionPolicy Bypass -Scope Process -Force
  iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}
if (-not (Test-Path 'C:\Program Files\7-Zip\7z.exe')) {
  Step 'installing 7zip'
  & $choco install -y 7zip --no-progress | Out-Null
}
$machinePath = [Environment]::GetEnvironmentVariable('Path','Machine')
if ($machinePath -notlike '*7-Zip*') { [Environment]::SetEnvironmentVariable('Path', "$machinePath;C:\Program Files\7-Zip", 'Machine') }

# --- 5. CUA computer-server venv ---
# cua-computer-server 0.3.39 passes websocket_max_size= to FastAPI(); FastAPI >= 0.130 / Starlette 1.x
# reject it, so pin the web stack to the versions the ALE images shipped with.
$cua = 'C:\cua-server\.venv\Scripts\python.exe'
if (-not (Test-Path $cua)) {
  Step "installing cua-computer-server $CuaServerVersion"
  & $py -m venv C:\cua-server\.venv
  & $cua -m pip install -q --upgrade pip
  & $cua -m pip install -q "cua-computer-server==$CuaServerVersion" "fastapi<0.130" "starlette<1.0"
}
# The import emits deprecation warnings on stderr; under $ErrorActionPreference=Stop PowerShell would turn
# native stderr into a terminating error, so relax it around the probe and judge by the exit code only.
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
& $cua -W ignore -c "import computer_server; print('computer_server ok')" 2>&1 | Out-Null
$importRc = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($importRc -ne 0) { throw 'computer_server import failed' }
Write-Host 'computer_server ok'

# --- 6. Run it in the interactive desktop session as Administrator (scheduled task, like the ALE images) ---
$script = 'C:\cua-server\run_cua_server.cmd'
Set-Content -Path $script -Value "@echo off`r`n:loop`r`n`"$cua`" -m computer_server --port 5000 >> C:\cua-server\server.log 2>&1`r`ntimeout /t 5 /nobreak >nul`r`ngoto loop`r`n"
if (-not (Get-ScheduledTask -TaskName 'CuaServer-ale' -ErrorAction SilentlyContinue)) {
  Step 'registering CuaServer-ale scheduled task'
  $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$script`""
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User 'Administrator'
  $principal = New-ScheduledTaskPrincipal -UserId 'Administrator' -LogonType Interactive -RunLevel Highest
  $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
  Register-ScheduledTask -TaskName 'CuaServer-ale' -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
}
if (-not (Get-NetFirewallRule -DisplayName 'ALE cua-server 5000' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName 'ALE cua-server 5000' -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow | Out-Null
}
$up = $false
try { $up = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 'http://127.0.0.1:5000/status').StatusCode -eq 200 } catch {}
if (-not $up) { Start-ScheduledTask -TaskName 'CuaServer-ale' }
$ok = $false
foreach ($i in 1..60) {
  try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 'http://127.0.0.1:5000/status'; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch {}
  Start-Sleep -Seconds 2
}
if (-not $ok) { Get-Content C:\cua-server\server.log -Tail 40; throw 'cua-server did not answer /status on 5000' }
Step "cua-server up: $($r.Content)"
Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,SessionId,Path | Format-Table | Out-String | Write-Host
Step 'bootstrap complete'
