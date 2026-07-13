"""Guest setup/teardown scripts for the DNS-override egress design.

Same idea on both OSes:
  1. resolve each allowed host's REAL IP (before the hosts override),
  2. point the host at 127.0.0.1 in the guest ``hosts`` file,
  3. run the passthrough proxy on 127.0.0.1:443,
  4. default-deny egress, allowing only the proxy + DNS + loopback + established.
``off`` = default-deny with no proxy/hosts (air-gap; the cua ingress survives).

Pure string builders → unit-testable on the host.
"""
from __future__ import annotations

_LINUX_HOSTS = "/etc/hosts"
_WIN_HOSTS = r"C:\Windows\System32\drivers\etc\hosts"


def _csv(hosts: list[str]) -> str:
    return ",".join(sorted({h.strip().lower() for h in hosts if h.strip()}))


def build_linux(mode: str, hosts: list[str], proxy_b64: str) -> str:
    csv = _csv(hosts)
    if mode == "off":
        block = ""
    else:
        block = (
            f"base64 -d > /opt/netguard/proxy.py <<'NGB64'\n{proxy_b64}\nNGB64\n"
            "id -u netguard >/dev/null 2>&1 || useradd -r -M -s /usr/sbin/nologin netguard\n"
            "sysctl -w net.ipv4.ip_unprivileged_port_start=443 >/dev/null 2>&1 || true\n"
            # resolve REAL ips before the hosts override, then point host at loopback
            f"python3 /opt/netguard/proxy.py --resolve '{csv}' --map /etc/netguard/map.json\n"
            "sed -i '/# netguard$/d' /etc/hosts\n"
            f"for h in $(echo '{csv}' | tr ',' ' '); do echo \"127.0.0.1 $h # netguard\" >> /etc/hosts; done\n"
            "touch /var/log/netguard.log && chown netguard /var/log/netguard.log\n"
            "pkill -f netguard/proxy.py 2>/dev/null || true\n"
            "setsid runuser -u netguard -- python3 /opt/netguard/proxy.py "
            "--map /etc/netguard/map.json --log /var/log/netguard.log >/tmp/netguard.boot 2>&1 &\n"
            "for i in $(seq 1 100); do ss -ltn 2>/dev/null | grep -q '127.0.0.1:443 ' && break; sleep 0.1; done\n"
            "ss -ltn 2>/dev/null | grep -q '127.0.0.1:443 ' || "
            "{ echo NETGUARD_PROXY_FAILED; cat /tmp/netguard.boot /var/log/netguard.log 2>/dev/null; exit 1; }\n"
        )
    proxy_rule = "    meta skuid $NGUID accept\n" if mode != "off" else ""
    dns_rule = "    udp dport 53 accept\n    tcp dport 53 accept\n" if mode != "off" else ""
    return (
        "set -e\n"
        "mkdir -p /opt/netguard /etc/netguard\n"
        "sysctl -w net.ipv6.conf.all.disable_ipv6=1 >/dev/null 2>&1 || true\n"
        + block +
        "NGUID=$(id -u netguard 2>/dev/null || echo 0)\n"
        "nft -f - <<NGNFT\n"
        "flush ruleset\n"
        "table inet netguard {\n"
        "  chain out {\n"
        "    type filter hook output priority 0; policy drop;\n"
        "    oif lo accept\n"
        "    ct state established,related accept\n"
        + proxy_rule + dns_rule +
        "  }\n"
        "}\n"
        "NGNFT\n"
        "echo NETGUARD_OK\n"
    )


def build_linux_teardown() -> str:
    return (
        "nft delete table inet netguard 2>/dev/null || true\n"
        "sed -i '/# netguard$/d' /etc/hosts 2>/dev/null || true\n"
        "pkill -f netguard/proxy.py 2>/dev/null || true\n"
        "echo NETGUARD_CLEARED\n"
    )


def build_windows(mode: str, hosts: list[str], proxy_b64: str) -> str:
    csv = _csv(hosts)
    launch = ""
    if mode != "off":
        hosts_ps = "; ".join(f"Add-Content $h \"127.0.0.1 {x} # netguard\"" for x in csv.split(",") if x)
        launch = (
            f"[IO.File]::WriteAllBytes('C:\\netguard\\proxy.py',[Convert]::FromBase64String('{proxy_b64}'))\n"
            "$py='C:\\ale-run\\.venv\\Scripts\\python.exe'; if(-not(Test-Path $py)){$py=(Get-Command python).Source}\n"
            f"& $py C:\\netguard\\proxy.py --resolve '{csv}' --map C:\\netguard\\map.json\n"
            "$h='C:\\Windows\\System32\\drivers\\etc\\hosts'\n"
            "(Get-Content $h) | Where-Object {$_ -notmatch '# netguard$'} | Set-Content $h\n"
            f"{hosts_ps}\n"
            "ipconfig /flushdns | Out-Null\n"
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object {$_.CommandLine -like '*netguard*proxy.py*'} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }\n"
            "Start-Process -FilePath $py -ArgumentList '\"C:\\netguard\\proxy.py\"','--map','\"C:\\netguard\\map.json\"','--log','\"C:\\netguard\\netguard.log\"' -WindowStyle Hidden\n"
            "for($i=0;$i -lt 40;$i++){ if(Test-NetConnection 127.0.0.1 -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue){break}; Start-Sleep -Milliseconds 250 }\n"
            "if(-not (Test-NetConnection 127.0.0.1 -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue)){ Write-Output NETGUARD_PROXY_FAILED; Get-Content C:\\netguard\\netguard.log -EA SilentlyContinue; exit 1 }\n"
            # Allow the proxy's outbound. The venv launcher can re-exec, so the
            # process that opens the upstream socket may be a different python.exe
            # than $py — cover every real python path or its egress is denied
            # (WinError 5).
            "$pyPaths=@($py,\"$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe\",'C:\\Program Files\\Python312\\python.exe')|Where-Object{Test-Path $_}|Select-Object -Unique\n"
            "$i=0; foreach($pp in $pyPaths){ New-NetFirewallRule -DisplayName \"netguard-proxy-$i\" -Direction Outbound -Program $pp -Action Allow -Profile Any | Out-Null; $i++ }\n"
        )
    return (
        "$ErrorActionPreference='Stop'\n"
        "New-Item -ItemType Directory -Force -Path C:\\netguard | Out-Null\n"
        "New-NetFirewallRule -DisplayName netguard-dns-udp -Direction Outbound -Protocol UDP -RemotePort 53 -Action Allow -Profile Any | Out-Null\n"
        "New-NetFirewallRule -DisplayName netguard-dns-tcp -Direction Outbound -Protocol TCP -RemotePort 53 -Action Allow -Profile Any | Out-Null\n"
        + launch +
        "Set-NetFirewallProfile -All -Enabled True -DefaultInboundAction Allow -DefaultOutboundAction Block\n"
        "Write-Output NETGUARD_OK\n"
    )


def build_windows_teardown() -> str:
    return (
        "Set-NetFirewallProfile -All -DefaultOutboundAction Allow -EA SilentlyContinue\n"
        "Remove-NetFirewallRule -DisplayName 'netguard-*' -EA SilentlyContinue\n"
        "$h='C:\\Windows\\System32\\drivers\\etc\\hosts'; (Get-Content $h)|Where-Object{$_ -notmatch '# netguard$'}|Set-Content $h\n"
        "Write-Output NETGUARD_CLEARED\n"
    )
