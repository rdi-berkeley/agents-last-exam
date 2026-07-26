#!/usr/bin/env bash
# Probe the sandbox from the inside and record what happened.
#
# Written to keep going after each failure — a probe that aborted on its first refusal
# would report one guarantee and stay silent about the rest, which is the opposite of
# what an adversarial check is for.
set -uo pipefail

mkdir -p /ale/output
report=/ale/output/probe.json

reachable() {
    # Deliberately IP-based as well as DNS-based: a task could otherwise "pass" merely
    # because name resolution was unavailable, which is not the guarantee being claimed.
    timeout 5 getent hosts "$1" >/dev/null 2>&1 && return 0
    return 1
}

direct_dns=false;  reachable example.com && direct_dns=true
direct_tcp=false;  timeout 5 bash -c 'cat < /dev/null > /dev/tcp/1.1.1.1/443' 2>/dev/null && direct_tcp=true

# The gateway is the one permitted egress, and must still work.
gateway_ok=false
if [ -n "${ALE_GATEWAY_URL:-}" ]; then
    timeout 10 python3 - <<'PY' && gateway_ok=true
import os, sys, urllib.request
try:
    with urllib.request.urlopen(os.environ["ALE_GATEWAY_URL"].rstrip("/") + "/healthz", timeout=8) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
fi

# Credentials must never be here — only a per-episode bearer token.
#
# The needle is assembled at run time rather than written literally, because this script
# is itself uploaded into the sandbox: a literal would be found in this very file and
# the probe would report a leak it had caused. Searching plausible credential locations
# rather than all of / keeps it both fast and free of that class of self-match.
needle="sk-$(printf 'ant')-api"
leaked_key=false

if env | grep -q -- "$needle"; then
    leaked_key=true
fi
for spot in /root /home /etc /tmp /ale/work; do
    [ -d "$spot" ] || continue
    if timeout 10 grep -rqI -- "$needle" "$spot" 2>/dev/null; then
        leaked_key=true
    fi
done

# Scoring material must be absent while the agent phase has not happened yet.
saw_secrets=false
for p in /ale/verify /ale/oracle /ale/reference; do
    [ -e "$p" ] && saw_secrets=true
done

printf '{"direct_dns": %s, "direct_tcp": %s, "gateway_ok": %s, "leaked_key": %s, "saw_secrets": %s}\n' \
    "$direct_dns" "$direct_tcp" "$gateway_ok" "$leaked_key" "$saw_secrets" > "$report"
cat "$report"
