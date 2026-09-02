"""apply_network_policy — provider-agnostic in-guest egress enforcement.

Enforcement is a property of the guest OS, not the provider. This drives the
guest only through the provider-agnostic ``SandboxHandle`` API. The mechanism is
identical on Linux and Windows (DNS override + a TLS-passthrough proxy on
127.0.0.1:443 + a default-deny egress firewall); only the setup script (bash vs
PowerShell) differs. See :mod:`.proxy` and :mod:`.setup`.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import setup

if TYPE_CHECKING:
    from ...base_interface import NetworkPolicy, SandboxHandle

logger = logging.getLogger(__name__)


def _proxy_b64() -> str:
    src = Path(__file__).with_name("proxy.py").read_text(encoding="utf-8")
    return base64.b64encode(src.encode("utf-8")).decode("ascii")


async def apply_network_policy(handle: "SandboxHandle", policy: "NetworkPolicy") -> None:
    """Install egress enforcement for ``policy``. ``open`` is a no-op;
    ``allowlist`` with an empty allow-list raises; any install failure raises so
    the caller deletes the VM rather than run unisolated."""
    if policy.mode == "open":
        return
    if policy.mode == "allowlist" and not policy.allow:
        raise RuntimeError(
            "network mode=allowlist but the allow-list is empty (no model host "
            "resolved and no task-declared hosts) — refusing to strand the agent"
        )
    hosts = list(policy.allow)
    b64 = _proxy_b64()
    if handle.os == "linux":
        script = setup.build_linux(policy.mode, hosts, b64)
        await handle.write_file("/tmp/netguard_setup.sh", script)
        res = await handle.run_command("sudo -n bash /tmp/netguard_setup.sh", timeout=150)
    elif handle.os == "windows":
        script = setup.build_windows(policy.mode, hosts, b64)
        await handle.run_command(r"cmd /c if not exist C:\netguard mkdir C:\netguard")
        await handle.write_file(r"C:\netguard\setup.ps1", script)
        res = await handle.run_command(
            r"powershell -NoProfile -ExecutionPolicy Bypass -File C:\netguard\setup.ps1", timeout=180)
    else:
        raise RuntimeError(f"network policy unsupported on os={handle.os!r}")

    out = res.stdout or ""
    if "NETGUARD_OK" not in out:
        raise RuntimeError(
            f"could not apply {handle.os} network policy (mode={policy.mode}, "
            f"allow={sorted(policy.allow)}): {res.stderr or out or 'no output'}"
        )
    logger.info(
        "netguard: applied policy mode=%s allow=%s on %s (%s)",
        policy.mode, sorted(policy.allow), handle.id, handle.os,
    )
