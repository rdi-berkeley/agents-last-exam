"""apply_network_policy — provider-agnostic egress enforcement over a sandbox.

The enforcement mechanism is a property of the guest OS, NOT of the provider:
any Linux sandbox (gcloud / qemu / aws / aliyun / static / docker) is governed
by the same nftables + aleguard transparent proxy, and any Windows sandbox by
the same Windows Firewall + CONNECT proxy. So this lives here, keyed on
``handle.os``, and talks to the guest only through the provider-agnostic
``SandboxHandle`` wire API (``write_file`` / ``run_command``). A provider opts in
by setting ``enforces_network_policy = True`` and calling this from ``acquire``.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import aleguard

if TYPE_CHECKING:
    from ...base_interface import NetworkPolicy, SandboxHandle

logger = logging.getLogger(__name__)


async def apply_network_policy(handle: "SandboxHandle", policy: "NetworkPolicy") -> None:
    """Install egress enforcement for ``policy`` on the sandbox behind ``handle``.

    ``open`` is a no-op. ``allowlist`` with an empty effective allow-list raises
    (refuse to strand the agent). Raises on any install failure so the caller
    deletes the VM and fails loudly rather than running unisolated.
    """
    if policy.mode == "open":
        return
    if policy.mode == "allowlist" and not policy.allow:
        raise RuntimeError(
            "network mode=allowlist but the allow-list is empty (no model host "
            "resolved and no task-declared hosts) — refusing to strand the agent"
        )
    if handle.os == "linux":
        await _apply_linux(handle, policy)
    elif handle.os == "windows":
        await _apply_windows(handle, policy)
    else:
        raise RuntimeError(f"network policy unsupported on os={handle.os!r}")
    logger.info(
        "netguard: applied policy mode=%s allow=%s on %s (%s)",
        policy.mode, sorted(policy.allow), handle.id, handle.os,
    )


async def _apply_linux(handle: "SandboxHandle", policy: "NetworkPolicy") -> None:
    proxy_src = (Path(__file__).with_name("aleguard.py")).read_text(encoding="utf-8")
    proxy_b64 = base64.b64encode(proxy_src.encode("utf-8")).decode("ascii")
    script = aleguard.build_setup_script(policy.mode, list(policy.allow), proxy_b64)

    # cua-server runs commands as a non-root user; the setup needs root
    # (nftables, useradd, /opt, /etc). Stage the script then run under sudo.
    script_path = "/tmp/aleguard_setup.sh"
    await handle.write_file(script_path, script)
    res = await handle.run_command(f"sudo -n bash {script_path}", timeout=120)
    out = res.stdout or ""
    if "ALEGUARD_OK" not in out:
        raise RuntimeError(
            f"could not apply linux network policy (mode={policy.mode}, "
            f"allow={sorted(policy.allow)}): {res.stderr or out or 'no output'}"
        )


async def _apply_windows(handle: "SandboxHandle", policy: "NetworkPolicy") -> None:
    from . import winguard  # lazy: only needed for Windows guests
    proxy_src = (Path(__file__).with_name("winguard.py")).read_text(encoding="utf-8")
    proxy_b64 = base64.b64encode(proxy_src.encode("utf-8")).decode("ascii")
    ps1 = winguard.build_setup_ps1(policy.mode, list(policy.allow), proxy_b64)

    # Stage the PowerShell setup + run it elevated. The cua-server on Windows
    # runs as an admin-capable user, so the firewall/proxy commands succeed.
    remote_ps1 = r"C:\aleguard\setup.ps1"
    await handle.run_command(r'cmd /c if not exist C:\aleguard mkdir C:\aleguard')
    await handle.write_file(remote_ps1, ps1)
    res = await handle.run_command(
        f'powershell -NoProfile -ExecutionPolicy Bypass -File {remote_ps1}',
        timeout=180,
    )
    out = res.stdout or ""
    if "ALEGUARD_OK" not in out:
        raise RuntimeError(
            f"could not apply windows network policy (mode={policy.mode}, "
            f"allow={sorted(policy.allow)}): {res.stderr or out or 'no output'}"
        )
