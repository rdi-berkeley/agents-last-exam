"""Shared bootstrap helpers for agent deployers.

Provides common system-dependency installers (npm, unzip, etc.) that
multiple deployers need when running on minimal container images.
"""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ale_run.base_interface import BaseExecutor, SandboxHandle

logger = logging.getLogger(__name__)

_NODE_MAJOR = "22"

_IS_WINDOWS = platform.system() == "Windows"

# Vendored cua MCP bridge source, shipped alongside ale_run into every
# substrate (docker bind-mount /ale_src, sandbox .ale-src ship). The
# ensure-step below copies it into the image's ``mcp_server_dir`` and runs
# ``npm install --production`` so the bridge actually exists for the MCP
# config every deployer writes. Sibling of this module:
# ``ale_run/agents/_assets/cua_mcp_server/``.
_CUA_BRIDGE_SRC = Path(__file__).resolve().parent / "_assets" / "cua_mcp_server"

# Sibling non-GUI bridge (run_command / fs / clipboard / pty primitives). Native
# agents (ale_claw) install this on the host and consume it as their substrate
# I/O backend; see ``ensure_vm_mcp_server`` / ``vm_bridge_env`` below.
_VM_BRIDGE_SRC = Path(__file__).resolve().parent / "_assets" / "vm_mcp_server"


async def _sh(cmd: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(
        subprocess.run,
        ["bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _find_windows_node_dir() -> str | None:
    """Locate a portable Node.js install dir on the Windows VM.

    The win10 image ships node unpacked but NOT on PATH (e.g.
    ``C:\\Users\\User\\node-v24.12.0-win-x64\\node.exe``). Glob the
    user profile for any ``node-v*-win-x64`` dir that contains both
    ``node.exe`` and ``npm.cmd``; fall back to common fixed locations.
    """
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, "node-v*-win-x64"),
        os.path.join(home, "node-v*-win-*"),
        r"C:\Program Files\nodejs",
        r"C:\nodejs",
    ]
    for pat in patterns:
        for cand in sorted(glob.glob(pat), reverse=True):
            node_exe = os.path.join(cand, "node.exe")
            npm_cmd = os.path.join(cand, "npm.cmd")
            if os.path.isfile(node_exe) and os.path.isfile(npm_cmd):
                return cand
    return None


async def _ensure_node_npm_windows() -> tuple[str, str]:
    """Windows path for :func:`ensure_node_npm`.

    Node ships unpacked but off PATH on the win VM. Find the dir, prepend
    it (and the npm global bin dir) to ``PATH`` so ``node`` / ``npm`` /
    globally-installed CLIs resolve, then return the binary paths.
    """
    node = shutil.which("node")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if node and npm:
        _configure_npm_prefix_windows()
        return node, shutil.which("npm") or shutil.which("npm.cmd") or npm

    node_dir = _find_windows_node_dir()
    if not node_dir:
        raise RuntimeError(
            "bootstrap: node not found on Windows. Looked on PATH and for "
            r"node-v*-win-x64 under %USERPROFILE% / Program Files."
        )
    if node_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = node_dir + os.pathsep + os.environ.get("PATH", "")
    node = os.path.join(node_dir, "node.exe")
    npm = os.path.join(node_dir, "npm.cmd")
    logger.info("bootstrap: using Windows node at %s", node)
    _configure_npm_prefix_windows()
    return node, npm


def _configure_npm_prefix_windows() -> None:
    """Ensure the npm global-install bin dir is on PATH (Windows).

    With the default prefix, ``npm install -g`` drops shims into
    ``%APPDATA%\\npm``; some node dirs also use the node dir itself.
    Prepend both so freshly-installed global CLIs resolve via
    ``shutil.which`` without a shell restart.
    """
    appdata = os.environ.get("APPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Roaming",
    )
    npm_bin = os.path.join(appdata, "npm")
    os.makedirs(npm_bin, exist_ok=True)
    if npm_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = npm_bin + os.pathsep + os.environ.get("PATH", "")


async def ensure_node_npm() -> tuple[str, str]:
    """Return (node_path, npm_path), installing Node 20 LTS if needed.

    Uses NodeSource apt repo to get a modern Node.js instead of the
    ancient v12 from Ubuntu 22.04's default repos.  Also configures
    npm's global prefix to ``~/.npm-global`` so no sudo is needed for
    ``npm install -g``.

    On Windows the VM ships node unpacked but off PATH; we locate it and
    fix PATH instead of installing.
    """
    if _IS_WINDOWS:
        return await _ensure_node_npm_windows()

    node = shutil.which("node")
    npm = shutil.which("npm")

    # Check if existing node is new enough (>= 16)
    if node and npm:
        try:
            ver = (await _sh(f"'{node}' --version", timeout=10)).stdout.strip()
            major = int(ver.lstrip("v").split(".")[0])
            if major >= 16:
                await _configure_npm_prefix()
                return node, npm
            logger.info("bootstrap: node %s too old (need >=16), upgrading ...", ver)
        except (ValueError, IndexError):
            pass

    logger.info("bootstrap: installing Node.js %s via NodeSource ...", _NODE_MAJOR)
    proc = await _sh(
        f"curl -fsSL https://deb.nodesource.com/setup_{_NODE_MAJOR}.x | sudo -E bash - "
        f"&& sudo apt-get install -y -qq nodejs 2>&1 | tail -10",
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bootstrap: NodeSource install failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[:500]}"
        )

    node = shutil.which("node") or "/usr/bin/node"
    npm = shutil.which("npm") or "/usr/bin/npm"
    if not os.path.isfile(node):
        raise RuntimeError("bootstrap: node still not found after install")
    logger.info("bootstrap: node installed at %s", node)

    await _configure_npm_prefix()
    return node, npm


async def _configure_npm_prefix() -> None:
    """Set npm global prefix to ~/.npm-global so installs don't need sudo."""
    home = os.path.expanduser("~")
    npm_global = f"{home}/.npm-global"
    npm_bin = f"{npm_global}/bin"
    os.makedirs(npm_global, exist_ok=True)

    await _sh(f"npm config set prefix '{npm_global}'", timeout=15)

    if npm_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{npm_bin}:{os.environ.get('PATH', '')}"


async def ensure_npm() -> str:
    """Return path to ``npm``, installing Node.js+npm if missing."""
    _, npm = await ensure_node_npm()
    return npm


def _bridge_installed(bridge_dir: str) -> bool:
    """Whether the MCP bridge at ``bridge_dir`` is present AND runnable.

    Surface-agnostic (used for both the cua GUI bridge and the vm primitives
    bridge). Gating on ``src/index.js`` alone is not enough: a bridge dir that
    has the entry script but no installed ``node_modules`` (e.g. a prebaked image
    where the source was copied in but ``npm install`` never ran, or a partially
    shipped tree) makes the node child die on its first import of
    ``@modelcontextprotocol/sdk`` with ``MODULE_NOT_FOUND``. We therefore require
    the SDK module the bridge imports first to exist alongside the entry script.
    This predicate is the fast-skip path: when both are present, the ensure-step
    is a no-op.
    """
    index = os.path.join(bridge_dir, "src", "index.js")
    sdk_dir = os.path.join(
        bridge_dir, "node_modules", "@modelcontextprotocol", "sdk",
    )
    return os.path.isfile(index) and os.path.isdir(sdk_dir)


async def _ensure_bridge_at(src_dir: Path, target_dir: str, *, what: str) -> str:
    """Copy ``src_dir`` → ``target_dir`` and ``npm install --production`` it.

    Shared implementation behind :func:`ensure_cua_mcp_server` (GUI bridge into
    the sandbox's ``mcp_server_dir``) and :func:`ensure_vm_mcp_server` (vm bridge
    into a host dir for native agents). Idempotent: a prebaked / already-installed
    tree is a no-op.

    Runs where the deployer runs — IN the sandbox for sandbox-resident agents, on
    the host for native agents — so it uses ``shutil`` / ``subprocess`` directly
    rather than ``run_command`` RPCs. ``what`` is a short label used only in log
    and error messages. Returns ``target_dir`` (the bridge root).
    """
    # Fast-path: prebaked / already-installed. Skip the copy + npm install.
    if _bridge_installed(target_dir):
        logger.info("ensure_%s_mcp_server: bridge already present at %s", what, target_dir)
        return target_dir

    if not src_dir.is_dir():
        raise RuntimeError(
            f"ensure_{what}_mcp_server: vendored bridge source missing at "
            f"{src_dir} — it must ship alongside ale_run (docker /ale_src mount, "
            "sandbox .ale-src ship, or the in-tree _assets dir on the host)."
        )

    logger.info("ensure_%s_mcp_server: installing bridge %s → %s", what, src_dir, target_dir)

    # 1. Copy the bridge source (package.json + package-lock.json + src/) into
    #    target_dir. Never copy node_modules — it is rebuilt by npm install
    #    (binaries must match the substrate's arch).
    os.makedirs(target_dir, exist_ok=True)
    for entry in src_dir.iterdir():
        if entry.name == "node_modules":
            continue
        dest = os.path.join(target_dir, entry.name)
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)

    # 2. npm install --production inside the bridge dir.
    npm = await ensure_npm()
    proc = await asyncio.to_thread(
        subprocess.run,
        [npm, "install", "--production"],
        cwd=target_dir,
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ensure_{what}_mcp_server: npm install --production failed in "
            f"{target_dir} (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[-800:]}"
        )

    # 3. Verify the install actually landed the SDK (npm can exit 0 but produce
    #    an unusable tree on a flaky registry); a broken bridge wedges MCP stdio
    #    handshakes downstream, so fail loud here instead.
    if not _bridge_installed(target_dir):
        raise RuntimeError(
            f"ensure_{what}_mcp_server: npm install completed but bridge still "
            f"not runnable at {target_dir} (missing src/index.js or "
            "node_modules/@modelcontextprotocol/sdk)"
        )

    logger.info("ensure_%s_mcp_server: bridge installed at %s", what, target_dir)
    return target_dir


async def ensure_cua_mcp_server(sandbox: "SandboxHandle") -> str:
    """Ensure the cua (GUI) MCP bridge is installed at ``sandbox.mcp_server_dir``.

    Idempotent (prebaked-image fast-path). Cross-OS: ``mcp_server_dir`` is
    whatever the image declares (``/home/.../cua_mcp_server`` on linux,
    ``C:\\Users\\User\\cua_mcp_server`` on windows). Prebaking the bridge into the
    image is purely a speed optimization; this dynamic install is the correctness
    guarantee on a thin image. Returns the resolved ``mcp_server_dir``.
    """
    return await _ensure_bridge_at(_CUA_BRIDGE_SRC, sandbox.mcp_server_dir, what="cua")


async def ensure_cua_mcp_server_at(target_dir: str) -> str:
    """Host-install variant of :func:`ensure_cua_mcp_server` (explicit dir).

    Native agents (ale_claw, ``local`` executor) run on the host and route GUI
    through the cua bridge in Phase 2; they pass an explicit host dir rather than
    a sandbox field. Idempotent; returns ``target_dir``.
    """
    return await _ensure_bridge_at(_CUA_BRIDGE_SRC, target_dir, what="cua")


async def ensure_vm_mcp_server(target_dir: str) -> str:
    """Ensure the vm (non-GUI) MCP bridge is installed at ``target_dir``.

    Counterpart to :func:`ensure_cua_mcp_server` for the vm primitives bridge.
    Native agents (ale_claw, ``local`` executor) run on the host, so they pass an
    explicit **host** dir (e.g. ``<work_dir>/mcp/vm``) rather than a sandbox
    field. Idempotent; returns ``target_dir``.
    """
    return await _ensure_bridge_at(_VM_BRIDGE_SRC, target_dir, what="vm")


def cua_bridge_env(executor: "BaseExecutor") -> dict[str, str]:
    """Env vars an MCP-capable deployer must pass to the cua MCP bridge.

    The bridge (``cua_mcp_server/src/index.js``) reads ``CUA_SERVER_URL`` and
    otherwise falls back to a built-in default that does not match every image
    (e.g. ale-kasm runs cua-server on 8000, the bridge's default is 5000). The
    executor knows the URL reachable from where the bridge runs
    (``SandboxExecutor`` → loopback + image port; Local/Docker → host endpoint),
    so deployers splat this into their ``mcpServers.cua`` entry's ``env``."""
    return {"CUA_SERVER_URL": executor.cua_bridge_url()}


def vm_bridge_env(executor: "BaseExecutor") -> dict[str, str]:
    """Env vars for the vm MCP bridge — identical contract to :func:`cua_bridge_env`.

    Both bridges talk to the same cua-server and read ``CUA_SERVER_URL``; the URL
    is whatever is reachable from where the bridge runs (host endpoint for the
    ``local`` executor that native agents use)."""
    return {"CUA_SERVER_URL": executor.cua_bridge_url()}


async def ensure_unzip() -> str:
    """Return path to ``unzip``, installing via apt if missing."""
    uz = shutil.which("unzip")
    if uz:
        return uz

    logger.info("bootstrap: unzip not found, installing via apt ...")
    proc = await _sh(
        "sudo apt-get update -qq "
        "&& sudo apt-get install -y -qq unzip 2>&1 | tail -3",
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bootstrap: apt-get install unzip failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[:500]}"
        )
    uz = shutil.which("unzip") or "/usr/bin/unzip"
    logger.info("bootstrap: unzip installed at %s", uz)
    return uz
