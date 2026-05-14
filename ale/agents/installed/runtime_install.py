"""Shared install helpers for installed-agent deployers.

These run against a ``cua_bench.DesktopSession`` and stage the toolchain a
generic CLI agent needs:

- ``ensure_node`` — verify (Linux) or download+extract (Windows) Node.js
- ``npm_install_global`` — ``npm i -g <package>`` (handles Windows PATH)
- ``upload_mcp_server`` — upload the vendored cua-mcp-server tree + npm install

All install paths come from :class:`InstallPaths` — never hardcoded here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cua_bench as cb

from ale.core.cmd_result import cmd_ok, cmd_rc, cmd_stderr, cmd_stdout

from .install_paths import InstallPaths

logger = logging.getLogger(__name__)


NODE_VERSION = "24.12.0"
"""Pinned Node version used by the Windows installer. Linux is image-baked."""


# Vendored MCP server lives next to this file at _assets/cua-mcp-server.
ASSETS_DIR = Path(__file__).resolve().parent / "_assets"
MCP_SERVER_SOURCE = ASSETS_DIR / "cua-mcp-server"


# =============================================================================
# Node
# =============================================================================

async def ensure_node(
    session: "cb.DesktopSession",
    install_paths: InstallPaths,
) -> None:
    """Ensure Node.js is callable. Linux must be pre-baked; Windows can download."""
    os_type = _os_type(session)
    node_exe = install_paths.node_exe(os_type)

    # First check: does our expected node binary exist + report version?
    if await session.exists(node_exe):
        cr = await session.run_command(_quote_cmd(os_type, f"{_q(os_type, node_exe)} --version"))
        if _ok(cr):
            logger.info("Node.js ok at %s: %s", node_exe, _stdout(cr).strip())
            return

    if os_type == "linux":
        # Linux: image must have Node baked. Fail loudly with actionable msg.
        raise RuntimeError(
            f"Node.js missing on Linux VM at {node_exe}. "
            f"Bake Node into the image; runtime apt install is intentionally not done."
        )

    # Windows: download + extract portable Node zip.
    await _install_node_windows(session, install_paths)


async def _install_node_windows(
    session: "cb.DesktopSession",
    install_paths: InstallPaths,
) -> None:
    """Download Node portable zip on Windows VM, extract, set npm prefix."""
    node_dir = rf"C:\Users\User\node-v{NODE_VERSION}-win-x64"
    node_zip = rf"C:\Users\User\node-v{NODE_VERSION}-win-x64.zip"
    npm_bin = r"C:\Users\User\AppData\Roaming\npm"
    node_url = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-win-x64.zip"

    logger.info("downloading Node.js %s on Windows VM", NODE_VERSION)
    download_ps = (
        f"$zip = '{node_zip}'; "
        f"if (Test-Path $zip) {{ Remove-Item -Force $zip }}; "
        f"$curl = Get-Command curl.exe -ErrorAction SilentlyContinue; "
        f"if ($curl) {{ "
        f"  & $curl.Source -L --retry 8 --retry-delay 5 --retry-all-errors "
        f"    --connect-timeout 30 --max-time 900 -o $zip '{node_url}' "
        f"}} else {{ "
        f"  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
        f"  Invoke-WebRequest -Uri '{node_url}' -OutFile $zip -UseBasicParsing "
        f"}}; "
        f"if (-not (Test-Path $zip)) {{ throw 'Node zip download failed' }}; "
        f"$s = (Get-Item $zip).Length; "
        f'if ($s -lt 20000000) {{ throw "Node zip too small ($s bytes)" }}'
    )
    cr = await session.run_command(
        f'powershell -NoProfile -Command "{download_ps}"', timeout=960,
    )
    if not _ok(cr):
        raise RuntimeError(f"Node download failed: {_stderr(cr)[:500]}")

    extract_ps = (
        f"if (Test-Path '{node_dir}') {{ Remove-Item -Recurse -Force '{node_dir}' }}; "
        f"Expand-Archive -Path '{node_zip}' -DestinationPath 'C:\\Users\\User' -Force"
    )
    cr = await session.run_command(
        f'powershell -NoProfile -Command "{extract_ps}"', timeout=300,
    )
    if not _ok(cr):
        raise RuntimeError(f"Node extract failed: {_stderr(cr)[:500]}")

    npm_cmd = rf"{node_dir}\npm.cmd"
    config_ps = (
        f"New-Item -ItemType Directory -Force -Path '{npm_bin}' | Out-Null; "
        f"& '{npm_cmd}' config set prefix '{npm_bin}'"
    )
    cr = await session.run_command(
        f'powershell -NoProfile -Command "{config_ps}"', timeout=120,
    )
    if not _ok(cr):
        raise RuntimeError(f"npm config failed: {_stderr(cr)[:500]}")

    logger.info("Node.js %s installed on Windows VM", NODE_VERSION)


# =============================================================================
# npm install -g
# =============================================================================

async def npm_install_global(
    session: "cb.DesktopSession",
    package: str,
    install_paths: InstallPaths,
    *,
    timeout: float = 1800.0,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run ``npm install -g <package>`` on the VM. Idempotent.

    ``extra_env`` is exported into the install shell (e.g. OpenClaw's
    ``OPENCLAW_EAGER_BUNDLED_PLUGIN_DEPS=1`` postinstall toggle).
    """
    os_type = _os_type(session)
    extra_env = extra_env or {}

    if os_type == "linux":
        env_prefix = "".join(f"export {k}={_sh_quote(v)}; " for k, v in extra_env.items())
        cmd = (
            f'{env_prefix}export PATH="{install_paths.agent_bin_dir("linux")}:$PATH"; '
            f"npm install -g --fetch-retries=5 --fetch-retry-maxtimeout=120000 "
            f"--fetch-timeout=300000 {package}"
        )
    else:
        node_dir = rf"C:\Users\User\node-v{NODE_VERSION}-win-x64"
        npm_bin = r"C:\Users\User\AppData\Roaming\npm"
        npm_cmd = rf"{node_dir}\npm.cmd"
        env_prefix = "".join(f'set "{k}={v}" && ' for k, v in extra_env.items())
        cmd = (
            f"{env_prefix}set PATH=%PATH%;{node_dir};{npm_bin} && "
            f'"{npm_cmd}" install -g {package}'
        )

    logger.info("npm install -g %s ...", package)
    cr = await session.run_command(cmd, timeout=timeout)
    if not _ok(cr):
        raise RuntimeError(
            f"npm install -g {package} failed (rc={_rc(cr)}): "
            f"stderr={_stderr(cr)[-1000:]}"
        )


# =============================================================================
# MCP server upload
# =============================================================================

async def upload_mcp_server(
    session: "cb.DesktopSession",
    install_paths: InstallPaths,
    *,
    timeout: float = 600.0,
) -> None:
    """Upload the vendored cua-mcp-server tree to the VM + ``npm install``."""
    if not MCP_SERVER_SOURCE.is_dir():
        raise RuntimeError(f"cua-mcp-server not vendored at {MCP_SERVER_SOURCE}")

    os_type = _os_type(session)
    remote_dir = install_paths.mcp_server_dir(os_type)
    logger.info("uploading cua-mcp-server → %s", remote_dir)
    await _upload_directory(session, MCP_SERVER_SOURCE, remote_dir)

    # Install npm deps inside the uploaded dir.
    if os_type == "linux":
        cmd = f"cd '{remote_dir}' && npm install --production"
    else:
        npm_cmd = rf"C:\Users\User\node-v{NODE_VERSION}-win-x64\npm.cmd"
        cmd = (
            f'powershell -NoProfile -Command "'
            f"Set-Location '{remote_dir}'; "
            f"& '{npm_cmd}' install --production"
            f'"'
        )

    cr = await session.run_command(cmd, timeout=timeout)
    if not _ok(cr):
        raise RuntimeError(
            f"cua-mcp-server npm install failed (rc={_rc(cr)}): "
            f"stderr={_stderr(cr)[-1000:]}"
        )


async def _upload_directory(
    session: "cb.DesktopSession",
    local_dir: Path,
    remote_dir: str,
) -> None:
    """Recursively upload ``local_dir`` to ``remote_dir`` on the VM.

    Uses ``session.write_bytes`` for content + ``run_command mkdir -p``
    for parent dirs. Slow on huge trees but adequate for our MCP bundle
    (~10 files).
    """
    sep = "\\" if "\\" in remote_dir or remote_dir[1:2] == ":" else "/"
    os_is_windows = sep == "\\"

    # Make root remote dir.
    if os_is_windows:
        await session.run_command(
            f"powershell -NoProfile -Command \""
            f"New-Item -ItemType Directory -Force -Path '{remote_dir}' | Out-Null"
            f"\"",
            timeout=60,
        )
    else:
        await session.run_command(f"mkdir -p {_sh_quote(remote_dir)}", timeout=60)

    # Walk local + upload files.
    for path in sorted(local_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(local_dir)
        rel_parts = rel.parts
        remote_path = remote_dir + sep + sep.join(rel_parts)
        # Ensure parent dir exists.
        if len(rel_parts) > 1:
            parent = remote_dir + sep + sep.join(rel_parts[:-1])
            if os_is_windows:
                await session.run_command(
                    f"powershell -NoProfile -Command \""
                    f"New-Item -ItemType Directory -Force -Path '{parent}' | Out-Null"
                    f"\"",
                    timeout=30,
                )
            else:
                await session.run_command(f"mkdir -p {_sh_quote(parent)}", timeout=30)
        # Upload content. Prefer write_bytes for binary safety.
        data = path.read_bytes()
        if hasattr(session, "write_bytes"):
            await session.write_bytes(remote_path, data)
        else:
            await session.write_file(remote_path, data.decode("utf-8", errors="replace"))


# =============================================================================
# Session helpers (mask differences across cua-bench versions / stubs)
# =============================================================================

def _os_type(session) -> str:
    """Pull the OS type from a session. Defaults to linux."""
    return getattr(session, "os_type", None) or "linux"


def _quote_cmd(os_type: str, cmd: str) -> str:
    """Wrap a bare cmd appropriately for the OS shell."""
    if os_type == "windows":
        return f'powershell -NoProfile -Command "{cmd}"'
    return cmd


def _q(os_type: str, path: str) -> str:
    """OS-appropriate quoting of a single path token in a shell cmd."""
    return f"'{path}'" if os_type == "linux" else path


def _sh_quote(value: str) -> str:
    """POSIX shell quote (single-quoted, escape embedded singles)."""
    if "'" not in value:
        return f"'{value}'"
    return "'" + value.replace("'", "'\\''") + "'"


# Result-shape compatibility (cua-bench dict vs stub dataclass) lives in
# ale.core.cmd_result; we keep these private names as thin aliases so the
# rest of this module reads simply.
_ok = cmd_ok
_rc = cmd_rc
_stdout = cmd_stdout
_stderr = cmd_stderr
