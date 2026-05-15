"""InstallPaths: single source of truth for in-VM paths.

Deployers never hardcode ``/usr/local/bin/node`` or ``C:\\Program Files\\...``;
they ask ``self._cfg.install_paths.node_exe(session.os)``. To change a path
for a custom image, replace the :class:`InstallPaths` instance in the agent
config — no code changes needed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstallPaths:
    """Per-OS install layout for in-VM agent toolchains.

    Defaults match the baked agenthle Ubuntu / Windows images. Override
    fields when targeting a different image.
    """

    # ---- Linux ----
    linux_node_exe: str = "/usr/local/bin/node"
    linux_user_home: str = "/home/user"
    linux_mcp_server_dir: str = "/home/user/cua_mcp_server"
    """Aligned with agenthle ``LINUX_MCP_SERVER_DIR`` for image cross-compat."""
    linux_agent_bin_dir: str = "/usr/local/bin"
    linux_work_dir_root: str = "/home/user/.ale"

    # ---- Windows ----
    windows_node_exe: str = r"C:\Users\User\node-v24.12.0-win-x64\node.exe"
    """Where ``runtime_install._install_node_windows`` puts portable Node."""
    windows_user_home: str = r"C:\Users\User"
    windows_mcp_server_dir: str = r"C:\Users\User\cua_mcp_server"
    """Aligned with agenthle ``REMOTE_MCP_SERVER_DIR`` for image cross-compat."""
    windows_agent_bin_dir: str = r"C:\Users\User\AppData\Roaming\npm"
    """``npm i -g`` puts shims here when prefix is set to this dir."""
    windows_work_dir_root: str = r"C:\Users\User\.ale"

    # ---- selectors ----
    def node_exe(self, os: str) -> str:
        return self.windows_node_exe if os == "windows" else self.linux_node_exe

    def user_home(self, os: str) -> str:
        return self.windows_user_home if os == "windows" else self.linux_user_home

    def mcp_server_dir(self, os: str) -> str:
        return (
            self.windows_mcp_server_dir if os == "windows" else self.linux_mcp_server_dir
        )

    def agent_bin_dir(self, os: str) -> str:
        return (
            self.windows_agent_bin_dir if os == "windows" else self.linux_agent_bin_dir
        )

    def work_dir_root(self, os: str) -> str:
        return (
            self.windows_work_dir_root if os == "windows" else self.linux_work_dir_root
        )

    def work_dir(self, os: str, agent_name: str) -> str:
        """Per-agent working directory under ``work_dir_root``."""
        sep = "\\" if os == "windows" else "/"
        return f"{self.work_dir_root(os)}{sep}{agent_name}"

    def cli_path(self, os: str, cli_name: str) -> str:
        """Absolute path to a CLI binary baked under ``agent_bin_dir``."""
        sep = "\\" if os == "windows" else "/"
        return f"{self.agent_bin_dir(os)}{sep}{cli_name}"
