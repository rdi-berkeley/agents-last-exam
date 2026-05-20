"""
Milestone Tool for saving screenshots on the remote computer.
"""

import base64
import logging
from typing import TYPE_CHECKING, Optional, Union

from agent.tools.base import BaseTool, register_tool

if TYPE_CHECKING:
    from computer.interface import BaseComputerInterface

logger = logging.getLogger(__name__)


@register_tool("save_milestone_screenshot")
class MilestoneTool(BaseTool):
    """
    Tool for saving milestone screenshots on the remote computer.
    """

    def __init__(self, interface: "BaseComputerInterface", cfg: Optional[dict] = None):
        """
        Initialize the MilestoneTool.

        Args:
            interface: A BaseComputerInterface instance
            cfg: Optional configuration dictionary
        """
        self.interface = interface
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return "Save the current screen as a milestone screenshot on the remote computer. Use this when you have completed a significant step or goal and want to save evidence of your progress."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Full path on the remote computer where the screenshot should be saved. Example: 'C:/Users/User/Desktop/milestones/step1.png'",
                },
                "description": {
                    "type": "string",
                    "description": "A brief description of the milestone achieved.",
                },
            },
            "required": ["path"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        """
        Execute the milestone screenshot save.

        Args:
            params: Action parameters (JSON string or dict)
            **kwargs: Additional keyword arguments

        Returns:
            Result of the action execution
        """
        import asyncio
        import concurrent.futures

        # Verify and parse parameters
        params_dict = self._verify_json_format_args(params)
        path = params_dict.get("path")
        description = params_dict.get("description", "")

        if not path:
            return {"success": False, "error": "path parameter is required"}

        # Execute action synchronously by running async method in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # Create a task and wait for it
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._execute_save(path, description)
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(self._execute_save(path, description))
            return result
        except Exception as e:
            logger.error(f"Error saving milestone screenshot: {e}")
            return {"success": False, "error": str(e)}

    def _is_windows_path(self, path: str) -> bool:
        """Detect if the path is a Windows path based on format.

        Accepts either separator after the drive letter — ``C:\\foo`` and
        ``C:/foo`` both count as Windows paths. Agents commonly emit the
        forward-slash form because it avoids double-escaping in JSON
        tool-args; the remote VM normalises slashes internally.
        """
        import re
        # Windows paths: C:\, C:/, D:\, \\server\, or contain backslashes
        return bool(re.match(r'^[A-Za-z]:[\\/]', path) or
                    path.startswith('\\\\') or
                    '\\' in path)

    async def _execute_save(self, path: str, description: str) -> dict:
        """Execute the screenshot save asynchronously."""
        try:
            # 1. Take screenshot
            screenshot_bytes = await self.interface.screenshot()

            # 2. Prepare remote directory
            # Detect remote OS from path format (not local os.name)
            is_windows = self._is_windows_path(path)
            
            if is_windows:
                # Use ntpath for Windows path manipulation
                import ntpath
                dir_path = ntpath.dirname(path)
            else:
                import posixpath
                dir_path = posixpath.dirname(path)
            
            if dir_path and dir_path != ".":
                # Create directory if it doesn't exist
                try:
                    # Try to create directory using run_command
                    # Use platform-agnostic approach based on path format
                    if is_windows:
                        # Use PowerShell for reliable directory creation on Windows
                        await self.interface.run_command(
                            f'powershell -Command "New-Item -ItemType Directory -Force -Path \'{dir_path}\' | Out-Null"'
                        )
                    else:  # Unix-like
                        await self.interface.run_command(f'mkdir -p "{dir_path}"')
                except Exception as dir_error:
                    logger.error(f"Error creating directory: {dir_error}")
                    return {"success": False, "error": f"Failed to create directory: {str(dir_error)}"}

            # 3. Save file using write_bytes (more reliable for binary data)
            try:
                await self.interface.write_bytes(path, screenshot_bytes)
                msg = f"✅ Milestone screenshot saved to: {path}"
                if description:
                    msg += f" (Milestone: {description})"
                return {"success": True, "message": msg}
            except Exception as save_error:
                logger.error(f"Error saving screenshot: {save_error}")
                return {"success": False, "error": f"Failed to save screenshot: {str(save_error)}"}

        except Exception as e:
            logger.error(f"Error in _execute_save: {e}")
            return {"success": False, "error": str(e)}
