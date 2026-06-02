"""
Analyze Image Tool — VLM-based image analysis for the agent.

Reads image(s) from remote VM, local filesystem, HTTP URLs, or data URIs,
sends them to a vision model with a prompt, and returns text analysis.
No raw images are added to the agent's context.

Design reference: openclaw/src/agents/tools/image-tool.ts
  - Core pattern: read image → base64 → send to VLM → return text
  - Multi-image support, data URI decoding, URL fetching
  - maxBytesMb safety cap
  Adapted for CUA: dual-path (remote VM / local), litellm instead of pi-ai
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Optional, Union

import litellm

from agent.tools.base import BaseTool, register_tool

from .image_sanitization import ImageLimits, sanitize_raw_image_bytes

if TYPE_CHECKING:
    from computer.interface import BaseComputerInterface

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe the image."
DEFAULT_MAX_IMAGES = 20
DEFAULT_MAX_BYTES_MB = 10

# Extension → MIME type mapping
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".svg": "image/svg+xml",
}


def _mime_from_extension(path: str) -> str:
    """Guess MIME type from file extension, defaulting to image/png."""
    for ext, mime in _MIME_MAP.items():
        if path.lower().endswith(ext):
            return mime
    return "image/png"


def _is_remote_path(path: str) -> bool:
    """Detect if the path is a Windows/remote path based on format.

    Accepts either separator after the drive letter — ``C:\\foo`` and
    ``C:/foo`` both count as Windows paths, since the remote VM normalises
    slashes and agents frequently emit forward-slash form (easier to
    escape in JSON tool-args).

    Same heuristic as MilestoneTool._is_windows_path().
    """
    return bool(
        re.match(r"^[A-Za-z]:[\\/]", path)
        or path.startswith("\\\\")
        or "\\" in path
    )


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    """Decode a data:image/...;base64,... URI into (bytes, mime_type).

    Reference: openclaw/src/agents/tools/image-tool.helpers.ts decodeDataUrl
    """
    match = re.match(r"^data:([^;,]+);base64,([A-Za-z0-9+/=\r\n]+)$", uri.strip())
    if not match:
        raise ValueError("expected base64 data: URL")
    mime_type = match.group(1).strip().lower()
    if not mime_type.startswith("image/"):
        raise ValueError(f"unsupported data URL type: {mime_type}")
    b64_data = match.group(2).strip()
    image_bytes = base64.b64decode(b64_data)
    if len(image_bytes) == 0:
        raise ValueError("empty payload")
    return image_bytes, mime_type


@register_tool("analyze_image")
class AnalyzeImageTool(BaseTool):
    """Analyze image(s) with a vision model and return text analysis.

    Supports remote VM paths (Windows), local filesystem paths (Unix),
    HTTP(S) URLs, and data URIs. Returns text only — no raw images
    are added to the agent's context.
    """

    def __init__(
        self,
        interface: "BaseComputerInterface",
        model: str | None = None,
        thinking_params: Optional[dict[str, Any]] = None,
        cfg: Optional[dict] = None,
    ):
        self.interface = interface
        self.model = model or "anthropic/claude-sonnet-4-20250514"
        self.thinking_params = thinking_params or {}
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Analyze one or more images with a vision model and return a text description. "
            "Use 'image' for a single path/URL/data-URI, or 'images' for multiple (up to 20). "
            "Provide a 'prompt' describing what to analyze. "
            "Returns text only — no images are added to your context."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Single image: file path, HTTP(S) URL, or data URI.",
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple image paths/URLs/data-URIs (up to 20).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Question or instruction about the image(s). Default: 'Describe the image.'",
                },
                "maxBytesMb": {
                    "type": "number",
                    "description": "Max size per image in MB (default 10). Images exceeding this are rejected.",
                },
            },
            "required": [],
        }

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        """Execute image analysis (sync wrapper around async implementation)."""
        params_dict = self._verify_json_format_args(params)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._execute(params_dict)
                    )
                    result = future.result()
            else:
                result = asyncio.run(self._execute(params_dict))
            return result
        except Exception as e:
            logger.error(f"Error in analyze_image: {e}")
            return f"Error: image analysis failed — {e}"

    # ------------------------------------------------------------------
    # Input normalization (from OpenClaw image-tool.ts:325-345)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_inputs(params: dict) -> list[str]:
        """Merge image + images, strip @ prefix, dedup while preserving order."""
        candidates: list[str] = []
        if isinstance(params.get("image"), str):
            candidates.append(params["image"])
        if isinstance(params.get("images"), list):
            candidates.extend(v for v in params["images"] if isinstance(v, str))

        seen: set[str] = set()
        result: list[str] = []
        for raw in candidates:
            trimmed = raw.strip()
            normalized = trimmed[1:].strip() if trimmed.startswith("@") else trimmed
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(trimmed)
        return result

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    async def _load_image(
        self, raw_input: str, max_bytes: int | None
    ) -> tuple[str, str] | str:
        """Load a single image and return (base64, mime_type) or error string."""
        trimmed = raw_input.strip()
        image_ref = trimmed[1:].strip() if trimmed.startswith("@") else trimmed

        if not image_ref:
            return "Error: empty image reference."

        # --- Detect input type ---
        is_data_uri = image_ref.lower().startswith("data:")
        is_http_url = re.match(r"^https?://", image_ref, re.IGNORECASE) is not None
        is_windows_drive = re.match(r"^[A-Za-z]:[/\\]", image_ref) is not None
        has_scheme = re.match(r"^[a-z][a-z0-9+.-]*:", image_ref, re.IGNORECASE) is not None

        # Reject unsupported schemes (OpenClaw image-tool.ts:409-422)
        if has_scheme and not is_data_uri and not is_http_url and not is_windows_drive:
            return (
                f"Error: unsupported image reference: {raw_input}. "
                "Use a file path, data: URL, or http(s) URL."
            )

        try:
            if is_data_uri:
                image_bytes, mime_type = _decode_data_uri(image_ref)
            elif is_http_url:
                image_bytes, mime_type = await self._fetch_http(image_ref)
            elif _is_remote_path(image_ref):
                image_bytes = await self.interface.read_bytes(image_ref)
                mime_type = _mime_from_extension(image_ref)
            else:
                # Local filesystem path
                try:
                    with open(image_ref, "rb") as f:
                        image_bytes = f.read()
                except FileNotFoundError:
                    return f"Error: file not found: {image_ref}"
                except OSError as e:
                    return f"Error: could not read file: {image_ref} — {e}"
                mime_type = _mime_from_extension(image_ref)
        except ValueError as e:
            # From _decode_data_uri
            return f"Error: invalid data URL — {e}"
        except Exception as e:
            if _is_remote_path(image_ref):
                return f"Error: could not read file from remote VM: {image_ref} — {e}"
            return f"Error: failed to load image: {image_ref} — {e}"

        if not image_bytes:
            return f"Error: file is empty: {image_ref}"

        # Resize/transcode oversized images instead of hard-rejecting (US-OC-073).
        # Per-call max_bytes overrides the OpenClaw default (5 MB); other limits
        # (1200 px, 25 MP) come from ImageLimits defaults.
        limits = (
            ImageLimits(max_bytes=int(max_bytes))
            if max_bytes is not None and max_bytes > 0
            else ImageLimits()
        )
        sanitized = sanitize_raw_image_bytes(
            image_bytes, mime_type, label=f"analyze_image:{image_ref}", limits=limits
        )
        if isinstance(sanitized, str):
            return f"Error: {sanitized}"
        out_bytes, out_mime = sanitized

        b64 = base64.b64encode(out_bytes).decode("utf-8")
        return (b64, out_mime)

    @staticmethod
    async def _fetch_http(url: str) -> tuple[bytes, str]:
        """Fetch image bytes from an HTTP(S) URL."""
        import httpx

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            mime_type = content_type.split(";")[0].strip() if content_type else ""
            if not mime_type or not mime_type.startswith("image/"):
                # Fallback: guess from URL path
                mime_type = _mime_from_extension(url.split("?")[0])
            return response.content, mime_type

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def _execute(self, params: dict) -> str:
        """Async implementation: normalize inputs, load images, call VLM."""

        # Normalize inputs
        image_inputs = self._normalize_inputs(params)
        if not image_inputs:
            return "Error: at least one image is required (use 'image' or 'images' parameter)."

        # Max images cap
        max_images = DEFAULT_MAX_IMAGES
        if len(image_inputs) > max_images:
            return (
                f"Error: too many images: {len(image_inputs)} provided, "
                f"maximum is {max_images}."
            )

        # Parse maxBytesMb
        max_bytes_mb_raw = params.get("maxBytesMb")
        if isinstance(max_bytes_mb_raw, (int, float)) and max_bytes_mb_raw > 0:
            max_bytes = int(max_bytes_mb_raw * 1024 * 1024)
        else:
            max_bytes = int(DEFAULT_MAX_BYTES_MB * 1024 * 1024)

        # Load all images
        loaded: list[tuple[str, str]] = []
        for raw in image_inputs:
            result = await self._load_image(raw, max_bytes)
            if isinstance(result, str):
                # Error string — return immediately
                return result
            loaded.append(result)

        # Build prompt
        prompt = params.get("prompt", "") or DEFAULT_PROMPT

        # Build litellm messages with vision content
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64, mime in loaded:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        messages = [{"role": "user", "content": content}]

        # Call VLM
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                timeout=60,
                **self.thinking_params,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return f"Error: image analysis failed — {e}"
