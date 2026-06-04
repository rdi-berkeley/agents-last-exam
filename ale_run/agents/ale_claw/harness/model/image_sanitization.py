"""Image sanitization (resize/transcode).

Mirrors OpenClaw's image-sanitization chokepoint so oversize / over-dimension
images are downsized and transcoded to JPEG instead of being hard-rejected.

Reference (target behavior):
  - openclaw/src/agents/tool-images.ts
      `sanitizeContentBlocksImages`, `resizeImageBase64IfNeeded`,
      `inferMimeTypeFromBase64`. Decision tree at lines 170-241.
  - openclaw/src/agents/image-sanitization.ts
      `DEFAULT_IMAGE_MAX_DIMENSION_PX = 1200`,
      `DEFAULT_IMAGE_MAX_BYTES = 5 * 1024 * 1024`.
  - openclaw/src/media/image-ops.ts
      `resizeToJpeg` (sharp-backed; we use Pillow),
      `buildImageResizeSideGrid`,
      `IMAGE_REDUCE_QUALITY_STEPS = [85, 75, 65, 55, 45, 35]`,
      `MAX_IMAGE_INPUT_PIXELS = 25_000_000`.

Fidelity-preserving by design:
  An image whose MIME is not in the provider allowlist (e.g. `image/bmp`,
  `image/tiff`, `image/avif`) but whose bytes/dimensions are within the
  limits is **passed through unchanged**, matching OpenClaw's early-return
  at tool-images.ts:170-183. This is a known upstream gap — the provider
  will 400 on such images. The fix (force-transcode any non-allowlisted
  MIME) is tracked separately and is potentially upstreamable
  to OpenClaw.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — match OpenClaw bit-for-bit
# ---------------------------------------------------------------------------

DEFAULT_MAX_DIMENSION_PX = 1200
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
MAX_INPUT_PIXELS = 25_000_000
QUALITY_STEPS: tuple[int, ...] = (85, 75, 65, 55, 45, 35)
SIDE_GRID_BASE: tuple[int, ...] = (1800, 1600, 1400, 1200, 1000, 800)


@dataclass(frozen=True)
class ImageLimits:
    """Sanitization limits. Defaults match OpenClaw."""

    max_dim_px: int = DEFAULT_MAX_DIMENSION_PX
    max_bytes: int = DEFAULT_MAX_BYTES
    max_input_pixels: int = MAX_INPUT_PIXELS


DEFAULT_LIMITS = ImageLimits()


# ---------------------------------------------------------------------------
# MIME sniffing — lifted from tools_fs.py / analyze_image.py
# ---------------------------------------------------------------------------


def sniff_mime_from_bytes(data: bytes) -> Optional[str]:
    """Return a MIME type sniffed from magic bytes, or ``None`` if unknown.

    Mirrors the subset of ``sniffMimeFromBase64`` needed for our supported
    image types plus PDF (for the non-image-sniff error path used by
    `tools_fs._read_image`).
    """
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 2 and data[:2] == b"BM":
        return "image/bmp"
    if len(data) >= 4 and data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if len(data) >= 5 and data[:5] == b"%PDF-":
        return "application/pdf"
    return None


# ---------------------------------------------------------------------------
# Side-grid construction — port of buildImageResizeSideGrid
# ---------------------------------------------------------------------------


def build_resize_side_grid(max_side: int, side_start: int) -> list[int]:
    """Mirror ``buildImageResizeSideGrid`` (image-ops.ts:17-22).

    Returns ``[side_start, 1800, 1600, 1400, 1200, 1000, 800]`` clamped to
    ``max_side``, deduplicated (preserving first-seen order), filtered to
    positive values, then sorted descending.
    """
    raw = [side_start, *SIDE_GRID_BASE]
    clamped = [min(max_side, v) for v in raw]
    seen: set[int] = set()
    unique: list[int] = []
    for v in clamped:
        if v > 0 and v not in seen:
            seen.add(v)
            unique.append(v)
    return sorted(unique, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.2f}MB"


def _format_mb_short(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):.0f}MB"


def _placeholder(label: str, reason: str) -> str:
    return f"[{label}] omitted image payload: {reason}"


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """JPEG cannot encode alpha or palette modes — convert before save."""
    if img.mode in ("RGB", "L"):
        return img
    if img.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        # paste alpha channel as mask
        alpha = img.split()[-1]
        background.paste(img.convert("RGB"), mask=alpha)
        return background
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Core sanitizer — port of resizeImageBase64IfNeeded
# ---------------------------------------------------------------------------


def _decode_image(data: bytes, label: str) -> Union[Image.Image, str]:
    """Decode bytes to a PIL image with EXIF orientation applied.

    Returns the loaded image, or a placeholder error string on failure.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        return _placeholder(label, "could not decode image")
    except Exception as exc:  # noqa: BLE001
        return _placeholder(label, f"decode failed: {exc}")

    # EXIF orientation — Python equivalent of sharp's `.rotate()` with no args.
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        # Bad EXIF — keep the raw image. Matches sharp's failOnError: false.
        pass
    return img


def _dimension_error(
    width: int,
    height: int,
    label: str,
    limits: ImageLimits,
) -> Optional[str]:
    """Placeholder error string if dimensions are invalid/over the pixel limit, else None."""
    if width <= 0 or height <= 0:
        return _placeholder(label, "invalid image dimensions")
    if width * height > limits.max_input_pixels:
        return _placeholder(
            label,
            f"image exceeds pixel limit ({width}x{height} > {limits.max_input_pixels} px)",
        )
    return None


def _resize_to_budget(
    img: Image.Image,
    data: bytes,
    width: int,
    height: int,
    label: str,
    limits: ImageLimits,
) -> Union[tuple[bytes, str], str]:
    """Search the side x quality grid for a JPEG under the byte budget.

    Returns ``(out_bytes, "image/jpeg")`` for the first candidate that fits,
    or a placeholder error string if none does.
    """
    side_start = min(limits.max_dim_px, max(width, height))
    sides = build_resize_side_grid(limits.max_dim_px, side_start)

    rgb = _ensure_rgb(img)

    smallest: Optional[bytes] = None
    for side in sides:
        for quality in QUALITY_STEPS:
            candidate = rgb.copy()
            # `thumbnail` is fit-inside + withoutEnlargement (matches OC).
            candidate.thumbnail((side, side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            candidate.save(
                buf,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            out = buf.getvalue()
            if smallest is None or len(out) < len(smallest):
                smallest = out
            if len(out) <= limits.max_bytes:
                logger.info(
                    "image resized %dx%d %s -> %s (side=%d quality=%d label=%s)",
                    width,
                    height,
                    _format_mb(len(data)),
                    _format_mb(len(out)),
                    side,
                    quality,
                    label,
                )
                return (out, "image/jpeg")

    got = smallest if smallest is not None else data
    return _placeholder(
        label,
        (
            f"Image could not be reduced below {_format_mb_short(limits.max_bytes)} "
            f"(got {_format_mb(len(got))})"
        ),
    )


def sanitize_raw_image_bytes(
    data: bytes,
    mime: str,
    *,
    label: str,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> Union[tuple[bytes, str], str]:
    """Apply OpenClaw's resize/passthrough policy to raw image bytes.

    Returns ``(out_bytes, out_mime)`` on success (passthrough or transcode)
    or a placeholder error string on failure. The string carries enough
    detail for the model to reason about what went wrong.
    """
    if not data:
        return _placeholder(label, "empty image payload")

    img = _decode_image(data, label)
    if isinstance(img, str):
        return img

    width, height = img.size
    dim_error = _dimension_error(width, height, label, limits)
    if dim_error is not None:
        return dim_error

    # Fidelity passthrough — matches resizeImageBase64IfNeeded:170-183 early-return.
    over_bytes = len(data) > limits.max_bytes
    over_dim = max(width, height) > limits.max_dim_px
    if not over_bytes and not over_dim:
        return (data, mime)

    return _resize_to_budget(img, data, width, height, label, limits)


# ---------------------------------------------------------------------------
# Block-level wrappers — port of sanitizeContentBlocksImages
# ---------------------------------------------------------------------------


def sanitize_image_block(
    block: dict,
    *,
    label: str,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> dict:
    """Sanitize a single ``{type: "image", data, mime_type}`` block.

    Returns the (possibly rewritten) image block on success, or a text
    block ``{type: "text", text: <placeholder>}`` on failure.
    """
    if block.get("type") != "image":
        return block
    raw_b64 = block.get("data")
    mime = block.get("mime_type") or block.get("mimeType")
    if not isinstance(raw_b64, str) or not isinstance(mime, str):
        return {"type": "text", "text": _placeholder(label, "malformed image block")}

    b64 = raw_b64.strip()
    if not b64:
        return {"type": "text", "text": _placeholder(label, "empty image payload")}

    try:
        data = base64.b64decode(b64, validate=False)
    except Exception:  # noqa: BLE001
        return {"type": "text", "text": _placeholder(label, "invalid base64")}

    result = sanitize_raw_image_bytes(data, mime, label=label, limits=limits)
    if isinstance(result, str):
        return {"type": "text", "text": result}

    out_bytes, out_mime = result
    if out_bytes is data and out_mime == mime:
        return block
    new_block = dict(block)
    new_block["data"] = base64.b64encode(out_bytes).decode("ascii")
    if "mime_type" in new_block:
        new_block["mime_type"] = out_mime
    if "mimeType" in new_block:
        new_block["mimeType"] = out_mime
    return new_block


def sanitize_tool_result_images(
    result: dict[str, Any],
    *,
    label: str,
    limits: ImageLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Walk a tool-result's ``content`` list, sanitize each image block.

    Mirrors ``sanitizeToolResultImages`` (tool-images.ts:350-362). Non-image
    blocks pass through unchanged. The returned dict is a shallow copy with
    a new ``content`` list when any block is rewritten; otherwise the
    original is returned.
    """
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result

    new_content: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            sanitized = sanitize_image_block(block, label=label, limits=limits)
            if sanitized is not block:
                changed = True
            new_content.append(sanitized)
        else:
            new_content.append(block)
    if not changed:
        return result
    out = dict(result)
    out["content"] = new_content
    return out


__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_DIMENSION_PX",
    "ImageLimits",
    "MAX_INPUT_PIXELS",
    "QUALITY_STEPS",
    "SIDE_GRID_BASE",
    "build_resize_side_grid",
    "sanitize_image_block",
    "sanitize_raw_image_bytes",
    "sanitize_tool_result_images",
    "sniff_mime_from_bytes",
]
