"""Unit tests for openclaw.image_sanitization.

Covers the resize / passthrough / placeholder decision tree, EXIF orientation,
the pixel-count guardrail, and the side-grid construction. Mirrors the
behavior described in openclaw/src/agents/tool-images.ts and
openclaw/src/media/image-ops.ts.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from ale_run.agents.ale_claw.harness.model.image_sanitization import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_DIMENSION_PX,
    ImageLimits,
    build_resize_side_grid,
    sanitize_image_block,
    sanitize_raw_image_bytes,
    sanitize_tool_result_images,
    sniff_mime_from_bytes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _png(width: int, height: int, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(width: int, height: int, color=(255, 0, 0), quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(
        buf, format="JPEG", quality=quality
    )
    return buf.getvalue()


def _bmp(width: int, height: int, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="BMP")
    return buf.getvalue()


def _png_with_exif_orientation(orientation: int) -> bytes:
    """Build a small portrait PNG, write JPEG with EXIF orientation tag.

    Returns JPEG bytes (PNG doesn't carry EXIF universally; JPEG does).
    """
    img = Image.new("RGB", (40, 80), (90, 200, 30))  # portrait 40w x 80h
    buf = io.BytesIO()
    # Pillow doesn't expose simple EXIF write; fabricate raw EXIF bytes.
    exif = img.getexif()
    exif[0x0112] = orientation  # Orientation tag
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# build_resize_side_grid
# ---------------------------------------------------------------------------


class TestSideGrid:
    def test_clamp_dedupe_sort_desc(self):
        # max_dim=1200 with start=2400 → start clamped to 1200; dedupe removes
        # the 1200 from SIDE_GRID_BASE; result sorted desc.
        assert build_resize_side_grid(1200, 2400) == [1200, 1000, 800]

    def test_start_below_grid_base(self):
        # max_dim=1200, start=900 → [900, 1200, 1000, 800] then sorted desc.
        assert build_resize_side_grid(1200, 900) == [1200, 1000, 900, 800]

    def test_max_dim_clamps_all(self):
        # max_dim=500 collapses everything to 500 (and 500 only once).
        assert build_resize_side_grid(500, 500) == [500]

    def test_start_zero_filtered(self):
        # Start of 0 is filtered as non-positive.
        result = build_resize_side_grid(1200, 0)
        assert 0 not in result
        assert max(result) <= 1200

    def test_oc_default_full_grid(self):
        # OpenClaw default: max_dim=1200, start=2400 (e.g. 2400x600 input).
        # Result should equal SIDE_GRID_BASE clamped + start, deduped.
        result = build_resize_side_grid(1200, 2400)
        assert all(v <= 1200 for v in result)
        assert result == sorted(set(result), reverse=True)


# ---------------------------------------------------------------------------
# sniff_mime_from_bytes
# ---------------------------------------------------------------------------


class TestSniff:
    def test_png(self):
        assert sniff_mime_from_bytes(_png(4, 4)) == "image/png"

    def test_jpeg(self):
        assert sniff_mime_from_bytes(_jpeg(4, 4)) == "image/jpeg"

    def test_bmp(self):
        assert sniff_mime_from_bytes(_bmp(4, 4)) == "image/bmp"

    def test_pdf(self):
        assert sniff_mime_from_bytes(b"%PDF-1.4\n...") == "application/pdf"

    def test_unknown(self):
        assert sniff_mime_from_bytes(b"not a real image") is None

    def test_empty(self):
        assert sniff_mime_from_bytes(b"") is None


# ---------------------------------------------------------------------------
# sanitize_raw_image_bytes — passthrough
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_small_jpeg_byte_identical(self):
        data = _jpeg(200, 200)
        out = sanitize_raw_image_bytes(data, "image/jpeg", label="t")
        assert isinstance(out, tuple)
        out_bytes, out_mime = out
        assert out_bytes is data  # identity preserved
        assert out_mime == "image/jpeg"

    def test_small_bmp_passthrough_fidelity(self):
        # FIDELITY check: OpenClaw passes small non-allowlisted MIMEs through
        # unchanged. deliberately preserves this behavior; the fix
        # is tracked .
        data = _bmp(100, 100)
        out = sanitize_raw_image_bytes(data, "image/bmp", label="t")
        assert isinstance(out, tuple)
        out_bytes, out_mime = out
        assert out_bytes is data
        assert out_mime == "image/bmp"  # mime stays as-is

    def test_small_png_passthrough(self):
        data = _png(50, 50)
        out = sanitize_raw_image_bytes(data, "image/png", label="t")
        assert isinstance(out, tuple)
        out_bytes, out_mime = out
        assert out_bytes is data
        assert out_mime == "image/png"


# ---------------------------------------------------------------------------
# sanitize_raw_image_bytes — resize
# ---------------------------------------------------------------------------


class TestResize:
    def test_oversize_dim_png_resized_to_jpeg(self):
        # 2400x1800 PNG > 1200 max-side limit → resized + transcoded.
        data = _png(2400, 1800, color=(80, 160, 240))
        out = sanitize_raw_image_bytes(data, "image/png", label="t")
        assert isinstance(out, tuple)
        out_bytes, out_mime = out
        assert out_mime == "image/jpeg"
        # Verify the actual decoded dimensions are within limits.
        with Image.open(io.BytesIO(out_bytes)) as im:
            assert max(im.size) <= DEFAULT_MAX_DIMENSION_PX
        assert len(out_bytes) <= DEFAULT_MAX_BYTES

    def test_oversize_bytes_jpeg_resized(self):
        # 1500x1500 source (max-side > 1200 default) → grid spans
        # [1200, 1000, 800]. Tight max_bytes forces stepping down to a
        # smaller side. Solid color compresses to ~4-9KB across the grid;
        # 5000 byte budget falls between the largest and smallest steps.
        data = _jpeg(1500, 1500, color=(200, 50, 50))
        limits = ImageLimits(max_bytes=5000)
        out = sanitize_raw_image_bytes(data, "image/jpeg", label="t", limits=limits)
        assert isinstance(out, tuple), out
        out_bytes, _out_mime = out
        assert len(out_bytes) <= 5000

    def test_large_png_under_byte_limit_still_resized_for_dim(self):
        # 4000x4000 PNG (= 16 MP — under the 25 MP guardrail) with flat
        # color may compress small as PNG, but max-side > 1200 → resize
        # path triggers on dimension regardless of byte size.
        data = _png(4000, 4000, color=(200, 200, 200))
        out = sanitize_raw_image_bytes(data, "image/png", label="t")
        assert isinstance(out, tuple), out
        out_bytes, out_mime = out
        assert out_mime == "image/jpeg"
        with Image.open(io.BytesIO(out_bytes)) as im:
            assert max(im.size) <= DEFAULT_MAX_DIMENSION_PX


# ---------------------------------------------------------------------------
# Failure / placeholder paths
# ---------------------------------------------------------------------------


class TestFailure:
    def test_empty_bytes(self):
        out = sanitize_raw_image_bytes(b"", "image/png", label="r:foo")
        assert isinstance(out, str)
        assert "[r:foo]" in out
        assert "empty" in out

    def test_garbage_bytes(self):
        out = sanitize_raw_image_bytes(
            b"this is not an image at all", "image/png", label="r:foo"
        )
        assert isinstance(out, str)
        assert "could not decode" in out or "decode failed" in out

    def test_pixel_guardrail(self):
        # 26 MP > 25 MP guardrail → placeholder, no resize attempt.
        # Pillow's own MAX_IMAGE_PIXELS may also raise; we expect the
        # sanitizer's pre-check to surface a clear placeholder regardless.
        # 5200x5001 = 26_005_200 pixels.
        prev_max = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 50_000_000  # let Pillow itself decode
        try:
            data = _png(5200, 5001, color=(10, 10, 10))
            out = sanitize_raw_image_bytes(data, "image/png", label="r:foo")
            assert isinstance(out, str)
            assert "pixel limit" in out
        finally:
            Image.MAX_IMAGE_PIXELS = prev_max

    def test_unreachable_byte_limit(self):
        # 50-byte ceiling is below any JPEG header overhead — full grid
        # exhausts and the placeholder mentions reduction failure.
        data = _jpeg(800, 600, color=(20, 30, 40))
        limits = ImageLimits(max_bytes=50)
        out = sanitize_raw_image_bytes(data, "image/jpeg", label="r:foo", limits=limits)
        assert isinstance(out, str)
        assert "could not be reduced" in out


# ---------------------------------------------------------------------------
# EXIF orientation
# ---------------------------------------------------------------------------


class TestExif:
    def test_orientation_6_swaps_dimensions_after_resize(self):
        # Source is 40w x 80h portrait JPEG with orientation=6 (rotate 90 CW).
        # After EXIF transpose, displayed image is 80w x 40h. Force resize so
        # the output reflects post-transpose dimensions.
        data = _png_with_exif_orientation(6)
        # Force resize via tight max_dim so the output is re-encoded.
        limits = ImageLimits(max_dim_px=20)
        out = sanitize_raw_image_bytes(data, "image/jpeg", label="t", limits=limits)
        assert isinstance(out, tuple)
        out_bytes, _ = out
        with Image.open(io.BytesIO(out_bytes)) as im:
            # After orientation=6 transpose, raw 40x80 portrait becomes 80x40
            # landscape; resized into a 20-side box preserves aspect → 20x10.
            assert im.size[0] >= im.size[1]


# ---------------------------------------------------------------------------
# Block-level wrappers
# ---------------------------------------------------------------------------


class TestBlockWrappers:
    def test_sanitize_image_block_passthrough(self):
        data = _jpeg(100, 100)
        block = {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": "image/jpeg",
        }
        out = sanitize_image_block(block, label="t")
        assert out is block  # identity preserved when nothing changed

    def test_sanitize_image_block_resize(self):
        data = _png(2400, 1800)
        block = {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": "image/png",
        }
        out = sanitize_image_block(block, label="t")
        assert out is not block
        assert out["type"] == "image"
        assert out["mime_type"] == "image/jpeg"
        new_bytes = base64.b64decode(out["data"])
        with Image.open(io.BytesIO(new_bytes)) as im:
            assert max(im.size) <= DEFAULT_MAX_DIMENSION_PX

    def test_sanitize_image_block_invalid_b64(self):
        block = {"type": "image", "data": "!!!not valid!!!", "mime_type": "image/png"}
        out = sanitize_image_block(block, label="r:foo")
        # base64.b64decode is permissive — empty string for garbage is the
        # likely path and produces an empty-payload placeholder.
        assert out["type"] == "text"
        assert out["text"].startswith("[r:foo]")

    def test_sanitize_image_block_empty_data(self):
        block = {"type": "image", "data": "", "mime_type": "image/png"}
        out = sanitize_image_block(block, label="r:foo")
        assert out["type"] == "text"
        assert "empty" in out["text"]

    def test_sanitize_image_block_camelcase_mime(self):
        # OC TS uses mimeType (camelCase); we accept both.
        data = _jpeg(100, 100)
        block = {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mimeType": "image/jpeg",
        }
        out = sanitize_image_block(block, label="t")
        assert out is block

    def test_sanitize_tool_result_images_walks_content(self):
        small = _jpeg(100, 100)
        big = _png(2400, 1800)
        result = {
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "image",
                    "data": base64.b64encode(small).decode("ascii"),
                    "mime_type": "image/jpeg",
                },
                {
                    "type": "image",
                    "data": base64.b64encode(big).decode("ascii"),
                    "mime_type": "image/png",
                },
            ]
        }
        out = sanitize_tool_result_images(result, label="t")
        assert out is not result  # changed (big image rewritten)
        assert out["content"][0] == {"type": "text", "text": "hello"}
        assert out["content"][1]["mime_type"] == "image/jpeg"  # passthrough
        assert out["content"][2]["mime_type"] == "image/jpeg"  # transcoded

    def test_sanitize_tool_result_images_no_change_returns_same(self):
        small = _jpeg(50, 50)
        result = {
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "image",
                    "data": base64.b64encode(small).decode("ascii"),
                    "mime_type": "image/jpeg",
                },
            ]
        }
        out = sanitize_tool_result_images(result, label="t")
        assert out is result

    def test_sanitize_tool_result_images_no_content(self):
        result = {"other": "field"}
        out = sanitize_tool_result_images(result, label="t")
        assert out is result


# ---------------------------------------------------------------------------
# OC parity — the side-grid test the PRD calls out by example
# ---------------------------------------------------------------------------


def test_oc_buildImageResizeSideGrid_parity():
    """Acceptance criterion sanity check: max_dim=1200, side_start=2400."""
    assert build_resize_side_grid(1200, 2400) == [1200, 1000, 800]
