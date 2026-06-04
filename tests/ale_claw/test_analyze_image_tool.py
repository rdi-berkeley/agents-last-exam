"""Tests for AnalyzeImageTool.

Covers:
  - Tool registration and schema validation
  - Input normalization (dedup, @ stripping, max cap)
  - Path type detection (remote vs local)
  - Image loading: remote VM, local filesystem, data URI, HTTP URL
  - Size enforcement (maxBytesMb)
  - VLM call forwarding and error handling
"""

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as _PILImage

from ale_run.agents.ale_claw.harness.tools.analyze_image import (
    AnalyzeImageTool,
    _decode_data_uri,
    _is_remote_path,
    _mime_from_extension,
)
from agent.tools.base import TOOL_REGISTRY


def _real_png(width: int = 4, height: int = 4, color=(255, 0, 0)) -> bytes:
    """Real PNG bytes for tests — sanitizer decodes via Pillow,
    so placeholder bytes like ``b"fakepng"`` no longer survive the loader."""
    buf = io.BytesIO()
    _PILImage.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


_REAL_PNG_BYTES = _real_png()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_tool(
    model: str = "test-model",
    *,
    thinking_params: dict | None = None,
) -> AnalyzeImageTool:
    """Create an AnalyzeImageTool with a mock interface."""
    mock_interface = MagicMock()
    mock_interface.read_bytes = AsyncMock(return_value=_REAL_PNG_BYTES)
    return AnalyzeImageTool(
        mock_interface,
        model=model,
        thinking_params=thinking_params,
    )


def _mock_vlm_response(text: str = "Floor 2 is shown"):
    """Create a mock litellm response."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=text))]
    return mock_resp


def _make_data_uri(data: bytes | None = None, mime: str = "image/png") -> str:
    """Build a data URI from bytes."""
    if data is None:
        data = _REAL_PNG_BYTES
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Registration and schema
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_in_tool_registry(self):
        assert "analyze_image" in TOOL_REGISTRY
        assert TOOL_REGISTRY["analyze_image"] is AnalyzeImageTool

    def test_name_attribute(self):
        assert AnalyzeImageTool.name == "analyze_image"

    def test_schema_has_expected_properties(self):
        tool = _make_tool()
        props = tool.parameters["properties"]
        assert "image" in props
        assert "images" in props
        assert "prompt" in props
        assert "maxBytesMb" in props

    def test_no_required_fields(self):
        tool = _make_tool()
        assert tool.parameters.get("required", []) == []


# ---------------------------------------------------------------------------
# Path detection
# ---------------------------------------------------------------------------


class TestPathDetection:
    def test_windows_backslash(self):
        assert _is_remote_path(r"C:\Users\User\test.png") is True

    def test_windows_drive_letter(self):
        assert _is_remote_path("D:\\folder\\image.jpg") is True

    def test_unc_path(self):
        assert _is_remote_path("\\\\server\\share\\file.png") is True

    def test_unix_path_not_remote(self):
        assert _is_remote_path("/Users/local/file.png") is False

    def test_relative_path_not_remote(self):
        assert _is_remote_path("./relative/file.png") is False


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------


class TestMimeDetection:
    def test_png(self):
        assert _mime_from_extension("test.png") == "image/png"

    def test_jpg(self):
        assert _mime_from_extension("test.jpg") == "image/jpeg"

    def test_jpeg(self):
        assert _mime_from_extension("test.jpeg") == "image/jpeg"

    def test_unknown_defaults_to_png(self):
        assert _mime_from_extension("test.xyz") == "image/png"

    def test_case_insensitive(self):
        assert _mime_from_extension("TEST.PNG") == "image/png"


# ---------------------------------------------------------------------------
# Data URI decoding
# ---------------------------------------------------------------------------


class TestDataUriDecoding:
    def test_valid_data_uri(self):
        data = b"hello"
        uri = _make_data_uri(data, "image/png")
        result_bytes, mime = _decode_data_uri(uri)
        assert result_bytes == data
        assert mime == "image/png"

    def test_non_image_mime_rejected(self):
        b64 = base64.b64encode(b"notimage").decode()
        uri = f"data:text/plain;base64,{b64}"
        with pytest.raises(ValueError, match="unsupported data URL type"):
            _decode_data_uri(uri)

    def test_invalid_format_rejected(self):
        with pytest.raises(ValueError, match="expected base64"):
            _decode_data_uri("data:not-valid")

    def test_empty_payload_rejected(self):
        uri = "data:image/png;base64,"
        with pytest.raises(ValueError):
            _decode_data_uri(uri)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


class TestInputNormalization:
    def test_single_image(self):
        result = AnalyzeImageTool._normalize_inputs({"image": "test.png"})
        assert result == ["test.png"]

    def test_images_array(self):
        result = AnalyzeImageTool._normalize_inputs({"images": ["a.png", "b.png"]})
        assert result == ["a.png", "b.png"]

    def test_merge_image_and_images(self):
        result = AnalyzeImageTool._normalize_inputs({
            "image": "a.png",
            "images": ["b.png", "c.png"],
        })
        assert result == ["a.png", "b.png", "c.png"]

    def test_dedup_preserves_order(self):
        result = AnalyzeImageTool._normalize_inputs({
            "image": "a.png",
            "images": ["a.png", "b.png"],
        })
        assert result == ["a.png", "b.png"]

    def test_at_prefix_stripped_for_dedup(self):
        result = AnalyzeImageTool._normalize_inputs({
            "image": "@test.png",
            "images": ["test.png"],
        })
        # Both refer to same file after @ stripping — dedup
        assert len(result) == 1

    def test_empty_strings_filtered(self):
        result = AnalyzeImageTool._normalize_inputs({
            "images": ["", "  ", "valid.png"],
        })
        assert result == ["valid.png"]

    def test_no_inputs_returns_empty(self):
        result = AnalyzeImageTool._normalize_inputs({})
        assert result == []


# ---------------------------------------------------------------------------
# Remote VM image loading
# ---------------------------------------------------------------------------


class TestRemoteImageLoading:
    def test_remote_path_calls_interface(self):
        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response())
            result = tool.call({"image": r"C:\Users\User\test.png", "prompt": "What floor?"})

        tool.interface.read_bytes.assert_called_once_with(r"C:\Users\User\test.png")
        assert "Floor 2" in result

    def test_remote_read_failure(self):
        tool = _make_tool()
        tool.interface.read_bytes = AsyncMock(side_effect=Exception("connection lost"))
        result = tool.call({"image": r"C:\test.png"})
        assert "Error" in result
        assert "remote VM" in result


# ---------------------------------------------------------------------------
# Local filesystem loading
# ---------------------------------------------------------------------------


class TestLocalImageLoading:
    def test_local_path_reads_file(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(_REAL_PNG_BYTES)

        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("A cat"))
            result = tool.call({"image": str(img_file), "prompt": "What is this?"})

        assert "A cat" in result

    def test_local_file_not_found(self):
        tool = _make_tool()
        result = tool.call({"image": "/nonexistent/path/test.png"})
        assert "Error" in result
        assert "not found" in result


# ---------------------------------------------------------------------------
# Data URI loading
# ---------------------------------------------------------------------------


class TestDataUriLoading:
    def test_data_uri_accepted(self):
        tool = _make_tool()
        uri = _make_data_uri(_REAL_PNG_BYTES, "image/png")
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("An image"))
            result = tool.call({"image": uri})
        assert "An image" in result

    def test_invalid_data_uri(self):
        tool = _make_tool()
        result = tool.call({"image": "data:not-valid"})
        assert "Error" in result
        assert "invalid data URL" in result


# ---------------------------------------------------------------------------
# HTTP URL loading
# ---------------------------------------------------------------------------


class TestHttpUrlLoading:
    def test_http_url_fetched(self):
        tool = _make_tool()
        with (
            patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm,
            patch("ale_run.agents.ale_claw.harness.tools.analyze_image.AnalyzeImageTool._fetch_http") as mock_fetch,
        ):
            mock_fetch.return_value = (_REAL_PNG_BYTES, "image/png")
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("Web image"))
            result = tool.call({"image": "https://example.com/img.png"})
        assert "Web image" in result

    def test_http_fetch_failure(self):
        tool = _make_tool()
        with patch(
            "ale_run.agents.ale_claw.harness.tools.analyze_image.AnalyzeImageTool._fetch_http",
            new_callable=AsyncMock,
            side_effect=Exception("timeout"),
        ):
            result = tool.call({"image": "https://example.com/broken.png"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# Unsupported schemes
# ---------------------------------------------------------------------------


class TestUnsupportedSchemes:
    def test_ftp_rejected(self):
        tool = _make_tool()
        result = tool.call({"image": "ftp://server/file.png"})
        assert "Error" in result
        assert "unsupported image reference" in result

    def test_custom_scheme_rejected(self):
        tool = _make_tool()
        result = tool.call({"image": "image:0"})
        assert "Error" in result
        assert "unsupported" in result


# ---------------------------------------------------------------------------
# Size enforcement
# ---------------------------------------------------------------------------


class TestSizeEnforcement:
    def test_oversized_image_unreachable_budget(self):
        # tiny maxBytesMb forces resize budget below any
        # reachable JPEG output → sanitizer exhausts grid → placeholder
        # error surfaced as a tool-level Error string.
        tool = _make_tool()
        tool.interface.read_bytes = AsyncMock(return_value=_real_png(800, 800))
        result = tool.call({
            "image": r"C:\test.png",
            "maxBytesMb": 0.00005,  # ~50 bytes — unreachable.
        })
        assert "Error" in result
        assert "could not be reduced" in result

    def test_within_limit_accepted(self, tmp_path):
        img_file = tmp_path / "small.png"
        img_file.write_bytes(_REAL_PNG_BYTES)

        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("OK"))
            result = tool.call({"image": str(img_file), "maxBytesMb": 10})
        assert "OK" in result


# ---------------------------------------------------------------------------
# Multi-image
# ---------------------------------------------------------------------------


class TestMultiImage:
    def test_multiple_images_forwarded(self, tmp_path):
        f1 = tmp_path / "a.png"
        f2 = tmp_path / "b.png"
        f1.write_bytes(_real_png(color=(255, 0, 0)))
        f2.write_bytes(_real_png(color=(0, 255, 0)))

        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("Both images show..."))
            result = tool.call({
                "images": [str(f1), str(f2)],
                "prompt": "Compare these",
            })
        assert "Both images" in result
        # Verify VLM received 3 content items: 1 text + 2 images
        call_args = mock_litellm.acompletion.call_args
        content = call_args.kwargs["messages"][0]["content"]
        assert len(content) == 3  # text + 2 images

    def test_too_many_images_rejected(self):
        tool = _make_tool()
        result = tool.call({"images": [f"img{i}.png" for i in range(25)]})
        assert "Error" in result
        assert "too many images" in result


# ---------------------------------------------------------------------------
# No images provided
# ---------------------------------------------------------------------------


class TestNoImages:
    def test_empty_params(self):
        tool = _make_tool()
        result = tool.call({})
        assert "Error" in result
        assert "at least one image" in result

    def test_empty_image_string(self):
        tool = _make_tool()
        result = tool.call({"image": ""})
        assert "Error" in result


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------


class TestVlmCall:
    def test_default_prompt(self):
        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("A screenshot"))
            tool.call({"image": r"C:\test.png"})
            call_args = mock_litellm.acompletion.call_args
            content = call_args.kwargs["messages"][0]["content"]
            assert content[0]["text"] == "Describe the image."

    def test_custom_prompt_forwarded(self):
        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("Floor 3"))
            tool.call({"image": r"C:\test.png", "prompt": "What floor?"})
            call_args = mock_litellm.acompletion.call_args
            content = call_args.kwargs["messages"][0]["content"]
            assert content[0]["text"] == "What floor?"

    def test_model_forwarded(self):
        tool = _make_tool(model="my-custom-model")
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("OK"))
            tool.call({"image": r"C:\test.png"})
            call_args = mock_litellm.acompletion.call_args
            assert call_args.kwargs["model"] == "my-custom-model"

    def test_thinking_params_forwarded(self):
        tool = _make_tool(
            thinking_params={"thinking": {"type": "enabled", "budget_tokens": 5000}}
        )
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_vlm_response("OK"))
            tool.call({"image": r"C:\test.png"})
            call_args = mock_litellm.acompletion.call_args
            assert call_args.kwargs["thinking"] == {
                "type": "enabled",
                "budget_tokens": 5000,
            }

    def test_vlm_failure_returns_error(self):
        tool = _make_tool()
        with patch("ale_run.agents.ale_claw.harness.tools.analyze_image.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("rate limited"))
            result = tool.call({"image": r"C:\test.png"})
        assert "Error" in result
        assert "image analysis failed" in result
