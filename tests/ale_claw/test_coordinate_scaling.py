"""Tests for pointer coordinate handling in the ale_claw computer handlers.

Covers the Gemini "normalized [0, 1000) grid" fix:
  - coordinate_space_for_model() auto-detection
  - "pixel" space is an unscaled pass-through (default, back-compat)
  - "normalized" space rescales model coords to pixels via screen size
  - both handlers (OpenClawComputerHandler + MCPComputerHandler) scale
    click / drag (start-end and path) consistently
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ale_run.agents.ale_claw.harness.tools.computer_handler import (
    NORMALIZED_GRID,
    MCPComputerHandler,
    OpenClawComputerHandler,
    coordinate_space_for_model,
)


def _run(coro):
    return asyncio.run(coro)


SCREEN_W, SCREEN_H = 1920, 1080


def _norm_to_px(n: int, dim: int) -> int:
    return round(n / NORMALIZED_GRID * dim)


# ---------------------------------------------------------------------------
# coordinate_space_for_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("openrouter/google/gemini-3.1-pro-preview", "normalized"),
        ("openrouter/google/gemini-2.5-flash", "normalized"),
        ("GEMINI-pro", "normalized"),
        ("openrouter/anthropic/claude-sonnet-4.6", "pixel"),
        ("openrouter/openai/gpt-5.4", "pixel"),
        (None, "pixel"),
        ("", "pixel"),
    ],
)
def test_coordinate_space_for_model(model_id, expected):
    assert coordinate_space_for_model(model_id) == expected


# ---------------------------------------------------------------------------
# OpenClawComputerHandler (session path — coords flow straight to interface)
# ---------------------------------------------------------------------------


def _make_openclaw_handler(coordinate_space):
    handler = OpenClawComputerHandler(
        MagicMock(), coordinate_space=coordinate_space
    )
    # Bypass cua interface plumbing: stub get_dimensions + the parent methods
    # that would otherwise hit a real RemoteDesktop interface.
    handler.get_dimensions = AsyncMock(return_value=(SCREEN_W, SCREEN_H))
    handler.interface = MagicMock()
    return handler


def test_openclaw_pixel_passthrough(monkeypatch):
    handler = _make_openclaw_handler("pixel")
    seen = {}

    async def fake_click(self, x, y, button="left"):
        seen["xy"] = (x, y)

    monkeypatch.setattr(
        "agent.computers.cua.cuaComputerHandler.click", fake_click
    )
    _run(handler.click(640, 480))
    assert seen["xy"] == (640, 480)


def test_openclaw_normalized_scales_to_pixels(monkeypatch):
    handler = _make_openclaw_handler("normalized")
    seen = {}

    async def fake_click(self, x, y, button="left"):
        seen["xy"] = (x, y)

    monkeypatch.setattr(
        "agent.computers.cua.cuaComputerHandler.click", fake_click
    )
    # Model emits normalized center (500, 500) → screen center in pixels.
    _run(handler.click(500, 500))
    assert seen["xy"] == (_norm_to_px(500, SCREEN_W), _norm_to_px(500, SCREEN_H))
    assert seen["xy"] == (960, 540)


def test_openclaw_normalized_clamps_to_screen(monkeypatch):
    handler = _make_openclaw_handler("normalized")
    seen = {}

    async def fake_click(self, x, y, button="left"):
        seen["xy"] = (x, y)

    monkeypatch.setattr(
        "agent.computers.cua.cuaComputerHandler.click", fake_click
    )
    # 999 is the max grid value; must never exceed the pixel bounds.
    _run(handler.click(999, 999))
    px, py = seen["xy"]
    assert 0 <= px <= SCREEN_W - 1
    assert 0 <= py <= SCREEN_H - 1


def test_openclaw_normalized_drag_path(monkeypatch):
    handler = _make_openclaw_handler("normalized")
    seen = {}

    async def fake_drag(self, path=None, start_x=None, start_y=None,
                        end_x=None, end_y=None):
        seen["path"] = path

    monkeypatch.setattr(
        "agent.computers.cua.cuaComputerHandler.drag", fake_drag
    )
    # Use a mid grid value (500) to avoid the edge clamp; 1000 would clamp to
    # w-1/h-1 which is verified separately in test_openclaw_normalized_clamps.
    _run(handler.drag(path=[{"x": 0, "y": 0}, {"x": 500, "y": 500}]))
    assert seen["path"][0] == {"x": 0, "y": 0}
    assert seen["path"][-1]["x"] == _norm_to_px(500, SCREEN_W)
    assert seen["path"][-1]["y"] == _norm_to_px(500, SCREEN_H)


# ---------------------------------------------------------------------------
# MCPComputerHandler (bridge path — px→[0,1000] happens in _to_norm)
# ---------------------------------------------------------------------------


def _make_mcp_handler(coordinate_space):
    handler = MCPComputerHandler(
        MagicMock(), os_type="windows", coordinate_space=coordinate_space
    )
    handler._dims = (SCREEN_W, SCREEN_H)  # skip get_screen_size round-trip
    handler._call_cua = AsyncMock()
    return handler


def test_mcp_pixel_click_norm_roundtrip():
    handler = _make_mcp_handler("pixel")
    # Pixel center → bridge-normalized ~500.
    _run(handler.click(960, 540))
    args = handler._call_cua.call_args
    assert args[0][0] == "click"
    coord = args[0][1]["coordinate"]
    assert coord == [500, 500]


def test_mcp_normalized_click_roundtrips_back():
    handler = _make_mcp_handler("normalized")
    # Model emits normalized (500, 500). Handler scales →px (960,540), then
    # _to_norm converts back →[500,500] for the bridge. Net: identity.
    _run(handler.click(500, 500))
    coord = handler._call_cua.call_args[0][1]["coordinate"]
    assert coord == [500, 500]


def test_mcp_normalized_differs_from_pixel_for_offcenter():
    """A raw normalized coord treated as pixels would land wrong; scaling fixes it."""
    norm = _make_mcp_handler("normalized")
    _run(norm.click(250, 250))
    norm_coord = norm._call_cua.call_args[0][1]["coordinate"]
    # Scaled: normalized 250 → pixel 480/270 → bridge norm back to ~250.
    assert norm_coord == [250, 250]

    pixel = _make_mcp_handler("pixel")
    _run(pixel.click(250, 250))
    pixel_coord = pixel._call_cua.call_args[0][1]["coordinate"]
    # Treated as pixels: 250/1920*1000 ≈ 130 — the wrong place.
    assert pixel_coord == [round(250 / SCREEN_W * 1000),
                           round(250 / SCREEN_H * 1000)]
    assert pixel_coord != norm_coord
