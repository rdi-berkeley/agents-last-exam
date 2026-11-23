"""Tests for pointer coordinate handling in the ale_claw computer handlers.

Covers the Gemini "normalized [0, 1000) grid" fix:
  - coordinate_space_for_model() auto-detection
  - "pixel" space is an unscaled pass-through (default, back-compat)
  - "normalized" space rescales model coords to pixels via screen size
  - the session handler scales click and drag actions consistently
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from ale_claw.harness.tools.computer_handler import (
    NORMALIZED_GRID,
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
