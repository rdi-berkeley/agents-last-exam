"""OpenClaw-specific cuaComputerHandler subclass.

Single source of truth for keypress semantics in the OpenClaw harness.

Why override ``keypress``:
    The OpenAI computer-use spec leaves ``keys=["right","right","down"]``
    ambiguous between chord (hold all keys) and sequence (press in order).
    Upstream cuaComputerHandler always routes a list of length > 1 through
    ``hotkey``, which collapses duplicates and produces unintended
    diagonal-key holds — silently breaking games (e.g. Magic Tower) and
    any app that expects discrete key events.

Why ship a local ``_normalize_key``:
    Older ``cua-verse/cua`` mainline pins (e.g. ``2a10d326``) do not define
    ``_normalize_key`` on ``cuaComputerHandler``; relying on the parent
    raised ``AttributeError`` the moment a list-keypress fires. Carrying
    the shim here keeps the openclaw harness working across both pins
    that ship the helper upstream and pins that don't.

Convention used here:
    - String input (``"ctrl+shift+s"`` or legacy ``"ctrl-shift-s"``) → chord.
    - List input (``["right","right","down"]``) → sequence of independent
      presses, in order.
    - Single-element list (``["enter"]``) → ``press_key`` (unchanged).

Why coerce pointer coordinates:
    Sonnet 4.6 occasionally emits ``{"action": "click", "x": "420, 390"}``
    — packing both coordinates into ``x`` as a comma-joined string with
    no ``y``. Upstream ``cuaComputerHandler.click(x, y, ...)`` then raises
    ``TypeError: missing 1 required positional argument: 'y'``, which the
    agent loop does not catch and which kills the entire run. Coercing
    that shape (and the related "x is a numeric string" pattern) here lets
    the model recover via the next observation instead of aborting the
    task.

    When the input is genuinely incomplete (e.g. only ``x`` provided with
    no recoverable ``y``), we raise ``ToolError`` rather than passing
    ``None`` to the parent. The agent loop converts ``ToolError`` into a
    ``function_call_output`` that the model sees on the next turn, so the
    run stays alive instead of crashing.

Why scale normalized coordinates:
    Some GUI models (notably the Gemini family) ground clicks to a fixed
    ``[0, 1000)`` grid rather than to screen pixels — regardless of the
    "Screen resolution: WxH pixels" hint the ``computer`` tool advertises.
    Passed through unscaled, a normalized click at ``(500, 500)`` lands near
    the top-left of a 1920x1080 screen instead of its center. When a handler
    is told its model emits ``"normalized"`` coordinates, it maps them back to
    pixels (``px = n / 1000 * W``) before the usual pixel-space path runs; the
    default ``"pixel"`` space is an unscaled pass-through. See
    :class:`_PointerCoords` and :func:`coordinate_space_for_model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from agent.computers.cua import cuaComputerHandler
from agent.types import ToolError

if TYPE_CHECKING:
    from .mcp_runtime import MCPRuntime


_KEY_MAPPING = {
    "ARROWUP": "up",
    "ARROWDOWN": "down",
    "ARROWLEFT": "left",
    "ARROWRIGHT": "right",
}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _coerce_xy(x: Any, y: Any) -> Tuple[Optional[int], Optional[int]]:
    """Normalize a model-emitted (x, y) pair.

    Handles the common Sonnet 4.6 malformation where both coordinates are
    packed into ``x`` as a string like ``"420, 390"`` (or ``"420 390"``,
    ``"(420, 390)"``) with ``y`` missing. Otherwise just coerces numeric
    strings to int.
    """
    if y is None and isinstance(x, str):
        s = x.strip().lstrip("([{").rstrip(")]}")
        for sep in (",", ";", " "):
            if sep in s:
                parts = [p for p in (p.strip() for p in s.split(sep)) if p]
                if len(parts) == 2:
                    return _coerce_int(parts[0]), _coerce_int(parts[1])
    if y is None and isinstance(x, (list, tuple)) and len(x) == 2:
        return _coerce_int(x[0]), _coerce_int(x[1])
    return _coerce_int(x), _coerce_int(y)


NORMALIZED_GRID = 1000
"""Grid size for models that emit *normalized* pointer coordinates.

Gemini-family GUI models ground clicks to a fixed ``[0, 1000)`` grid rather
than to screen pixels. To recover a pixel coordinate we divide by this and
multiply by the screen dimension: ``px = n / NORMALIZED_GRID * W``.

Note the model emits values in ``0-999`` but the denominator is ``1000`` (not
999), per Google's reference denormalization ``int(x / 1000 * width)``. See
the Gemini computer-use docs: https://ai.google.dev/gemini-api/docs/computer-use
"""


def coordinate_space_for_model(model_id: Optional[str]) -> str:
    """Best-effort pointer coordinate space for a LiteLLM model id.

    Returns ``"normalized"`` for Gemini-family models (which ground clicks to a
    fixed ``[0, 1000)`` grid) and ``"pixel"`` for everything else. This is only
    a default — callers can force either space via
    ``AleClawConfig.coordinate_space``.
    """
    if model_id and "gemini" in model_id.lower():
        return "normalized"
    return "pixel"


class _PointerCoords:
    """Coordinate coercion + optional normalized→pixel scaling for pointers.

    Mixed into both computer handlers so the two coordinate conventions live in
    one place. ``_coord_space`` — set by each handler's ``__init__`` — selects
    the behavior:

    - ``"pixel"`` (default): the model already emits screen-pixel coords, so
      just coerce and pass through.
    - ``"normalized"``: the model emits a :data:`NORMALIZED_GRID` grid (Gemini),
      so coerce, then scale to pixels using the live screen size.

    Requires the host handler to provide an async ``get_dimensions()`` returning
    ``(width, height)`` — both handlers satisfy this.
    """

    _coord_space: str = "pixel"

    def _require_xy(self, action: str, x: Any, y: Any) -> Tuple[int, int]:
        nx, ny = _coerce_xy(x, y)
        if nx is None or ny is None:
            raise ToolError(
                f"computer.{action} requires both x and y coordinates "
                f"(got x={x!r}, y={y!r}); pass them as separate integers, "
                f'e.g. {{"action": "{action}", "x": 420, "y": 390}}.'
            )
        return nx, ny

    async def _scale_xy(self, x: int, y: int) -> Tuple[int, int]:
        """Map (x, y) to pixel space when the model emits normalized coords."""
        if self._coord_space != "normalized":
            return x, y
        w, h = await self.get_dimensions()
        px = round(x / NORMALIZED_GRID * w)
        py = round(y / NORMALIZED_GRID * h)
        return max(0, min(w - 1, px)), max(0, min(h - 1, py))

    async def _pointer_xy(self, action: str, x: Any, y: Any) -> Tuple[int, int]:
        """Coerce then (for normalized models) scale a single (x, y) pair."""
        nx, ny = self._require_xy(action, x, y)
        return await self._scale_xy(nx, ny)

    async def _scale_path(
        self, path: List[Dict[str, int]]
    ) -> List[Dict[str, int]]:
        """Scale every point of a drag ``path`` when coords are normalized."""
        if self._coord_space != "normalized":
            return path
        scaled: List[Dict[str, int]] = []
        for pt in path:
            px, py = await self._scale_xy(
                _coerce_int(pt.get("x")) or 0, _coerce_int(pt.get("y")) or 0
            )
            scaled.append({**pt, "x": px, "y": py})
        return scaled


class OpenClawComputerHandler(_PointerCoords, cuaComputerHandler):
    """cuaComputerHandler with sequential multi-key keypress, tolerant
    coercion for malformed pointer coordinates, and optional normalized→pixel
    scaling for models (e.g. Gemini) that emit a ``[0, 1000)`` grid."""

    def __init__(
        self, cua_computer: Any, *, coordinate_space: str = "pixel"
    ) -> None:
        super().__init__(cua_computer)
        self._coord_space = coordinate_space

    def _normalize_key(self, key: str) -> str:
        parent = getattr(super(), "_normalize_key", None)
        if callable(parent):
            return parent(key)
        return _KEY_MAPPING.get(key.upper(), key.lower())

    async def keypress(self, keys: Union[List[str], str]) -> None:
        assert self.interface is not None
        if isinstance(keys, str):
            await super().keypress(keys)
            return
        for k in keys:
            await self.interface.press_key(self._normalize_key(k))

    async def click(self, x: Any, y: Any = None, button: str = "left") -> None:
        nx, ny = await self._pointer_xy("click", x, y)
        await super().click(nx, ny, button)

    async def double_click(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("double_click", x, y)
        await super().double_click(nx, ny)

    async def right_click(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("right_click", x, y)
        await super().right_click(nx, ny)

    async def move(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("move", x, y)
        await super().move(nx, ny)

    async def scroll(
        self, x: Any, y: Any = None, scroll_x: Any = 0, scroll_y: Any = 0
    ) -> None:
        nx, ny = await self._pointer_xy("scroll", x, y)
        sx = _coerce_int(scroll_x) or 0
        sy = _coerce_int(scroll_y) or 0
        await super().scroll(nx, ny, sx, sy)

    async def drag(
        self,
        path: Optional[List[Dict[str, int]]] = None,
        start_x: Any = None,
        start_y: Any = None,
        end_x: Any = None,
        end_y: Any = None,
    ) -> None:
        if path:
            await super().drag(path=await self._scale_path(path))
            return
        sx, sy = await self._pointer_xy("drag (start)", start_x, start_y)
        ex, ey = await self._pointer_xy("drag (end)", end_x, end_y)
        await super().drag(start_x=sx, start_y=sy, end_x=ex, end_y=ey)


def _normalize_key(key: str) -> str:
    """Map LLM-style key names to the form the cua bridge expects.

    The bridge does its own normalization too; this only smooths the arrow-key
    aliases the harness has always handled and lowercases the rest.
    """
    return _KEY_MAPPING.get(key.upper(), key.lower())


class MCPComputerHandler(_PointerCoords):
    """GUI computer handler that drives the VM through the cua MCP bridge.

    Implements the ``AsyncComputerHandler`` protocol (a ``runtime_checkable``
    Protocol — only method *names* are checked), so it drops into ``build_tools``
    as the ``computer`` tool in place of :class:`OpenClawComputerHandler` when GUI
    is routed over MCP (Phase 2). Every action becomes a cua-bridge tool call;
    the harness no longer touches ``RemoteDesktopSession`` for GUI.

    Coordinates: the model emits **pixel** coords in screenshot space (or, for
    ``coordinate_space="normalized"`` models like Gemini, a [0, 1000) grid that
    :class:`_PointerCoords` first rescales to pixels), but the cua bridge speaks
    **normalized [0, 1000]**. We convert px→[0,1000] here using the screen size
    (fetched once via the bridge's ``get_screen_size`` tool); the bridge converts
    back to pixels. Rounding is ≤ 1px at typical resolutions.

    Keypress keeps the harness's chord-vs-sequence rule (mirrors
    :class:`OpenClawComputerHandler`): a list is a *sequence* of independent
    presses (one ``key`` call each), a ``+``/``-`` string is a *chord*.
    """

    def __init__(
        self,
        runtime: "MCPRuntime",
        *,
        os_type: Optional[str] = None,
        coordinate_space: str = "pixel",
    ) -> None:
        self._runtime = runtime
        self._os_type = (os_type or "").lower()
        self._dims: Optional[Tuple[int, int]] = None
        self._coord_space = coordinate_space
        # Present for parity with cua handlers that expose ``.interface``; the
        # MCP path has no direct interface object.
        self.interface = None

    async def _initialize(self) -> None:
        await self._dimensions()

    # -- helpers --

    async def _call_cua(self, tool: str, args: dict):
        """Call a cua bridge tool, mapping tool failures to ``ToolError``.

        A GUI op can fail for recoverable, environment-dependent reasons (e.g.
        the desktop is locked, a window isn't focused). The harness loop turns
        ``ToolError`` into a ``function_call_output`` the model sees next turn, so
        the run adapts instead of crashing — matching the session handler's
        semantics. Without this, the raw ``MCPToolError`` (a ``RuntimeError``)
        would propagate and abort the episode.
        """
        from .mcp_runtime import MCPToolError
        try:
            return await self._runtime.call("cua", tool, args)
        except MCPToolError as e:
            raise ToolError(str(e)) from e

    async def _dimensions(self) -> Tuple[int, int]:
        if self._dims is None:
            res = await self._call_cua("get_screen_size", {})
            sc = res.structuredContent or {}
            self._dims = (int(sc["width"]), int(sc["height"]))
        return self._dims

    async def _to_norm(self, x: int, y: int) -> List[int]:
        w, h = await self._dimensions()
        nx = 0 if w <= 0 else round(x / w * 1000)
        ny = 0 if h <= 0 else round(y / h * 1000)
        return [max(0, min(1000, nx)), max(0, min(1000, ny))]

    # -- observation --

    async def get_environment(self) -> str:
        if self._os_type.startswith("win"):
            return "windows"
        if self._os_type.startswith("mac") or self._os_type.startswith("darwin"):
            return "mac"
        return "linux"

    async def get_dimensions(self) -> Tuple[int, int]:
        return await self._dimensions()

    async def screenshot(self, text: Optional[str] = None) -> str:
        res = await self._call_cua("screenshot", {})
        for block in res.content:
            if getattr(block, "type", None) == "image":
                return block.data  # base64 str
        raise ToolError("cua screenshot returned no image content")

    async def get_current_url(self) -> str:
        return ""  # no browser-URL primitive over the cua bridge

    # -- pointer --

    async def click(self, x: Any, y: Any = None, button: str = "left") -> None:
        nx, ny = await self._pointer_xy("click", x, y)
        await self._call_cua("click", {"coordinate": await self._to_norm(nx, ny), "button": button})

    async def double_click(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("double_click", x, y)
        await self._call_cua("click", {"coordinate": await self._to_norm(nx, ny), "clicks": 2})

    async def right_click(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("right_click", x, y)
        await self._call_cua("click", {"coordinate": await self._to_norm(nx, ny), "button": "right"})

    async def move(self, x: Any, y: Any = None) -> None:
        nx, ny = await self._pointer_xy("move", x, y)
        await self._call_cua("mouse_move", {"coordinate": await self._to_norm(nx, ny)})

    async def scroll(self, x: Any, y: Any = None, scroll_x: Any = 0, scroll_y: Any = 0) -> None:
        nx, ny = await self._pointer_xy("scroll", x, y)
        coord = await self._to_norm(nx, ny)
        sx = _coerce_int(scroll_x) or 0
        sy = _coerce_int(scroll_y) or 0
        if sy:
            await self._call_cua("scroll", {
                "direction": "down" if sy > 0 else "up", "amount": abs(sy), "coordinate": coord})
        if sx:
            await self._call_cua("scroll", {
                "direction": "right" if sx > 0 else "left", "amount": abs(sx), "coordinate": coord})

    async def drag(
        self,
        path: Optional[List[Dict[str, int]]] = None,
        start_x: Any = None,
        start_y: Any = None,
        end_x: Any = None,
        end_y: Any = None,
    ) -> None:
        if path:
            path = await self._scale_path(path)
            sx, sy = path[0]["x"], path[0]["y"]
            ex, ey = path[-1]["x"], path[-1]["y"]
        else:
            sx, sy = await self._pointer_xy("drag (start)", start_x, start_y)
            ex, ey = await self._pointer_xy("drag (end)", end_x, end_y)
        await self._call_cua("drag", {
            "start_coordinate": await self._to_norm(sx, sy),
            "coordinate": await self._to_norm(ex, ey),
            "button": "left",
        })

    async def left_mouse_down(self, x: Any = None, y: Any = None) -> None:
        if x is not None and y is not None:
            await self.move(x, y)
        await self._call_cua("mouse_down", {"button": "left"})

    async def left_mouse_up(self, x: Any = None, y: Any = None) -> None:
        if x is not None and y is not None:
            await self.move(x, y)
        await self._call_cua("mouse_up", {"button": "left"})

    # -- keyboard --

    async def type(self, text: str) -> None:
        await self._call_cua("type", {"text": text})

    async def keypress(self, keys: Union[List[str], str]) -> None:
        if isinstance(keys, str):
            # chord: split "ctrl+shift+s" / legacy "ctrl-shift-s"
            parts = [p for p in keys.replace("-", "+").split("+") if p] or [keys]
            await self._call_cua("key", {"keys": [_normalize_key(k) for k in parts]})
            return
        # list: sequence of independent presses, in order
        for k in keys:
            await self._call_cua("key", {"keys": [_normalize_key(k)]})

    async def wait(self, ms: int = 1000) -> None:
        await self._call_cua("wait", {"duration": (ms or 0) / 1000})

    # -- misc --

    async def terminate(self, status: str = "success") -> Dict[str, Any]:
        return {"status": status}
