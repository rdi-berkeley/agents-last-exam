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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from agent.computers.cua import cuaComputerHandler
from agent.types import ToolError


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


class OpenClawComputerHandler(cuaComputerHandler):
    """cuaComputerHandler with sequential multi-key keypress and tolerant
    coercion for malformed pointer coordinates."""

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

    def _require_xy(self, action: str, x: Any, y: Any) -> Tuple[int, int]:
        nx, ny = _coerce_xy(x, y)
        if nx is None or ny is None:
            raise ToolError(
                f"computer.{action} requires both x and y coordinates "
                f"(got x={x!r}, y={y!r}); pass them as separate integers, "
                f'e.g. {{"action": "{action}", "x": 420, "y": 390}}.'
            )
        return nx, ny

    async def click(self, x: Any, y: Any = None, button: str = "left") -> None:
        nx, ny = self._require_xy("click", x, y)
        await super().click(nx, ny, button)

    async def double_click(self, x: Any, y: Any = None) -> None:
        nx, ny = self._require_xy("double_click", x, y)
        await super().double_click(nx, ny)

    async def right_click(self, x: Any, y: Any = None) -> None:
        nx, ny = self._require_xy("right_click", x, y)
        await super().right_click(nx, ny)

    async def move(self, x: Any, y: Any = None) -> None:
        nx, ny = self._require_xy("move", x, y)
        await super().move(nx, ny)

    async def scroll(
        self, x: Any, y: Any = None, scroll_x: Any = 0, scroll_y: Any = 0
    ) -> None:
        nx, ny = self._require_xy("scroll", x, y)
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
            await super().drag(path=path)
            return
        sx, sy = self._require_xy("drag (start)", start_x, start_y)
        ex, ey = self._require_xy("drag (end)", end_x, end_y)
        await super().drag(start_x=sx, start_y=sy, end_x=ex, end_y=ey)
