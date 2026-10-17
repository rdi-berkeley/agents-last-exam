from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[2] / "ale_run/agents/_assets/cua_mcp_server/src/x11-paste.py"
)
spec = importlib.util.spec_from_file_location("cua_x11_paste", SOURCE)
assert spec is not None and spec.loader is not None
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def _display(monkeypatch: pytest.MonkeyPatch, has_extension: bool = True):
    constants = SimpleNamespace(
        CopyFromParent=0,
        InputOutput=1,
        PropertyChangeMask=1,
        CurrentTime=0,
        KeyPress=2,
        KeyRelease=3,
        SelectionClear=29,
        SelectionRequest=30,
        PropertyNotify=28,
        PropertyDelete=1,
        NONE=0,
    )
    events = deque()
    writes = []
    actions = []
    atoms = {
        "CLIPBOARD": 1,
        "TARGETS": 2,
        "UTF8_STRING": 3,
        "text/plain;charset=utf-8": 4,
        "INCR": 5,
    }

    class Window:
        def __init__(self, resource_id):
            self.id = resource_id

        def change_attributes(self, **kwargs):
            pass

        def change_property(self, property_atom, target, bits, data):
            writes.append((self.id, property_atom, target, bits, data))
            if target != 2:
                events.append(
                    SimpleNamespace(
                        type=constants.PropertyNotify,
                        state=constants.PropertyDelete,
                        window=self,
                        atom=property_atom,
                    )
                )

        def set_selection_owner(self, *args):
            pass

        def send_event(self, event):
            pass

        def create_window(self, *args, **kwargs):
            return self

        def get_wm_class(self):
            return ("fixture", "Tk")

        def query_tree(self):
            return SimpleNamespace(parent=owner)

    owner = Window(0x300001)
    focus = Window(0x100001)
    manager = Window(0x200001)

    class Connection:
        def has_extension(self, name):
            return has_extension

        def get_input_focus(self):
            return SimpleNamespace(focus=focus)

        def res_query_clients(self):
            return SimpleNamespace(
                clients=[SimpleNamespace(resource_mask=0xFFFFF, resource_base=0x100000)]
            )

        def screen(self):
            return SimpleNamespace(root=owner)

        def intern_atom(self, name):
            return atoms[name]

        def sync(self):
            pass

        def flush(self):
            pass

        def get_selection_owner(self, *args):
            return owner

        def keysym_to_keycode(self, key):
            return key

        def pending_events(self):
            return len(events)

        def next_event(self):
            if not events:
                raise AssertionError("Paste did not complete after consumption")
            return events.popleft()

        def close(self):
            actions.append("closed")

    def request(window):
        return SimpleNamespace(
            type=constants.SelectionRequest,
            requestor=window,
            property=42,
            target=3,
            time=0,
            selection=1,
        )

    def inject(connection, event_type, key):
        if event_type == constants.KeyRelease and key == "v":
            events.extend([request(manager), request(focus)])

    xlib = ModuleType("Xlib")
    xlib.X = constants
    xlib.Xatom = SimpleNamespace(ATOM=2)
    xlib.XK = SimpleNamespace(string_to_keysym=lambda key: key)
    xlib.display = SimpleNamespace(Display=Connection)
    xlib.protocol = SimpleNamespace(event=SimpleNamespace(SelectionNotify=lambda **kwargs: kwargs))
    ext = ModuleType("Xlib.ext")
    ext.xtest = SimpleNamespace(fake_input=inject)
    monkeypatch.setitem(sys.modules, "Xlib", xlib)
    monkeypatch.setitem(sys.modules, "Xlib.ext", ext)
    return events, writes, actions, focus, manager, constants


def test_ack_requires_focused_client_not_clipboard_manager(monkeypatch):
    events, writes, actions, focus, manager, constants = _display(monkeypatch)
    result = helper.paste(b"abc")
    assert result == {"consumed": True, "bytes": 3, "requestor": focus.id}
    assert {entry[0] for entry in writes} == {manager.id, focus.id}
    assert not events
    assert actions == ["closed"]


def test_large_text_requires_final_incr_property_consumption(monkeypatch):
    events, writes, actions, focus, manager, constants = _display(monkeypatch)
    payload = b"x" * 150000
    result = helper.paste(payload)
    chunks = [entry[4] for entry in writes if entry[0] == focus.id and entry[3] == 8]
    assert chunks[-1] == b""
    assert b"".join(chunks) == payload
    assert result["bytes"] == len(payload)
    assert actions == ["closed"]


def test_missing_consumer_is_an_error_not_a_success(monkeypatch):
    events, writes, actions, focus, manager, constants = _display(monkeypatch)
    with pytest.raises(TimeoutError, match="did not consume"):
        helper.paste(b"abc", timeout=0)
    assert actions == ["closed"]


@pytest.mark.parametrize("window_class", ["Gnome-terminal", "XTerm", "kitty", "konsole"])
def test_terminal_shortcut(window_class):
    focus = SimpleNamespace(id=10, get_wm_class=lambda: (window_class,))
    assert helper.paste_keys(focus, 0) == ["Control_L", "Shift_L", "v"]


def test_missing_xres_fails_before_injecting_input(monkeypatch):
    events, writes, actions, focus, manager, constants = _display(monkeypatch, has_extension=False)
    with pytest.raises(RuntimeError, match="X-Resource"):
        helper.paste(b"abc")
    assert not writes
    assert actions == ["closed"]
