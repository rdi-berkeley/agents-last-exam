"""Paste UTF-8 text and acknowledge transfer to the focused X11 client."""

import base64
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import sys
import time


def paste_keys(focus, root_id: int) -> list[str]:
    terminal_classes = {
        "gnome-terminal",
        "gnome-terminal-server",
        "kgx",
        "konsole",
        "xterm",
        "xfce4-terminal",
        "tilix",
        "kitty",
        "alacritty",
        "terminator",
        "wezterm",
        "mate-terminal",
        "lxterminal",
        "rxvt",
        "urxvt",
        "qterminal",
        "st",
        "st-256color",
    }
    window = focus
    while window.id != root_id:
        classes = window.get_wm_class() or ()
        if terminal_classes.intersection(value.lower() for value in classes):
            return ["Control_L", "Shift_L", "v"]
        window = window.query_tree().parent
    return ["Control_L", "v"]


def paste(text: bytes, timeout: float = 15) -> dict:
    from Xlib import X, Xatom, XK, display, protocol
    from Xlib.ext import xtest

    connection = display.Display()
    try:
        focus = connection.get_input_focus().focus
        if not hasattr(focus, "id"):
            raise RuntimeError("No focused X11 application")
        if not connection.has_extension("X-Resource"):
            raise RuntimeError("Acknowledged paste requires the X11 X-Resource extension")
        clients = connection.res_query_clients().clients
        focused_client = next(
            client for client in clients if focus.id & ~client.resource_mask == client.resource_base
        )
        owner = connection.screen().root.create_window(
            0,
            0,
            1,
            1,
            0,
            X.CopyFromParent,
            X.InputOutput,
            X.CopyFromParent,
            event_mask=X.PropertyChangeMask,
        )
        clipboard = connection.intern_atom("CLIPBOARD")
        targets = connection.intern_atom("TARGETS")
        utf8 = connection.intern_atom("UTF8_STRING")
        plain = connection.intern_atom("text/plain;charset=utf-8")
        incr = connection.intern_atom("INCR")
        owner.set_selection_owner(clipboard, X.CurrentTime)
        connection.sync()
        if connection.get_selection_owner(clipboard).id != owner.id:
            raise RuntimeError("Could not acquire X11 clipboard")
        keys = [
            connection.keysym_to_keycode(XK.string_to_keysym(key))
            for key in paste_keys(focus, connection.screen().root.id)
        ]
        for key in keys:
            xtest.fake_input(connection, X.KeyPress, key)
        for key in reversed(keys):
            xtest.fake_input(connection, X.KeyRelease, key)
        connection.flush()
        deadline = time.monotonic() + timeout
        transfers = {}
        observed = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Focused application did not consume clipboard text: {observed[-12:]}"
                )
            if not connection.pending_events():
                if not select.select([connection.fileno()], [], [], remaining)[0]:
                    raise TimeoutError(
                        f"Focused application did not consume clipboard text: {observed[-12:]}"
                    )
            event = connection.next_event()
            if event.type == X.SelectionClear:
                raise RuntimeError("Clipboard ownership changed before consumption")
            if event.type == X.SelectionRequest:
                requestor = event.requestor
                observed.append(("request", hex(requestor.id), event.target, hex(focus.id)))
                property_atom = event.property or event.target
                accepted = property_atom
                if event.target == targets:
                    requestor.change_property(property_atom, Xatom.ATOM, 32, [targets, utf8, plain])
                elif event.target in (utf8, plain):
                    requestor.change_attributes(event_mask=X.PropertyChangeMask)
                    same_client = (
                        requestor.id & ~focused_client.resource_mask == focused_client.resource_base
                    )
                    requestor.change_property(property_atom, incr, 32, [len(text)])
                    transfers[(requestor.id, property_atom)] = [
                        requestor,
                        event.target,
                        0,
                        same_client,
                    ]
                else:
                    accepted = X.NONE
                requestor.send_event(
                    protocol.event.SelectionNotify(
                        time=event.time,
                        requestor=requestor,
                        selection=event.selection,
                        target=event.target,
                        property=accepted,
                    )
                )
                connection.flush()
            elif event.type == X.PropertyNotify and event.state == X.PropertyDelete:
                observed.append(("delete", hex(event.window.id), event.atom))
                transfer = transfers.get((event.window.id, event.atom))
                if transfer is None:
                    continue
                requestor, target, offset, same_client = transfer
                if offset is None:
                    del transfers[(event.window.id, event.atom)]
                    if same_client:
                        return {"consumed": True, "bytes": len(text), "requestor": requestor.id}
                else:
                    chunk = text[offset : offset + 65536]
                    requestor.change_property(event.atom, target, 8, chunk)
                    transfer[2] = offset + len(chunk) if chunk else None
                    connection.flush()
    finally:
        connection.close()


def main() -> None:
    signal.alarm(18)
    if sys.argv[1] == "--file":
        payload = Path(sys.argv[2])
        text = payload.read_bytes()
        payload.unlink()
    else:
        text = base64.b64decode(sys.argv[1], validate=True)
    lock_dir = Path.home() / ".cache" / "ale-cua"
    lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    display_key = base64.urlsafe_b64encode(os.environ.get("DISPLAY", "").encode()).decode()
    with (lock_dir / f"input-{display_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(paste(text)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
