"""Demo task: ``demo/clickhit`` (Linux) — does the agent click where it means to?

Linux counterpart of ``demo/clickhit_win``; the pointer-accuracy sibling of
``demo/seecheck`` (which isolates the *vision* path, this the *click* path).

``setup``:
  1. Reads the live screen size and computes a single **off-center** target
     (default 80% width, 50% height — a clear area right of center, far enough
     from the left edge that a normalized-vs-pixel bug diverges by hundreds of
     px in x).
  2. Paints that target — a red box with a white crosshair at its exact center,
     plus instructions — onto the desktop **wallpaper** (the channel
     ``seecheck`` proves the harness screenshots), via the shared GNOME/XFCE
     ``set_desktop_wallpaper`` helper.
  3. Parks the mouse cursor in the top-left corner, and records the target in
     ``truth.json``.

Capture mechanism (differs from the Windows probe): X11's low-level input tools
can't be launched as a persistent recorder here — the cua-server's
``run_command`` blocks until every descendant exits, so a streaming ``xinput``
recorder would hang setup. Instead we use the fact that **the X pointer rests
wherever the last click landed**: a left click warps the cursor to the click
point, and nothing in a single-click task moves it afterward. ``evaluate`` reads
the resting pointer with ``xdotool getmouselocation`` and compares it to the
target. (Limitation: this sees only the final pointer position, so the task asks
for exactly one click — enough to isolate the coordinate mapping.)

Why this exposes the Gemini coordinate bug: Gemini grounds clicks to a fixed
``[0, 1000)`` grid, not screen pixels. For a target at ``(0.80W, 0.50H)`` on a
1920x1080 screen the model emits ``~(800, 500)``. Forwarded unscaled, the click
lands at pixel ``(800, 500)`` — 736 px left of the target — and the pointer
rests there → 0.0. With normalized→pixel rescaling (this branch's fix) it maps
to ``(1536, 540)`` → 1.0. Same-branch A/B: ``coordinate_space: pixel``
(reproduces the miss) vs ``normalized``/``null`` (the fix).

Self-contained (``REQUIRES_TASK_DATA = False``): stages no data. The target is
chosen at ``setup`` from the live resolution and written to ``truth.json`` (read
back by ``evaluate``); it is intentionally NOT in the prompt.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig, set_desktop_wallpaper

logger = logging.getLogger(__name__)

DOMAIN_NAME = "demo"
TASK_NAME = "clickhit"
VARIANT_NAME = "base"

PROBE_DIR = "/tmp/clickprobe"
TRUTH_PATH = "/tmp/clickprobe/target.png.json"  # truth beside the rendered image
IMAGE_PATH = "/tmp/clickprobe/target.png"

# Target placement as an integer percentage of the screen. Off-center (right of
# center) so the normalized-vs-pixel divergence in x is large and unambiguous,
# while staying clear of the GNOME dock (left) and top bar.
TARGET_PCT_X = 80
TARGET_PCT_Y = 50
TARGET_W = 200
TARGET_H = 140
# A hit is a resting pointer within this many px of the target center — generous
# vs. the box, tiny vs. the coordinate-bug miss (~736px in x at 1920 wide).
TOLERANCE_PX = 70


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REQUIRES_TASK_DATA: bool = False

    @property
    def task_description(self) -> str:
        return (
            "Pointer check — a GUI task. The desktop shows a single red target "
            "box with a white crosshair at its center, on a dark background, "
            "labeled 'TARGET'.\n\n"
            "1. Take a screenshot of the desktop.\n"
            "2. Click ONCE, precisely on the white crosshair at the CENTER of "
            "the red box.\n\n"
            "That is the entire task — a single accurate click. Do not click "
            "anywhere else first, and do not move the mouse after clicking. You "
            "do not need to write any file."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "probe_dir": PROBE_DIR,
            "truth_path": TRUTH_PATH,
            "tolerance_px": TOLERANCE_PX,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={
            "provider": "computer",
            "setup_config": {"os_type": cfg.OS_TYPE},
        },
    )]


# Renders the target wallpaper, records truth.json, and parks the cursor. No
# background process — everything here completes, so run_command returns cleanly.
# Placeholders are substituted below (kept out of an f-string to avoid escaping
# the shell's ``$(( ))`` arithmetic and ``${}`` expansions).
_SETUP_SH = r"""
set -e
export DISPLAY="${DISPLAY:-:0}"
mkdir -p __PROBE_DIR__

# Live screen geometry -> off-center target center (tx,ty) and box corners.
# POSIX sh (dash): no process substitution — split the "W H" output via set --.
set -- $(xdotool getdisplaygeometry)
W=$1; H=$2
tx=$(( W * __PCTX__ / 100 )); ty=$(( H * __PCTY__ / 100 ))
tw=__TW__; th=__TH__
rx=$(( tx - tw/2 )); ry=$(( ty - th/2 )); rx2=$(( tx + tw/2 )); ry2=$(( ty + th/2 ))

# Paint: dark background, filled red box, white border, white crosshair, labels.
convert -size ${W}x${H} xc:'#14141c' \
  -fill white -pointsize 34 -annotate +60+90 'POINTER ACCURACY CHECK' \
  -pointsize 22 -annotate +60+140 'Click the WHITE CROSSHAIR at the center of the RED box.' \
  -fill '#dc143c' -draw "rectangle ${rx},${ry} ${rx2},${ry2}" \
  -stroke white -strokewidth 3 -fill none -draw "rectangle ${rx},${ry} ${rx2},${ry2}" \
  -stroke white -strokewidth 3 -draw "line $((tx-20)),${ty} $((tx+20)),${ty}" \
  -draw "line ${tx},$((ty-20)) ${tx},$((ty+20))" \
  -stroke none -fill white -pointsize 18 -annotate +${rx}+$((ry-12)) 'TARGET' \
  __IMAGE_PATH__

# Ground truth for evaluate().
printf '{"x":%d,"y":%d,"w":%d,"h":%d}\n' "$tx" "$ty" "$W" "$H" > __TRUTH_PATH__

# Park the cursor in the corner so a "never clicked" run can't accidentally rest
# near the target.
xdotool mousemove 5 5
echo "clickhit-setup-ok tx=$tx ty=$ty W=$W H=$H"
"""


def _setup_sh() -> str:
    return (
        _SETUP_SH
        .replace("__PROBE_DIR__", PROBE_DIR)
        .replace("__IMAGE_PATH__", IMAGE_PATH)
        .replace("__TRUTH_PATH__", TRUTH_PATH)
        .replace("__PCTX__", str(TARGET_PCT_X))
        .replace("__PCTY__", str(TARGET_PCT_Y))
        .replace("__TW__", str(TARGET_W))
        .replace("__TH__", str(TARGET_H))
    )


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Paint the target wallpaper, stash the truth, park the cursor."""
    r = await session.run_command(_setup_sh(), check=False)
    if r["return_code"] != 0:
        raise RuntimeError(
            f"[clickhit] setup failed: "
            f"{(r.get('stderr') or r.get('stdout') or '')[:400]}"
        )
    # Set the painted image as the wallpaper on whichever desktop is live
    # (GNOME on the VM, XFCE in the docker container).
    await set_desktop_wallpaper(session, IMAGE_PATH)
    logger.info("[clickhit] target painted onto wallpaper; cursor parked")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Pass iff the resting pointer is within TOLERANCE_PX of the target."""
    try:
        truth = json.loads(await session.read_file(TRUTH_PATH))
        tx, ty = int(truth["x"]), int(truth["y"])
    except Exception as exc:
        logger.info("[clickhit] truth unreadable (setup failed?): %s", exc)
        return [0.0]

    # The X pointer rests wherever the last click landed.
    r = await session.run_command(
        'DISPLAY="${DISPLAY:-:0}" xdotool getmouselocation --shell', check=False
    )
    out = r.get("stdout") or ""
    coords = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, val = line.partition("=")
            coords[k.strip()] = val.strip()
    try:
        x, y = int(coords["X"]), int(coords["Y"])
    except (KeyError, ValueError):
        logger.info("[clickhit] FAIL — could not read pointer (%r)", out[:120])
        return [0.0]

    d = math.hypot(x - tx, y - ty)
    if d <= TOLERANCE_PX:
        logger.info(
            "[clickhit] PASS — resting pointer (%d,%d) is %.0fpx from target "
            "(%d,%d), tol=%d", x, y, d, tx, ty, TOLERANCE_PX)
        return [1.0]
    logger.info(
        "[clickhit] FAIL — resting pointer (%d,%d) is %.0fpx from target (%d,%d), "
        "tol=%d (a large miss near the model's raw normalized coords indicates "
        "the harness did not rescale to pixels)", x, y, d, tx, ty, TOLERANCE_PX)
    return [0.0]
