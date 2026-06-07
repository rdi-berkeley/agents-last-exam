"""Demo task: ``demo/seecheck`` — does the screenshot image reach the model?

A deliberately tiny GUI task that isolates the vision bridge. ``setup`` renders
a unique ``SCREEN CODE`` (e.g. ``SCRN-7QX42K``) full-screen onto the desktop
wallpaper. The code is never written to any input file — it exists ONLY as
pixels on screen. The agent must:

  1. take a single screenshot,
  2. read the SCREEN CODE off the screen,
  3. write it (exact) to ``output/result.txt``.

If the agent's harness drops screenshot images before they reach the model
(e.g. the gemini-cli OpenRouter converter not emitting ``image_url``), the model
is blind and writes ``NO_IMAGE`` or a hallucinated string → score 0.0. When the
image is forwarded, the model reads the code → score 1.0. This is the smallest
end-to-end probe of the desktop GUI → model image path.

Self-contained: it stages no GCS data and writes no input/reference. setup()
generates the code, paints it, and stashes the expected value at
``EXPECTED_PATH`` for evaluate() to compare against.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "demo"
TASK_NAME = "seecheck"
VARIANT_NAME = "base"

# Where setup() stashes the ground-truth code for evaluate(). Not referenced by
# the prompt; the code is meant to be read off the screen, not the filesystem.
EXPECTED_PATH = "/home/user/.seecheck_expected"
IMAGE_PATH = "/tmp/seecheck_screen.png"
# Unambiguous alphabet (no O/0/I/1) so OCR vs ground-truth is clean.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def result_path(self) -> str:
        return f"{self.remote_output_dir}/result.txt"

    @property
    def task_description(self) -> str:
        return (
            "Vision check — a GUI task. The desktop shows a large code on a "
            "blue background, formatted as 'SCREEN CODE:' followed by a value "
            "like CODE-XXXXXX.\n\n"
            "1. Take exactly ONE screenshot of the desktop.\n"
            "2. Read the SCREEN CODE shown on screen.\n"
            f"3. Write that code, exactly, on a single line, to "
            f"{self.result_path}.\n\n"
            "The code is only visible on the screen — it is not in any file. "
            "If you receive no image / cannot see the screen at all, write "
            "exactly NO_IMAGE to that file instead."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "result_path": self.result_path,
            "expected_path": EXPECTED_PATH,
            "image_path": IMAGE_PATH,
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


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Paint a fresh SCREEN CODE onto the wallpaper; stash the truth."""
    meta = task_cfg.metadata
    out_dir = meta["remote_output_dir"]
    await session.run_command(f"mkdir -p {out_dir!r}", check=False)
    await session.run_command(f"rm -f {meta['result_path']!r}", check=False)

    code = "CODE-" + "".join(secrets.choice(_ALPHABET) for _ in range(6))

    # Render the code full-screen, then set it as the GNOME wallpaper. A
    # wallpaper needs no long-lived process (the cua-server reaps children of
    # run_command), so it reliably survives until the agent screenshots.
    text = f"SCREEN CODE:\\n{code}"
    render = (
        f"convert -size 1920x1080 xc:'#0b3d91' -gravity center -fill white "
        f"-pointsize 130 -annotate +0+0 '{text}' {IMAGE_PATH}"
    )
    r = await session.run_command(render, check=False)
    if r["return_code"] != 0:
        raise RuntimeError(f"[seecheck] render failed: {(r.get('stderr') or '')[:300]}")

    set_wp = (
        'U=$(id -u user); export DISPLAY=:0 '
        'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$U/bus; '
        'gsettings set org.gnome.desktop.background picture-options scaled; '
        f'gsettings set org.gnome.desktop.background picture-uri file://{IMAGE_PATH}; '
        f'gsettings set org.gnome.desktop.background picture-uri-dark file://{IMAGE_PATH}'
    )
    r = await session.run_command(set_wp, check=False)
    if r["return_code"] != 0:
        raise RuntimeError(f"[seecheck] set wallpaper failed: {(r.get('stderr') or '')[:300]}")

    await session.run_command(
        f"printf %s {code!r} > {EXPECTED_PATH!r}", check=False,
    )
    logger.info("[seecheck] painted SCREEN CODE and stashed expected value")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Exact-match the code the agent transcribed from the screen."""
    meta = task_cfg.metadata
    try:
        expected = (await session.read_file(meta["expected_path"])).strip()
    except Exception as exc:
        raise RuntimeError(f"[seecheck] expected unreadable: {exc}")

    try:
        actual = (await session.read_file(meta["result_path"])).strip()
    except Exception as exc:
        logger.info("[seecheck] output unreadable at %s: %s", meta["result_path"], exc)
        return [0.0]

    if actual.upper() == expected.upper():
        logger.info("[seecheck] PASS — model read the screen code %r", expected)
        return [1.0]
    logger.info("[seecheck] FAIL — got %r, expected %r (NO_IMAGE => image never "
                "reached the model)", actual[:80], expected)
    return [0.0]
