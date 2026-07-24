"""Demo task: ``demo/clickhit_win`` (Windows) — does the agent click where it means to?

Pointer-accuracy counterpart of ``demo/seecheck``. Where ``seecheck`` isolates
the *vision* path (are screenshot pixels reaching the model?), this isolates the
*click* path (do the model's coordinates land on the pixel it aimed at?).

``setup``:
  1. Detects the live screen size and computes a single **off-center** target
     (default 82% width, 83% height — deliberately far from center, where a
     coordinate-space bug diverges most).
  2. Paints that target — a red box with a white crosshair at its exact center,
     plus instructions — onto the desktop **wallpaper** (the same channel
     ``seecheck`` uses, which the harness reliably screenshots), minimizes
     windows and hides icons so the target is unobstructed.
  3. Starts a background **global low-level mouse hook** (``WH_MOUSE_LL``) that
     appends the screen-pixel coordinate of every left click to ``clicks.log``.
     The hook records real input events, so it captures wherever a click
     actually lands — no on-desktop window of our own is required (window
     rendering on a detached Windows console is unreliable; input hooking is not).

The agent must take a screenshot and click the crosshair at the center of the
box. ``evaluate`` passes iff a recorded click landed within ``TOLERANCE_PX`` of
the target center.

Why this exposes the Gemini coordinate bug: Gemini grounds clicks to a fixed
``[0, 1000)`` grid, not screen pixels. For a target at pixel ``(0.82W, 0.83H)``
the model emits ``~(820, 830)``. If the harness forwards that unscaled, the
click lands at pixel ``(820, 830)`` — hundreds of pixels from the target on any
screen wider/taller than 1000 px — and the probe scores 0.0. With normalized→
pixel rescaling (this branch's fix), it maps to ``(0.82W, 0.83H)`` and scores 1.0.
The same-branch A/B is ``coordinate_space: pixel`` (reproduces the miss) vs
``normalized``/``null`` (the fix).

Self-contained (``REQUIRES_TASK_DATA = False``): stages no data. The target
coordinate is chosen at ``setup`` from the live resolution and written to
``truth.json`` (read back by ``evaluate``); it is intentionally NOT in the prompt,
so the only way to hit it is to look at the screen and click. (The probe measures
a cooperating model's real clicks; it does not defend the click log against an
agent that forges it — acceptable for a harness self-test.)
"""
from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "demo"
TASK_NAME = "clickhit_win"
VARIANT_NAME = "base"

# Everything lives under one throwaway dir, outside the agent's task/output tree
# and unmentioned in the prompt.
PROBE_DIR = r"C:\clickprobe"
LOG_PATH = r"C:\clickprobe\clicks.log"
TRUTH_PATH = r"C:\clickprobe\truth.json"
HOOK_PATH = r"C:\clickprobe\hook.ps1"
RENDER_PATH = r"C:\clickprobe\render.ps1"

# Target placement as a fraction of the screen. Far from center so the
# normalized-vs-pixel divergence is large and unambiguous.
TARGET_FRAC_X = 0.82
TARGET_FRAC_Y = 0.83
TARGET_W = 200
TARGET_H = 140
# A hit is a click within this many pixels of the target center. Generous
# relative to the box, but tiny next to the coordinate-bug miss (which is
# ``frac * (dim - 1000)`` px — e.g. ~230px on 1280x1024, ~750px on 1920x1080).
TOLERANCE_PX = 70


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"
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
            "anywhere else first. You do not need to write any file."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "probe_dir": PROBE_DIR,
            "log_path": LOG_PATH,
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


def _ps_encoded(script: str) -> str:
    """Wrap a PowerShell script as ``powershell -EncodedCommand <b64>``.

    EncodedCommand takes UTF-16LE base64 — this sidesteps all cmd/PowerShell
    quoting hazards for the multi-line scripts run over cua run_command.
    """
    b64 = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {b64}"


# ---------------------------------------------------------------------------
# The background click recorder: a WH_MOUSE_LL global mouse hook.
#
# A low-level mouse hook fires for every mouse event on the session's input
# desktop and reports absolute screen coordinates, so it records exactly where
# a click landed regardless of which window (if any) is under the cursor. It
# needs a running message loop, so it lives in its own detached process; the
# hardcoded log path keeps this script substitution-free.
# ---------------------------------------------------------------------------
_HOOK_PS = r"""
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path 'C:\clickprobe' | Out-Null
Add-Type @"
using System;
using System.IO;
using System.Runtime.InteropServices;
public class LLHook {
  const int WH_MOUSE_LL = 14, WM_LBUTTONDOWN = 0x0201;
  static IntPtr hook = IntPtr.Zero;
  static string logPath;
  public delegate IntPtr Proc(int code, IntPtr w, IntPtr l);
  static Proc proc;
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int x; public int y; }
  [StructLayout(LayoutKind.Sequential)] public struct MSLLHOOKSTRUCT { public POINT pt; public uint mouseData; public uint flags; public uint time; public IntPtr extra; }
  [StructLayout(LayoutKind.Sequential)] public struct MSG { public IntPtr hwnd; public uint message; public IntPtr wParam; public IntPtr lParam; public uint time; public POINT pt; }
  [DllImport("user32.dll")] static extern IntPtr SetWindowsHookEx(int id, Proc cb, IntPtr mod, uint tid);
  [DllImport("user32.dll")] static extern IntPtr CallNextHookEx(IntPtr h, int code, IntPtr w, IntPtr l);
  [DllImport("kernel32.dll")] static extern IntPtr GetModuleHandle(string n);
  [DllImport("user32.dll")] static extern int GetMessage(out MSG msg, IntPtr hwnd, uint mn, uint mx);
  static IntPtr HookProc(int code, IntPtr w, IntPtr l) {
    if (code >= 0 && (int)w == WM_LBUTTONDOWN) {
      MSLLHOOKSTRUCT m = (MSLLHOOKSTRUCT)Marshal.PtrToStructure(l, typeof(MSLLHOOKSTRUCT));
      try { File.AppendAllText(logPath, m.pt.x + "," + m.pt.y + Environment.NewLine); } catch {}
    }
    return CallNextHookEx(hook, code, w, l);
  }
  public static void Run(string path) {
    logPath = path; proc = new Proc(HookProc);
    hook = SetWindowsHookEx(WH_MOUSE_LL, proc, GetModuleHandle(null), 0);
    MSG msg; while (GetMessage(out msg, IntPtr.Zero, 0, 0) > 0) {}
  }
}
"@
[LLHook]::Run('C:\clickprobe\clicks.log')
"""


def _render_and_start_ps() -> str:
    """PowerShell that paints the target wallpaper, records the truth, and
    starts the detached mouse-hook recorder."""
    script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Sys {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int SystemParametersInfo(int a, int u, string p, int f);
}
"@
[Sys]::SetProcessDPIAware() | Out-Null

$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$W = $b.Width; $H = $b.Height
$tx = [int]($W * __FX__); $ty = [int]($H * __FY__)
$tw = __TW__; $th = __TH__

# Ground truth for evaluate().
$dir = 'C:\clickprobe'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Set-Content -Path 'C:\clickprobe\truth.json' -Value ("{""x"":$tx,""y"":$ty,""w"":$W,""h"":$H,""tw"":$tw,""th"":$th}")

# Paint the target onto a full-screen bitmap and set it as the wallpaper.
$bmp = New-Object System.Drawing.Bitmap($W, $H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAlias
$g.Clear([System.Drawing.Color]::FromArgb(20,20,28))

$title = New-Object System.Drawing.Font('Arial',30,[System.Drawing.FontStyle]::Bold)
$body  = New-Object System.Drawing.Font('Arial',20,[System.Drawing.FontStyle]::Regular)
$g.DrawString('POINTER ACCURACY CHECK', $title, [System.Drawing.Brushes]::White, 60, 60)
$g.DrawString('Click the WHITE CROSSHAIR at the center of the RED box.', $body, [System.Drawing.Brushes]::Gainsboro, 60, 120)

$rx = $tx - [int]($tw/2); $ry = $ty - [int]($th/2)
$rect = New-Object System.Drawing.Rectangle($rx, $ry, $tw, $th)
$g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Crimson)), $rect)
$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::White, 3)
$g.DrawRectangle($pen, $rect)
# Crosshair at the exact target center.
$g.DrawLine($pen, ($tx-20), $ty, ($tx+20), $ty)
$g.DrawLine($pen, $tx, ($ty-20), $tx, ($ty+20))
$lbl = New-Object System.Drawing.Font('Arial',16,[System.Drawing.FontStyle]::Bold)
$g.DrawString('TARGET', $lbl, [System.Drawing.Brushes]::White, ($rx), ($ry-30))
$g.Dispose()

$img = 'C:\clickprobe\target.bmp'
$bmp.Save($img, [System.Drawing.Imaging.ImageFormat]::Bmp)
$bmp.Dispose()
# SPI_SETDESKWALLPAPER=20, SPIF_UPDATEINIFILE|SPIF_SENDWININICHANGE=3
[Sys]::SystemParametersInfo(20, 0, $img, 3) | Out-Null

# Clear the desktop so the wallpaper target is unobstructed (mirrors seecheck).
Set-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced' -Name HideIcons -Value 1 -ErrorAction SilentlyContinue
try { Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue } catch {}
Start-Sleep -Seconds 3
try { (New-Object -ComObject Shell.Application).MinimizeAll() } catch {}

# Start the detached global click recorder (survives this command's exit).
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','C:\clickprobe\hook.ps1'
"""
    return (
        script
        .replace("__FX__", repr(TARGET_FRAC_X))
        .replace("__FY__", repr(TARGET_FRAC_Y))
        .replace("__TW__", str(TARGET_W))
        .replace("__TH__", str(TARGET_H))
    )


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Paint the target wallpaper, stash the truth, start the click recorder.

    The render script is large (it embeds an inline C# type), so it is staged
    as a file and run via ``powershell -File`` — ``-EncodedCommand`` would blow
    past cmd.exe's ~8 KB command-line limit.
    """
    # Fresh probe dir: kill any recorder from a prior run, then recreate clean.
    await session.run_command(_ps_encoded(
        "Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" | "
        "Where-Object { $_.CommandLine -match 'clickprobe' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; "
        "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue 'C:\\clickprobe'; "
        "New-Item -ItemType Directory -Force -Path 'C:\\clickprobe' | Out-Null"
    ), check=False)

    # Stage the recorder + renderer as files, then run the renderer (which
    # paints the wallpaper, writes truth.json, and launches the recorder).
    await session.write_file(HOOK_PATH, _HOOK_PS)
    await session.write_file(RENDER_PATH, _render_and_start_ps())
    r = await session.run_command(
        f"powershell -NoProfile -ExecutionPolicy Bypass -File {RENDER_PATH}",
        check=False,
    )
    if r["return_code"] != 0:
        raise RuntimeError(
            f"[clickhit_win] setup failed: "
            f"{(r.get('stderr') or r.get('stdout') or '')[:400]}"
        )
    logger.info("[clickhit_win] target painted + click recorder started")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Pass iff a recorded click landed within TOLERANCE_PX of the target."""
    try:
        truth = json.loads(await session.read_file(TRUTH_PATH))
        tx, ty = int(truth["x"]), int(truth["y"])
    except Exception as exc:
        logger.info("[clickhit_win] truth unreadable (setup failed?): %s", exc)
        return [0.0]

    try:
        raw = await session.read_file(LOG_PATH)
    except Exception:
        logger.info("[clickhit_win] FAIL — no clicks recorded (clicks.log absent)")
        return [0.0]

    best = None
    best_xy = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            xs, ys = line.split(",")[:2]
            x, y = int(float(xs)), int(float(ys))
        except (ValueError, IndexError):
            continue
        d = math.hypot(x - tx, y - ty)
        if best is None or d < best:
            best, best_xy = d, (x, y)

    if best is None:
        logger.info("[clickhit_win] FAIL — clicks.log had no parseable clicks")
        return [0.0]

    if best <= TOLERANCE_PX:
        logger.info(
            "[clickhit_win] PASS — closest click %s is %.0fpx from target "
            "(%d,%d), tol=%d", best_xy, best, tx, ty, TOLERANCE_PX)
        return [1.0]
    logger.info(
        "[clickhit_win] FAIL — closest click %s is %.0fpx from target (%d,%d), "
        "tol=%d (a large miss near the model's raw normalized coords indicates "
        "the harness did not rescale to pixels)", best_xy, best, tx, ty, TOLERANCE_PX)
    return [0.0]
