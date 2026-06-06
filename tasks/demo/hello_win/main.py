"""Demo task: ``demo/hello_win`` (Windows) — GUI-centric end-to-end demo.

Windows counterpart of ``demo/hello``; identical logic and data, only OS
conventions differ (E:\\ paths, ``.cmd`` launcher, PowerShell setup).

The agent must drive the desktop GUI:
  * Task data (input/ + software/) ships with the image / GCS bucket and is
    staged by the framework before the agent runs — not generated here.
  * input\\app.html is a self-contained multi-step "Acme Order Entry" web app.
  * input\\order.json holds the data the agent must type into the app.
  * software\\launch.cmd opens the app in Chrome.
  * The app computes a confirmation code from values entered IN THE GUI; the
    code is in no input file, so the agent must complete the wizard with
    screenshots + clicks + typing.
  * The agent writes the confirmation code to output\\result.txt.
  * reference\\result.txt (decrypted from reference.7z at eval) holds the
    expected code; evaluate() compares exactly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "demo"
TASK_NAME = "hello_win"
VARIANT_NAME = "base"


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def app_path(self) -> str:
        return rf"{self.input_dir}\app.html"

    @property
    def order_path(self) -> str:
        return rf"{self.input_dir}\order.json"

    @property
    def instructions_path(self) -> str:
        return rf"{self.input_dir}\instructions.txt"

    @property
    def launcher_path(self) -> str:
        return rf"{self.software_dir}\launch.cmd"

    @property
    def result_path(self) -> str:
        return rf"{self.remote_output_dir}\result.txt"

    @property
    def reference_result_path(self) -> str:
        return rf"{self.reference_dir}\result.txt"

    @property
    def task_description(self) -> str:
        return (
            "Acme Code Generator — a GUI task. Use the desktop GUI (screenshot, "
            "click, type) to generate a confirmation code in a web app; the "
            "answer cannot be obtained from files alone.\n\n"
            f"1. Open the app: run `cmd /c {self.launcher_path}` to launch Chrome "
            f"on it, or open {self.app_path} in Chrome yourself.\n"
            f"2. Read {self.order_path} to get the Order ID.\n"
            "3. In the app: click the \"Order ID\" field, type the Order ID, "
            "then click \"Generate Code\".\n"
            "4. The app displays a confirmation code (CONF-...). Read it from "
            "the screen.\n"
            f"5. Write that code, exactly, on a single line, to "
            f"{self.result_path}.\n\n"
            f"See {self.instructions_path} for the same steps. The code is "
            "computed by the app from the Order ID you type, so you must use "
            "the GUI."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "app_path": self.app_path,
            "order_path": self.order_path,
            "instructions_path": self.instructions_path,
            "launcher_path": self.launcher_path,
            "result_path": self.result_path,
            "reference_result_path": self.reference_result_path,
            "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
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


def _ps_mkdir(path: str) -> str:
    return f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path \'{path}\' | Out-Null"'


def _ps_rm(path: str) -> str:
    return f'powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue \'{path}\'"'


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Prepare output; sanity-check staged input is present (no data gen here)."""
    meta = task_cfg.metadata
    await session.run_command(_ps_mkdir(meta["remote_output_dir"]), check=False)
    await session.run_command(_ps_rm(meta["result_path"]), check=False)

    for path in (meta["app_path"], meta["order_path"]):
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(
                f"staged input missing: {path} unreadable ({exc}). "
                "Re-bake the image or check the GCS task data."
            )
    logger.info("[hello_win] staged input present; output dir ready at %s",
                meta["remote_output_dir"])


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Exact-match the confirmation code in output\\result.txt vs reference."""
    meta = task_cfg.metadata
    out_path = meta["result_path"]
    ref_path = meta["reference_result_path"]

    try:
        expected = (await session.read_file(ref_path)).strip()
    except Exception as exc:
        raise RuntimeError(f"reference unreadable at {ref_path}: {exc}")

    try:
        actual = (await session.read_file(out_path)).strip()
    except Exception as exc:
        logger.info("[hello_win] output unreadable at %s: %s", out_path, exc)
        return [0.0]

    if actual == expected:
        logger.info("[hello_win] correct confirmation code")
        return [1.0]
    logger.info("[hello_win] mismatch: got %r, expected %r", actual[:80], expected)
    return [0.0]
