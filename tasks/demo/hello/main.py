"""Demo task: ``demo/hello`` (Linux) — a GUI-centric end-to-end demo.

This is the canonical "does the whole stack work" task. Unlike a trivial
file-write, it forces the agent to drive a real desktop GUI:

  * Task data ships with the image / GCS bucket (input/ + software/), staged by
    the framework before the agent runs — this task does NOT generate it.
  * input/app.html is a self-contained multi-step "Acme Order Entry" web app.
  * input/order.json holds the data the agent must type into the app.
  * software/launch.sh is a shortcut that opens the app in Chrome.
  * The app computes a confirmation code from the values entered IN THE GUI;
    the code is not present in any input file, so the agent must complete the
    wizard with screenshots + clicks + typing.
  * The agent writes the confirmation code to output/result.txt.
  * reference/result.txt (decrypted from reference.7z at eval time) holds the
    expected code; evaluate() compares exactly.

Windows counterpart: ``demo/hello_win`` — identical except OS conventions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "demo"
TASK_NAME = "hello"
VARIANT_NAME = "base"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def app_path(self) -> str:
        return f"{self.input_dir}/app.html"

    @property
    def order_path(self) -> str:
        return f"{self.input_dir}/order.json"

    @property
    def instructions_path(self) -> str:
        return f"{self.input_dir}/instructions.txt"

    @property
    def launcher_path(self) -> str:
        return f"{self.software_dir}/launch.sh"

    @property
    def result_path(self) -> str:
        return f"{self.remote_output_dir}/result.txt"

    @property
    def reference_result_path(self) -> str:
        return f"{self.reference_dir}/result.txt"

    @property
    def task_description(self) -> str:
        return (
            "Acme Code Generator — a GUI task. Use the desktop GUI (screenshot, "
            "click, type) to generate a confirmation code in a web app; the "
            "answer cannot be obtained from files alone.\n\n"
            f"1. Open the app: run `bash {self.launcher_path}` to launch Chrome "
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


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Prepare output; sanity-check that staged input is present.

    Task data (input/ + software/) is staged by the framework before setup, so
    this does NOT write any input/software/reference — it only ensures a clean
    output dir and asserts the app + order data landed.
    """
    meta = task_cfg.metadata
    out_dir = meta["remote_output_dir"]
    await session.run_command(f"mkdir -p {out_dir!r}", check=False)
    await session.run_command(f"rm -f {meta['result_path']!r}", check=False)

    for path in (meta["app_path"], meta["order_path"]):
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(
                f"staged input missing: {path} unreadable ({exc}). "
                "Re-bake the image or check the GCS task data."
            )
    logger.info("[hello] staged input present; output dir ready at %s", out_dir)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Exact-match the confirmation code in output/result.txt vs reference."""
    meta = task_cfg.metadata
    out_path = meta["result_path"]
    ref_path = meta["reference_result_path"]

    try:
        expected = (await session.read_file(ref_path)).strip()
    except Exception as exc:
        # Reference is staged (decrypted) by the framework before evaluate.
        raise RuntimeError(f"reference unreadable at {ref_path}: {exc}")

    try:
        actual = (await session.read_file(out_path)).strip()
    except Exception as exc:
        logger.info("[hello] output unreadable at %s: %s", out_path, exc)
        return [0.0]

    if actual == expected:
        logger.info("[hello] correct confirmation code")
        return [1.0]
    logger.info("[hello] mismatch: got %r, expected %r", actual[:80], expected)
    return [0.0]
