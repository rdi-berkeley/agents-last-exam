"""Stage 2 task implementation for computing_math/paper_reproduction_instance_1."""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import score_outputs  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "paper_reproduction_instance_1"
VARIANT_NAME = "base"

OUTPUT_FILENAME = "results.json"
GOLD_FILENAME = "gold_table2.json"

EXPECTED_DATASET_TOKENS = (
    "imagenet-v2",
    "imagenet-s",
    "imagenet-r",
    "imagenet-a",
    "objectnet",
)

DATASET_PROBE_NAMES = {
    "imagenet-v2": ("imagenetv2", "imagenet-v2", "imagenet_v2"),
    "imagenet-s": ("imagenet-sketch", "imagenet_sketch", "imagenet-s", "sketch"),
    "imagenet-r": ("imagenet-r", "imagenet_r", "imagenet-rendition"),
    "imagenet-a": ("imagenet-a", "imagenet_a", "imagenet-adversarial"),
    "objectnet": ("objectnet-1.0", "objectnet"),
}

DATASET_SEARCH_ROOTS = (
    "/media/user/data/agenthle/computing_math/paper_reproduction_instance_1/base",
    "/media/user/data",
    "/home/user",
    "/tmp",
)

DATASET_MIN_FILES = 100  # a loadable validation split has more than a handful of images


@dataclass
class PaperReproductionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    def __init__(self, *, REMOTE_OUTPUT_DIR: str = ""):
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR
            or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/{OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/{GOLD_FILENAME}"

    @property
    def task_description(self) -> str:
        return f"""\
You are reproducing Table 2 of the ICML 2024 paper \
"LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class \
Taxonomies" (arXiv:2407.16067).

## Inputs (under `{self.input_dir}`)

- `paper.pdf` - local copy of the paper; do not re-fetch from arXiv.
- `codebase.zip` - snapshot of the official repo
  `github.com/ElvishElvis/LCA-on-the-line` at commit 8c0e74d. The archive ships
  precomputed per-model statistics (`ICML_dict_result_metric_dict`,
  `ICML_dict_agreement_dict_*.npy`) that encode Top-1 / Top-5 / LCA-distance
  numbers for 75 pretrained models across the 6 evaluation sets used by the
  paper. You do NOT need to re-run the full inference pipeline on raw images
  to reproduce Table 2 - the shipped `main.py` / `draw_plot.py` already
  consume those cached artifacts.
- `README.md` - short orientation note.

## Software

- `{self.software_dir}/python` is a stable Python 3 entry point on the VM.
- The VM is CPU-only. Running 75 pretrained models over five full OOD
  validation splits is infeasible here; the precomputed-artifact path in the
  codebase is the intended CPU route.
- You are responsible for any additional Python packages (e.g. `torch`,
  `timm`, `open_clip_torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`,
  `statsmodels`) that the codebase imports at runtime.

## Your Task

1. Read `paper.pdf` and inspect `codebase.zip` to confirm that Table 2 is the
   correlation table you are asked to reproduce.
2. Unpack the codebase and run (or adapt) its correlation pipeline. The
   shipped `ICML_dict_result_metric_dict` plus `ICML_dict_agreement_dict_*.npy`
   are sufficient to regenerate all 40 Table 2 correlation cells without
   downloading the OOD validation images themselves.
3. If you do choose to download any of the five ImageNet-OOD datasets listed
   in the paper (ImageNet-V2, ImageNet-Sketch, ImageNet-R, ImageNet-A,
   ObjectNet), put them on this VM and list what you actually downloaded in
   `datasets_downloaded`. Claiming a dataset you did not download will not
   earn credit - the evaluator will look for it on disk.
4. Write `{self.output_file}` with the schema below.

## Output Schema

```json
{{
  "identified_key_table": "Table 2",
  "datasets_downloaded": ["imagenet-v2", "imagenet-s", "imagenet-r", "imagenet-a", "objectnet"],
  "table2_values": {{
    "<ID>_<OOD>_<dataset>_<metric>": <float>,
    ...
  }}
}}
```

- `identified_key_table` must be exactly `"Table 2"`.
- `datasets_downloaded` must be a JSON array using the tokens
  `imagenet-v2`, `imagenet-s`, `imagenet-r`, `imagenet-a`, `objectnet` (lower
  case, hyphenated). List only the OOD splits you actually downloaded.
- `table2_values` holds the 4 row-pairs x 5 datasets x 2 metrics = 40 cells
  of Table 2. Each key follows
  `<ID>_<OOD>_<dataset>_<metric>` where:
    - ID  in {{`Top1`, `LCA`}} - the in-distribution (ImageNet) indicator.
    - OOD in {{`Top1`, `Top5`}} - the out-of-distribution accuracy target.
    - dataset in {{`ImgNv2`, `ImgNS`, `ImgNR`, `ImgNA`, `ObjN`}}.
    - metric in {{`R2`, `PEA`}} - coefficient of determination and Pearson
      correlation. Per the paper's Table 2 caption, report the absolute value
      of Pearson.
  Example: `Top1_Top1_ImgNv2_R2 = 0.962`, `LCA_Top1_ObjN_PEA = 0.956`.

## Rules

- Write only under `{self.remote_output_dir}/`.
- Do not modify anything under `{self.input_dir}/`.
- `results.json` must be valid UTF-8 JSON.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "expected_dataset_tokens": list(EXPECTED_DATASET_TOKENS),
            }
        )
        return metadata


config = PaperReproductionConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _run_command(
    session: cb.DesktopSession, command: str, *, check: bool = False
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_bytes_safe(session: cb.DesktopSession, path: str) -> bytes | None:
    try:
        data = await session.read_bytes(path)
    except Exception as exc:
        logger.debug("read_bytes failed for %s: %s", path, exc)
        return None
    return data or None


async def _verify_dataset_on_vm(session: cb.DesktopSession, token: str) -> bool:
    """Return True iff a directory clearly holding <token>'s validation split
    is present on the VM with at least DATASET_MIN_FILES image files."""
    candidate_names = DATASET_PROBE_NAMES.get(token, (token,))
    for root in DATASET_SEARCH_ROOTS:
        names_expr = " -o ".join(f'-iname "*{n}*"' for n in candidate_names)
        # First locate candidate directories.
        find_dirs = (
            f'find "{root}" -maxdepth 6 -type d \\( {names_expr} \\) '
            f"2>/dev/null | head -n 5"
        )
        result = await _run_command(session, find_dirs)
        stdout = (result.get("stdout") or "").strip()
        if not stdout:
            continue
        for candidate in stdout.splitlines():
            candidate = candidate.strip()
            if not candidate:
                continue
            count_cmd = (
                f'find "{candidate}" -maxdepth 4 -type f '
                f'\\( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" '
                f'-o -iname "*.webp" \\) 2>/dev/null | head -n {DATASET_MIN_FILES + 1} | wc -l'
            )
            count_result = await _run_command(session, count_cmd)
            try:
                n = int((count_result.get("stdout") or "0").strip())
            except ValueError:
                n = 0
            if n >= DATASET_MIN_FILES:
                return True
    return False


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]

    agent_bytes = await _read_bytes_safe(session, output_file)
    if agent_bytes is None:
        logger.warning("agent output missing: %s", output_file)
        return [0.0]

    gold_bytes = await _read_bytes_safe(session, reference_file)
    if gold_bytes is None:
        raise RuntimeError(
            f"hidden gold reference not found at {reference_file}"
        )
    gold = json.loads(gold_bytes)
    gold_cells = gold.get("table2_values", {})
    if not gold_cells:
        raise RuntimeError("gold_table2.json has no table2_values cells")

    # Probe the VM for each expected dataset. Limited to claimed tokens so we
    # never waste time probing datasets the agent did not claim to download.
    try:
        claimed = json.loads(agent_bytes).get("datasets_downloaded", [])
    except Exception:
        claimed = []
    verified_tokens: list[str] = []
    if isinstance(claimed, list):
        for raw in claimed:
            if not isinstance(raw, str):
                continue
            token = raw.strip().lower()
            if token not in EXPECTED_DATASET_TOKENS:
                continue
            if await _verify_dataset_on_vm(session, token):
                verified_tokens.append(token)

    result = score_outputs.score(agent_bytes, gold_cells, verified_tokens)
    logger.info(
        "paper_reproduction_instance_1 final=%.3f rule1=%.2f rule2=%.2f rule3=%.3f cells=%d/%d reason=%s",
        result.score,
        result.rule1,
        result.rule2,
        result.rule3,
        result.cells_matched,
        result.cells_total,
        result.reason,
    )
    return [float(result.score)]
