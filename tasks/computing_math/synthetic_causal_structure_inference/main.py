"""AgentHLE task: computing_math/synthetic_causal_structure_inference."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import evaluate_submission_json

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "synthetic_causal_structure_inference"
VARIANT_NAME = "base"
OUTPUT_FILENAME = "submission.json"


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def benchmark_prompt_file(self) -> str:
        return f"{self.input_dir}/AGENT_PROMPT_FOR_SYNTHETIC_CAUSAL_TASK.md"

    @property
    def manifest_file(self) -> str:
        return f"{self.input_dir}/manifest.json"

    @property
    def public_dataset_dir(self) -> str:
        return f"{self.input_dir}/public"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lockfile(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_causal_env.sh"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/{OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/gold_standard_submission.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to solve a synthetic causal-structure benchmark.

## Input Files
- Task brief: `{self.task_brief_file}`
- Benchmark prompt: `{self.benchmark_prompt_file}`
- Sanitized dataset manifest: `{self.manifest_file}`
- Dataset directories: `{self.public_dataset_dir}/dataset_###/`
- Optional runtime manifest: `{self.runtime_env_dir}`
- Optional Python wrapper: `{self.python_wrapper}`

## Optional Python Setup
If you want the staged scientific Python environment, use the wrapper:

```bash
"{self.python_wrapper}" your_script.py
```

That wrapper keeps any created runtime environment under `{self.output_dir}`.

## Your Task
1. Read `{self.task_brief_file}` and `{self.benchmark_prompt_file}`.
2. Read `{self.manifest_file}` to get the full sanitized dataset list.
3. For each dataset directory under `{self.public_dataset_dir}`, infer:
   - `scenario`
   - `identification_strategy`
   - `identifiable_effect`
   - `variable_roles`
   - `directed_edges`
   - `latent_confounders`
4. Save one final JSON file at `{self.output_file}`.

## Output Requirements
- The file must be valid JSON.
- Top-level keys must be `benchmark_name` and `predictions`.
- `benchmark_name` must match the staged benchmark value shown in `{self.manifest_file}`.
- Include exactly one prediction object for every dataset listed in `{self.manifest_file}`.
- Use the sanitized dataset ids from the manifest exactly as written.
- Do not modify files under `{self.input_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_brief_file": self.task_brief_file,
                "benchmark_prompt_file": self.benchmark_prompt_file,
                "manifest_file": self.manifest_file,
                "public_dataset_dir": self.public_dataset_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lockfile": self.runtime_lockfile,
                "python_wrapper": self.python_wrapper,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "canonical_gcs_root": (
                    "gs://ale-data-all/computing_math/"
                    "synthetic_causal_structure_inference/base"
                ),
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    try:
        output_text = await session.read_file(meta["output_file"])
    except Exception as exc:
        logger.error("Failed to read agent output %s: %s", meta["output_file"], exc)
        return [0.0]

    try:
        reference_text = await session.read_file(meta["reference_file"])
    except Exception as exc:
        logger.error("Failed to read hidden reference %s: %s", meta["reference_file"], exc)
        return [0.0]

    try:
        output_json = json.loads(
            output_text.decode("utf-8") if isinstance(output_text, bytes) else str(output_text)
        )
        reference_json = json.loads(
            reference_text.decode("utf-8")
            if isinstance(reference_text, bytes)
            else str(reference_text)
        )
    except Exception as exc:
        logger.error("Failed to parse output/reference JSON: %s", exc)
        return [0.0]

    try:
        report = evaluate_submission_json(output_json, reference_json)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return [0.0]

    logger.info("evaluation_report=%s", json.dumps(report["aggregate"], sort_keys=True))
    return [float(report["aggregate"]["overall_score"])]
