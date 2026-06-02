"""Ubuntu-native RGI MCR-1 contig benchmark (v2 submission)."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_output import ScoreResult, score_output_payloads  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "rgi_mcr1_colistin_v2"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
VISIBLE_TASK_NAME = "amr_contig_annotation_instance_1"
CANONICAL_OUTPUT_DIR_NAMES = {
    "output",
    "output_test_pos",
    "output_test_neg",
    "output_admin_pos",
    "output_admin_neg",
}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in CANONICAL_OUTPUT_DIR_NAMES:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must normalize to one of: output, output_test_pos, output_test_neg"
        )
    return normalized


def _decode_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


@dataclass
class RgiMcr1ColistinV2Config(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def data_task_dir(self) -> str:
        return f"{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def task_dir(self) -> str:
        return f"{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{VISIBLE_TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def reference_dir(self) -> str:
        return f"{self.data_task_dir}/reference"

    @property
    def eval_dir(self) -> str:
        return f"{self.data_task_dir}/eval_data"

    @property
    def output_dir_name(self) -> str:
        return _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        if self.output_dir_name == "output":
            return f"{self.task_dir}/{self.output_dir_name}"
        return f"{self.data_task_dir}/{self.output_dir_name}"

    @property
    def input_fasta(self) -> str:
        return f"{self.input_dir}/input_contig.fasta"

    @property
    def card_json(self) -> str:
        return f"{self.input_dir}/card.json"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def answer_file(self) -> str:
        return f"{self.remote_output_dir}/answer.json"

    @property
    def rgi_result_tsv(self) -> str:
        return f"{self.remote_output_dir}/rgi_result.txt"

    @property
    def rgi_result_json(self) -> str:
        return f"{self.remote_output_dir}/rgi_result.json"

    @property
    def verification_targets_file(self) -> str:
        return f"{self.data_task_dir}/reference/metadata/verification_targets.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to annotate one plasmid-mediated antimicrobial-resistance gene from a staged DNA contig with CARD RGI.

Visible task workspace:
- `{self.task_dir}`

Visible inputs:
- Contig FASTA: `{self.input_fasta}`
- CARD database JSON: `{self.card_json}`
- Runtime manifest for installing the official GitHub RGI package: `{self.runtime_pyproject}`

Required workflow:
1. Work from a writable directory under `{self.remote_output_dir}`.
2. Create a task-local Python environment from `{self.runtime_env_dir}` so the `rgi` CLI is available.
3. Use the staged `{self.card_json}` with `rgi load --local -i ../input/card.json`.
4. Run RGI in contig mode against `../input/input_contig.fasta` with:
   `rgi main -i ../input/input_contig.fasta -o rgi_result -t contig --local --clean -g PYRODIGAL`
5. Parse the first data row of `rgi_result.txt`.
6. Write exactly one UTF-8 JSON file to `{self.answer_file}` with these keys:
   - `best_hit_aro`
   - `percent_identity`
   - `drug_class`
   - `resistance_mechanism`

Output requirements:
- `best_hit_aro` should be the best-hit ARO gene name string from the RGI TSV.
- `percent_identity` should be numeric.
- `drug_class` should report CARD's drug class wording for the hit.
- `resistance_mechanism` should report CARD's resistance mechanism wording for the hit.
- Keep solver-created files under `{self.remote_output_dir}`.
- Treat `{self.input_dir}` as read-only source data.

Important constraints:
- Use the provided `{self.card_json}` instead of downloading a different CARD version.
- Use contig mode, not protein mode.
- Report the specific best-hit ARO gene name from the RGI TSV rather than a generic enzyme-family label.
- Keep hidden evaluator-owned directories out of your workflow.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "task_id": TASK_ID,
                "workspace_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "output_dir_name": self.output_dir_name,
                "input_fasta": self.input_fasta,
                "card_json": self.card_json,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "answer_file": self.answer_file,
                "rgi_result_tsv": self.rgi_result_tsv,
                "rgi_result_json": self.rgi_result_json,
                "verification_targets_file": self.verification_targets_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = RgiMcr1ColistinV2Config()


@cb.tasks_config(split="train")
def load():
    cfg = RgiMcr1ColistinV2Config(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    if not (await session.file_exists(meta["answer_file"]) or await session.directory_exists(meta["answer_file"])):
        logger.error("agent missing output: %s", meta["answer_file"])
        return [0.0]

    if not (await session.file_exists(meta["verification_targets_file"]) or await session.directory_exists(meta["verification_targets_file"])):
        raise RuntimeError(
            f"evaluator-controlled reference missing: {meta['verification_targets_file']}"
        )

    answer_text = _decode_text(await session.read_bytes(meta["answer_file"]))
    reference_text = _decode_text(await session.read_bytes(meta["verification_targets_file"]))
    result: ScoreResult = score_output_payloads(
        output_json_text=answer_text,
        reference_json_text=reference_text,
    )

    logger.info(
        "[%s] score=%.4f gene=%.4f identity=%.4f drug_class=%.4f mechanism=%.4f valid=%s reason=%s",
        meta["variant_name"],
        result.score,
        result.gene_score,
        result.identity_score,
        result.drug_class_score,
        result.resistance_mechanism_score,
        result.valid,
        result.reason,
    )
    logger.info("[%s] scoring_details=%s", meta["variant_name"], json.dumps(result.to_dict()))
    return [result.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
