"""Stage 2 task implementation for ising_post_measurement_1."""

from __future__ import annotations

import json
import logging
import posixpath
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import score_submission_bytes  # noqa: E402
from variant_specs import DOMAIN_NAME, TASK_NAME, VARIANTS, VariantSpec, get_variant  # noqa: E402

logger = logging.getLogger(__name__)


def _output_dir_name(remote_output_dir: str) -> str:
    return posixpath.basename(posixpath.normpath(remote_output_dir))


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANTS[0].variant_name
    VARIANT_LABEL: str = VARIANTS[0].variant_label
    N_QUBITS: int = VARIANTS[0].n_qubits
    ANCILLA_MODE: str = VARIANTS[0].ancilla_mode
    HAS_ANCILLA_STATE: bool = VARIANTS[0].has_ancilla_state
    COUPLING_U: float = VARIANTS[0].coupling_u
    REQUIRES_CORRELATORS: bool = VARIANTS[0].requires_correlators

    @property
    def output_dir_name(self) -> str:
        return _output_dir_name(self.remote_output_dir)

    @property
    def config_file(self) -> str:
        return f"{self.input_dir}/config.json"

    @property
    def task_specification_file(self) -> str:
        return f"{self.input_dir}/task_specification.md"

    @property
    def ancilla_state_file(self) -> str:
        return f"{self.input_dir}/ancilla_state.npy"

    @property
    def requirements_file(self) -> str:
        return f"{self.input_dir}/requirements.txt"

    @property
    def runtime_manifest_file(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def software_readme_file(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def output_files(self) -> dict[str, str]:
        files = {
            "critical_state.npy": f"{self.remote_output_dir}/critical_state.npy",
            "post_probs.npy": f"{self.remote_output_dir}/post_probs.npy",
            "rdm_site1.npy": f"{self.remote_output_dir}/rdm_site1.npy",
        }
        if self.REQUIRES_CORRELATORS:
            files["correlators.npz"] = f"{self.remote_output_dir}/correlators.npz"
        return files

    @property
    def reference_files(self) -> dict[str, str]:
        files = {
            "critical_state.npy": f"{self.reference_dir}/critical_state.npy",
            "post_probs.npy": f"{self.reference_dir}/post_probs.npy",
            "rdm_site1.npy": f"{self.reference_dir}/rdm_site1.npy",
        }
        if self.REQUIRES_CORRELATORS:
            files["correlators.npz"] = f"{self.reference_dir}/correlators.npz"
        return files

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task specification: `{self.task_specification_file}`
- Numerical parameters: `{self.config_file}`
- Optional ancilla state file when staged by the selected variant: `{self.ancilla_state_file}`
- Runtime manifest: `{self.runtime_manifest_file}`
- Requirements fallback: `{self.requirements_file}`
- Runtime notes: `{self.software_readme_file}`

## Your Task
1. Read `{self.task_specification_file}` for the full physics definition, conventions, output contract, and public tolerances.
2. Read `{self.config_file}` for the variant parameters.
3. If `{self.ancilla_state_file}` is present, use it as the staged ancilla state; otherwise use the critical ancilla rule from the spec.
4. If you need Python packages, install them from `input/runtime_env/pyproject.toml` or `input/requirements.txt`.
5. Compute the required outputs and write them under `{self.remote_output_dir}`.

## Required Outputs
- `{self.remote_output_dir}/critical_state.npy`
- `{self.remote_output_dir}/post_probs.npy`
- `{self.remote_output_dir}/rdm_site1.npy`
- If the selected variant requires one-body correlators, also write `{self.remote_output_dir}/correlators.npz` with keys `Z_one_body` and `X_one_body`

Do not modify files under `{self.input_dir}`.
Write final outputs only under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "n_qubits": self.N_QUBITS,
                "ancilla_mode": self.ANCILLA_MODE,
                "has_ancilla_state": self.HAS_ANCILLA_STATE,
                "coupling_u": self.COUPLING_U,
                "requires_correlators": self.REQUIRES_CORRELATORS,
                "config_file": self.config_file,
                "task_specification_file": self.task_specification_file,
                "ancilla_state_file": self.ancilla_state_file,
                "requirements_file": self.requirements_file,
                "runtime_manifest_file": self.runtime_manifest_file,
                "software_readme_file": self.software_readme_file,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "reference_files": self.reference_files,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


def _cfg_for_variant(spec: VariantSpec, remote_output_dir: str | None = None) -> TaskConfig:
    kwargs = {
        "VARIANT_NAME": spec.variant_name,
        "VARIANT_LABEL": spec.variant_label,
        "N_QUBITS": spec.n_qubits,
        "ANCILLA_MODE": spec.ancilla_mode,
        "HAS_ANCILLA_STATE": spec.has_ancilla_state,
        "COUPLING_U": spec.coupling_u,
        "REQUIRES_CORRELATORS": spec.requires_correlators,
    }
    if remote_output_dir is not None:
        kwargs["REMOTE_OUTPUT_DIR"] = remote_output_dir
    return TaskConfig(**kwargs)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(spec).task_description,
            metadata=_cfg_for_variant(spec).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_payloads(
    session: cb.DesktopSession,
    paths: dict[str, str],
) -> tuple[dict[str, bytes], list[str]]:
    payloads: dict[str, bytes] = {}
    missing: list[str] = []
    for name, path in paths.items():
        if not await session.exists(path):
            missing.append(path)
            continue
        payloads[name] = await session.read_bytes(path)
    return payloads, missing


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    variant = get_variant(meta["variant_name"])

    agent_payloads, missing_output_paths = await _read_payloads(session, meta["output_files"])
    if missing_output_paths:
        logger.info("[%s] missing output paths: %s", variant.variant_name, missing_output_paths)
        return [0.0]

    reference_payloads, missing_reference_paths = await _read_payloads(
        session, meta["reference_files"]
    )
    if missing_reference_paths:
        message = (
            f"[{variant.variant_name}] missing hidden reference paths: {missing_reference_paths}"
        )
        logger.error(message)
        raise RuntimeError(message)

    result = score_submission_bytes(
        variant=variant,
        agent_payloads=agent_payloads,
        reference_payloads=reference_payloads,
    )
    logger.info(
        "[%s] evaluation=%s", variant.variant_name, json.dumps(result.to_dict(), sort_keys=True)
    )
    return [float(result.score)]
