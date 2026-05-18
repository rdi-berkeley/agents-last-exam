"""TaskDataSpec — per-task GCS-data layout metadata.

Tells :mod:`ale.io.data_staging` what to rsync from GCS onto the VM and
where it lands on the VM filesystem. Ported (shape-equivalent) from
``agenthle/scripts/web_console/lib/simprun/task_loader.py`` so the same
``task_card.json`` / ``LinuxTaskConfig.to_metadata()`` shape works in
ALE without task author changes.

Visibility rule (formal benchmark — see ``agenthle/CLAUDE.md``):
  - input/, software/, output/  → staged BEFORE the agent runs (visible)
  - reference/, eval/           → reference is staged ONLY before evaluate;
                                   it MUST NOT be visible to the agent
                                   during solve. The framework enforces
                                   this by deferring :func:`stage_reference`
                                   to ``env.step_async(Submit())``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Default GCS bucket layout — matches simprun's `gs://agenthle/<domain>/<task>/<variant>/`.
# Override via env var ALE_GCS_TASK_DATA_BUCKET.
DEFAULT_TASK_DATA_BUCKET = "gs://agenthle"

# Default GCS results bucket for upload_output. Override via ALE_GCS_RESULTS_BUCKET.
DEFAULT_RESULTS_BUCKET = "gs://agenthle-run-results"

# VM-side data roots — matches simprun.config.vm_data_root.
_VM_DATA_ROOT_LINUX = "/media/user/data/agenthle"
_VM_DATA_ROOT_WINDOWS = "E:\\agenthle"


@dataclass(slots=True)
class TaskDataSpec:
    """Per-task data layout: what to stage from GCS, where it lands on VM.

    ``requires_task_data=False`` (the default) → all stage_*/upload_* are
    no-ops. ``demo/hello`` and other in-VM-only tasks need none of this.

    Real agenthle tasks set ``requires_task_data=True`` (via
    ``task.metadata['requires_task_data']`` or
    ``TaskConfig.REQUIRES_TASK_DATA``) and provide ``domain_name`` /
    ``task_name`` / ``variant_name`` in metadata so the framework can
    derive GCS prefixes + VM paths.
    """

    requires_task_data: bool = False

    # Identity (for deriving GCS paths). All three required when requires_task_data=True.
    domain_name: str | None = None
    task_name: str | None = None
    variant_name: str | None = None

    # VM-side absolute paths. When None, derived from (os_type, domain, task, variant)
    # via :func:`vm_subdir`. Explicit values come from task's metadata when the task
    # author wants to override the default layout.
    input_dir: str | None = None
    software_dir: str | None = None
    reference_dir: str | None = None
    remote_output_dir: str | None = None
    eval_dir: str | None = None

    # GCS-side overrides. When None, derived from
    # ``<DEFAULT_TASK_DATA_BUCKET>/<domain>/<task>/<variant>/<sub>``.
    reference_gcs_prefix: str | None = None
    eval_gcs_prefix: str | None = None


# =============================================================================
# Path conventions (simprun parity)
# =============================================================================

def vm_data_root(os_type: str) -> str:
    """``"linux"`` → ``/media/user/data/agenthle``, ``"windows"`` → ``E:\\agenthle``."""
    return _VM_DATA_ROOT_LINUX if os_type == "linux" else _VM_DATA_ROOT_WINDOWS


def vm_task_dir(os_type: str, domain: str, task: str, variant: str) -> str:
    root = vm_data_root(os_type)
    sep = "/" if os_type == "linux" else "\\"
    return f"{root}{sep}{domain}{sep}{task}{sep}{variant}"


def vm_subdir(os_type: str, domain: str, task: str, variant: str, sub: str) -> str:
    """E.g. ``vm_subdir("linux", "demo", "hello", "simple", "input")`` →
    ``/media/user/data/agenthle/demo/hello/simple/input``.
    Empty ``sub`` returns the task root with no trailing separator suffix."""
    base = vm_task_dir(os_type, domain, task, variant)
    if not sub:
        return base
    sep = "/" if os_type == "linux" else "\\"
    return f"{base}{sep}{sub}"


def gcs_task_prefix(domain: str, task: str, variant: str, *, bucket: str = DEFAULT_TASK_DATA_BUCKET) -> str:
    """``"gs://agenthle/<domain>/<task>/<variant>"``."""
    return f"{bucket}/{domain}/{task}/{variant}"


# =============================================================================
# Extractor (called from ale.core.loader)
# =============================================================================

def extract_task_data(
    *,
    metadata: dict[str, Any] | None,
    cb_task_config: Any | None = None,
) -> TaskDataSpec:
    """Build a :class:`TaskDataSpec` from a task's metadata + optional config.

    Detection (explicit opt-in only):
      1. ``metadata["requires_task_data"]`` if present
      2. ``cb_task_config.REQUIRES_TASK_DATA`` if present

    No implicit heuristic. Tasks that need GCS data staging MUST set one
    of the two flags above; otherwise staging is a full no-op. This is
    deliberately stricter than ``simprun.task_loader._extract_task_data``
    (which inferred from the presence of ``input_dir`` in metadata) —
    ``LinuxTaskConfig.to_metadata()`` always sets ``input_dir`` etc as
    in-VM path conventions, even for tasks like ``demo/hello`` that
    don't need GCS data. The implicit heuristic confused those.
    """
    metadata = metadata or {}
    explicit = metadata.get("requires_task_data")
    if explicit is None and cb_task_config is not None:
        explicit = getattr(cb_task_config, "REQUIRES_TASK_DATA", None)

    if not bool(explicit):
        return TaskDataSpec(requires_task_data=False)

    domain = str(metadata.get("domain_name") or "").strip()
    task = str(metadata.get("task_name") or "").strip()
    variant = str(metadata.get("variant_name") or "").strip()
    if not (domain and task and variant):
        raise RuntimeError(
            "task declares requires_task_data=True but metadata is missing "
            f"domain_name / task_name / variant_name (got domain={domain!r}, "
            f"task={task!r}, variant={variant!r}). Set them in your "
            f"TaskConfig.to_metadata() or task_card.json."
        )

    return TaskDataSpec(
        requires_task_data=True,
        domain_name=domain,
        task_name=task,
        variant_name=variant,
        input_dir=metadata.get("input_dir"),
        software_dir=metadata.get("software_dir"),
        reference_dir=metadata.get("reference_dir"),
        remote_output_dir=metadata.get("remote_output_dir"),
        eval_dir=metadata.get("eval_dir"),
        reference_gcs_prefix=metadata.get("reference_gcs_prefix"),
        eval_gcs_prefix=metadata.get("eval_gcs_prefix"),
    )
