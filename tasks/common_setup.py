"""Common task setup base for Agent-LE tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CREDENTIALS_LOCAL_DIR = (
    Path.home() / ".config" / "agenthle" / "credentials"
)


def credentials_local_dir() -> Path:
    """Operator-side directory holding agent-credential source files.
    """
    env = os.environ.get("AGENTHLE_CREDENTIALS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CREDENTIALS_LOCAL_DIR


class BaseTaskSetup:
    """Unified entry point for per-task setup.

    The framework's hook point for any per-run setup behavior that
    should apply uniformly across all tasks. Today ``_pre`` injects
    declared agent-side credentials (see
    ``docs/task_impl_guides/admin/stage1/07_CREDENTIALS_AND_LICENSES.md``).
    Other future features land here too: per-run secret allocation,
    telemetry spans, network proxy configuration, snapshot verification.

    Every task's ``start()`` dispatches through an instance of this
    class. Tasks with no per-task setup needs use the base directly;
    the small number of tasks that need per-run state mutation
    subclass it and override ``setup()``.
    """

    async def __call__(self, task_cfg: Any, session: Any) -> None:
        await self._pre(task_cfg, session)
        await self.setup(task_cfg, session)
        await self._post(task_cfg, session)

    async def _pre(self, task_cfg: Any, session: Any) -> None:
        """Framework-managed pre-setup hook.

        Currently injects credentials declared in
        ``task_card.json``'s ``requiredCredentials`` (surfaced via
        ``task_cfg.metadata["required_credentials"]``) from the operator's
        local credentials directory into ``{input_dir}/credentials/`` on
        the VM. Tasks declaring no credentials get a fast no-op.
        """
        await _inject_required_credentials(task_cfg, session)

    async def setup(self, task_cfg: Any, session: Any) -> None:
        """Task-specific setup. Default is no-op.

        Override only when the work is genuinely irreducible:
        - Per-run unique secrets
        - External-system state reset
        - Per-run container
        - Credential rotation tied to this specific run
        """
        return None

    async def _post(self, task_cfg: Any, session: Any) -> None:
        """Framework-managed post-setup hook. Reserved for future use."""
        return None


async def _inject_required_credentials(task_cfg: Any, session: Any) -> None:
    meta = getattr(task_cfg, "metadata", None) or {}
    creds = meta.get("required_credentials") or []
    if not creds:
        return

    input_dir = meta.get("input_dir")
    if not input_dir:
        raise RuntimeError(
            "BaseTaskSetup credential injection: task_cfg.metadata is missing "
            "'input_dir'. Either the task config does not surface input_dir "
            "in to_metadata(), or requiredCredentials is declared on a task "
            "without an input/ surface."
        )

    sep = "\\" if "\\" in input_dir else "/"
    target_dir = f"{input_dir}{sep}credentials"
    await session.makedirs(target_dir)

    local_dir = credentials_local_dir()
    task_id = meta.get("task_id") or f"{meta.get('domain_name')}/{meta.get('task_name')}"
    for cred in creds:
        cred_type = cred.get("type")
        filename = cred.get("file")
        if not cred_type or not filename:
            raise RuntimeError(
                f"requiredCredentials entry missing 'type' or 'file' "
                f"(task {task_id}): {cred!r}"
            )
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise RuntimeError(
                f"requiredCredentials.file must be a plain filename "
                f"(task {task_id}), got: {filename!r}"
            )
        local_path = local_dir / filename
        if not local_path.exists():
            raise RuntimeError(
                f"Required credential file missing: {local_path}\n"
                f"Task {task_id} declares "
                f"requiredCredentials[type={cred_type}, file={filename}]. "
                f"Operator must provision this file before running the task. "
                f"See docs/task_impl_guides/admin/stage1/07_CREDENTIALS_AND_LICENSES.md."
            )
        content = local_path.read_text(encoding="utf-8")
        await session.write_file(f"{target_dir}{sep}{filename}", content)
