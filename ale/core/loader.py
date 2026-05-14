"""File-based task discovery.

Tasks live at ``<repo-root>/tasks/<domain>/<name>/main.py`` and follow the
existing agenthle convention:
    - module top-level ``config`` instance (TaskConfig)
    - ``@cb.tasks_config(...)`` decorated ``load() -> list[cb.Task]``
    - ``@cb.setup_task(...)`` decorated async ``start(task, session)``
    - ``@cb.evaluate_task(...)`` decorated async ``evaluate(task, session) -> [float]``

Optional sibling ``task_card.json`` declares VM resources (snapshot, vcpus,
memory_gb, disk_gb, gpu). ``load_task`` returns a :class:`LoadedTask` bundle.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .provider import OS, EnvSpec


# Repo root: parent of the directory containing this file, two levels up
# (.../agent-last-exam/ale/core/loader.py → .../agent-last-exam/).
REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "tasks"


# -----------------------------------------------------------------------------
# Snapshot → OS inference
# -----------------------------------------------------------------------------

_OS_BY_SNAPSHOT: dict[str, OS] = {
    "cpu-free": "windows",
    "cpu-license": "windows",
    "cpu-free-ubuntu": "linux",
    "gpu-free": "linux",
    "gpu-free-ubuntu": "linux",
    "gpu-license": "linux",
}


def infer_os_from_snapshot(snapshot: str) -> OS:
    """Map a snapshot tag to its OS. Falls back to keyword sniffing."""
    if snapshot in _OS_BY_SNAPSHOT:
        return _OS_BY_SNAPSHOT[snapshot]
    low = snapshot.lower()
    if "ubuntu" in low or "linux" in low:
        return "linux"
    if "windows" in low or "win" in low:
        return "windows"
    raise ValueError(
        f"Unknown snapshot {snapshot!r}; register in _OS_BY_SNAPSHOT or "
        f"include 'linux'/'ubuntu'/'windows' in the tag."
    )


# -----------------------------------------------------------------------------
# LoadedTask
# -----------------------------------------------------------------------------

StartFn = Callable[[Any, Any], Awaitable[None]]
EvaluateFn = Callable[[Any, Any], Awaitable[list[float]]]


@dataclass
class LoadedTask:
    """One picked variant of a task, ready for AgenthleEnv to drive."""

    task_path: str                # e.g. "demo/hello"
    variant_index: int
    cb_task: Any                  # cua_bench.Task instance (the chosen variant)
    start_fn: StartFn             # async (cb_task, session) -> None
    evaluate_fn: EvaluateFn       # async (cb_task, session) -> [float]
    task_card: dict[str, Any]     # parsed task_card.json, or {}

    @property
    def description(self) -> str:
        return self.cb_task.description or ""

    @property
    def env_spec(self) -> EnvSpec:
        """Build an EnvSpec from task_card.json + cb_task.computer.

        Accepts both layouts:
            {"snapshot": "...", ...}                 (flat, ale early style)
            {"vm": {"snapshot": "...", ...}, ...}    (agenthle-aligned)
        """
        vm = self.task_card.get("vm", {}) or {}
        snapshot = self.task_card.get("snapshot") or vm.get("snapshot")
        if not snapshot:
            raise ValueError(
                f"task {self.task_path}: missing 'snapshot' in task_card.json "
                f"(checked top-level and vm.snapshot)"
            )
        os_type: OS | None = None
        if self.cb_task.computer:
            os_type = self.cb_task.computer.get("setup_config", {}).get("os_type")
        os_type = os_type or infer_os_from_snapshot(snapshot)
        return EnvSpec(
            snapshot=snapshot,
            os=os_type,  # type: ignore[arg-type]
            vcpus=int(vm.get("vcpus", 4)),
            memory_gb=int(vm.get("memory_gb", 16)),
            disk_gb=int(vm.get("disk_gb", 200)),
            gpu=vm.get("gpu"),
        )


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _resolve_task_dir(task_path: str) -> Path:
    """``"demo/hello"`` → ``<repo>/tasks/demo/hello``."""
    rel = Path(task_path.strip("/"))
    target = TASKS_DIR / rel
    if not target.is_dir():
        raise FileNotFoundError(
            f"Task directory not found: {target}\n"
            f"(repo root: {REPO_ROOT}; task_path={task_path!r})"
        )
    return target


def _import_task_module(main_py: Path, task_path: str):
    """Load the task's ``main.py`` via importlib.

    The module name is uniquefied by task path so re-loading different
    variants of different tasks works in the same process. Ensures
    ``REPO_ROOT`` is on ``sys.path`` so ``from tasks.common_config import ...``
    resolves correctly.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module_name = f"_ale_task_{task_path.replace('/', '_').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build module spec for {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_by_td_type(module, td_type: str):
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_td_type", None) == td_type:
            return obj
    return None


def load_task(task_path: str, variant_index: int = 0) -> LoadedTask:
    """Load ``tasks/<task_path>/main.py`` + ``task_card.json``; pick variant.

    Args:
        task_path: e.g. ``"demo/hello"``. Resolves to ``<repo>/tasks/demo/hello/``.
        variant_index: which entry of ``load()``'s returned list to use.
    """
    task_dir = _resolve_task_dir(task_path)
    main_py = task_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"main.py missing: {main_py}")
    module = _import_task_module(main_py, task_path)

    # Find decorated functions.
    load_fn = _find_by_td_type(module, "tasks_config")
    start_fn = _find_by_td_type(module, "setup_task")
    evaluate_fn = _find_by_td_type(module, "evaluate_task")
    missing = [
        n for n, fn in (("tasks_config", load_fn),
                         ("setup_task", start_fn),
                         ("evaluate_task", evaluate_fn))
        if fn is None
    ]
    if missing:
        raise ValueError(
            f"Task {task_path}: missing @cb.{'/@cb.'.join(missing)} decorated function(s)"
        )

    variants = load_fn()  # type: ignore[misc]
    if not isinstance(variants, list) or not variants:
        raise ValueError(
            f"Task {task_path}: load() must return a non-empty list, got {variants!r}"
        )
    if not 0 <= variant_index < len(variants):
        raise IndexError(
            f"Task {task_path}: variant_index={variant_index} out of range "
            f"(have {len(variants)} variants)"
        )
    cb_task = variants[variant_index]

    # Parse task_card.json (optional).
    task_card_path = task_dir / "task_card.json"
    task_card: dict[str, Any] = {}
    if task_card_path.is_file():
        task_card = json.loads(task_card_path.read_text())

    return LoadedTask(
        task_path=task_path,
        variant_index=variant_index,
        cb_task=cb_task,
        start_fn=start_fn,           # type: ignore[arg-type]
        evaluate_fn=evaluate_fn,     # type: ignore[arg-type]
        task_card=task_card,
    )
