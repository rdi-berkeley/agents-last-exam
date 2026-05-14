"""gym-style env registry.

Bind a task to an env id once, instantiate by id any number of times::

    import ale

    ale.register("demo/hello", entry_point=ale.AgenthleEnv, task_path="demo/hello")
    env = ale.make("demo/hello", provider=GCSDirectProvider(...))
    obs = await env.reset_async(variant_index=0)

Auto-discovery: ``make()`` lazily registers any id that resolves to
``tasks/<env_id>/main.py`` if not explicitly registered, so for the common
case (one env per task module) you don't need to call ``register()``
manually.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.env import AgenthleEnv


# Repo root is two levels up from this file (.../ale/registry.py → repo).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASKS_DIR = _REPO_ROOT / "tasks"


@dataclass(frozen=True)
class _Entry:
    env_id: str
    entry_point: type
    kwargs: dict[str, Any]


_REGISTRY: dict[str, _Entry] = {}


# =============================================================================
# Public API
# =============================================================================

def register(
    env_id: str,
    *,
    entry_point: type,
    **kwargs: Any,
) -> None:
    """Bind ``env_id`` to an env class + fixed ctor kwargs.

    Re-registering the same id is an error — call :func:`unregister` first if
    you need to override. ``entry_point`` is typically :class:`AgenthleEnv`;
    ``kwargs`` typically just ``{"task_path": "..."}``.
    """
    if env_id in _REGISTRY:
        raise ValueError(f"env id already registered: {env_id!r}")
    _REGISTRY[env_id] = _Entry(env_id=env_id, entry_point=entry_point, kwargs=dict(kwargs))


def unregister(env_id: str) -> None:
    _REGISTRY.pop(env_id, None)


def make(env_id: str, **runtime_kwargs: Any) -> "AgenthleEnv":
    """Instantiate the env registered as ``env_id``.

    ``runtime_kwargs`` are merged with the registered kwargs; runtime wins.
    Typical use::

        env = ale.make("demo/hello", provider=GCSDirectProvider(...))
    """
    if env_id not in _REGISTRY:
        _try_auto_register(env_id)
    if env_id not in _REGISTRY:
        available = sorted(_REGISTRY) + _list_unregistered_task_ids()
        raise KeyError(
            f"no env registered: {env_id!r}. "
            f"Available (registered + auto-discoverable): {available}"
        )
    entry = _REGISTRY[env_id]
    final_kwargs = {**entry.kwargs, **runtime_kwargs}
    return entry.entry_point(**final_kwargs)


def list_envs(*, include_auto: bool = True) -> list[str]:
    """List env ids. Default includes auto-discoverable ones from tasks/."""
    ids = set(_REGISTRY)
    if include_auto:
        ids.update(_list_unregistered_task_ids())
    return sorted(ids)


def auto_discover() -> int:
    """Eagerly register every ``tasks/<domain>/<name>/main.py`` as an env id.

    Returns the count of newly-registered ids. Safe to call repeatedly — only
    unregistered ids are added.
    """
    count = 0
    for task_path in _list_task_paths():
        if task_path not in _REGISTRY:
            register(task_path, entry_point=_default_entry_point(), task_path=task_path)
            count += 1
    return count


# =============================================================================
# Internals
# =============================================================================

def _default_entry_point() -> type:
    # Lazy import to avoid circular: registry → env → loader → registry.
    from .core.env import AgenthleEnv
    return AgenthleEnv


def _try_auto_register(env_id: str) -> None:
    """If ``tasks/<env_id>/main.py`` exists, register it on the fly."""
    main_py = _TASKS_DIR / env_id / "main.py"
    if main_py.is_file():
        register(env_id, entry_point=_default_entry_point(), task_path=env_id)


def _list_task_paths() -> list[str]:
    """Walk ``tasks/`` for ``<dir>/main.py`` and return env-id strings."""
    if not _TASKS_DIR.is_dir():
        return []
    out: list[str] = []
    for main_py in _TASKS_DIR.rglob("main.py"):
        rel = main_py.relative_to(_TASKS_DIR).parent
        out.append(str(rel).replace("\\", "/"))
    return sorted(out)


def _list_unregistered_task_ids() -> list[str]:
    return [tp for tp in _list_task_paths() if tp not in _REGISTRY]
