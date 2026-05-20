"""Task discovery + setup/evaluate lifecycle.

Decoupled from VM provisioning and from agents. Tasks are discovered via
``TaskLoader`` (reads ``main.py`` + ``task_card.json``); ``TaskEnv`` runs
their setup/evaluate functions against an open CUA session supplied by
the environment layer.

``TaskEnv`` is intentionally NOT re-exported at package level — importing
it pulls in ``cua_bench``, which a yaml-only ``--dry-run`` shouldn't need.
``from ale_run.tasks.task_env import TaskEnv`` when you need it.
"""

from .loader import TaskDataSpec, TaskLoader

__all__ = ["TaskDataSpec", "TaskLoader"]
