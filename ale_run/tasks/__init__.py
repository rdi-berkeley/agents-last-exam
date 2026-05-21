"""Task discovery + setup/evaluate lifecycle.

Decoupled from VM provisioning and from agents. Tasks are discovered via
``TaskLoader`` (reads ``main.py`` + ``task_card.json``); ``TaskDriver`` runs
their setup/evaluate functions against an open CUA session supplied by
the environment layer.

``TaskDriver`` is intentionally NOT re-exported at package level — importing
it pulls in ``cua_bench``, which a yaml-only ``--dry-run`` shouldn't need.
``from ale_run.tasks.driver import TaskDriver`` when you need it.
"""

from .loader import TaskDataSpec, TaskLoader

__all__ = ["TaskDataSpec", "TaskLoader"]
