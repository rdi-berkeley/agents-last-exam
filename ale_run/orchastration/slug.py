"""Slug + run_id helpers per LOG_SPEC §1.

Verbatim translation of the reference Python in the spec.
"""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slug_model(model: str) -> str:
    if not model:
        return "unknown-model"
    s = model.lower().replace(".", "-").replace("/", "-").replace("_", "-")
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "unknown-model"


def slug_task(task_path: str) -> str:
    return task_path.strip("/").replace("/", "__")


def slug_agent(agent_name: str) -> str:
    s = (agent_name or "unknown").lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_") or "unknown"


def build_run_id(*, agent_id: str, model: str, task_path: str, variant_index: int, ts: str) -> str:
    return (
        f"{slug_agent(agent_id)}__{slug_model(model)}__"
        f"{slug_task(task_path)}__v{variant_index}__{ts}"
    )
