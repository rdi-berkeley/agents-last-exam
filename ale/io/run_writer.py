"""RunWriter: emits the on-disk layout defined in the log spec.

Owns one run directory:
    ``<output_root>/<agent>/<model>/<task_slug>/v<i>/<YYYYMMDD_HHMMSS>/``

Produces:
- ``events.jsonl``     — append-only, one record per :meth:`emit_event`
- ``trajectory.json``  — written once by :meth:`write_trajectory`
- ``eval_result.json`` — written once by :meth:`write_eval_result`
- ``run.json``         — written once by :meth:`write_run_json` (finalize)
- ``output/``          — populated externally (ArtifactMirror)
- ``stdout.log`` / ``stderr.log`` — optional, written by :meth:`write_log`

There is no ``runner.log`` — ``events.jsonl`` is the single source of
machine + human readable phase trace. Render via
``jq -r '.ts + "  " + .type' events.jsonl`` for glance.

See docs/sessions/2026-05-13_orchestration_plan.md §"Log spec proposal".
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Slug helpers
# =============================================================================

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slug_model(model: str) -> str:
    """Canonical model slug. ``.`` / ``/`` → ``-``, lowercased."""
    s = model.lower().replace(".", "-").replace("/", "-").replace("_", "-")
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "unknown-model"


def slug_task(task_path: str) -> str:
    """``"demo/hello"`` → ``"demo__hello"`` (matches agenthle convention)."""
    return task_path.strip("/").replace("/", "__")


def slug_agent(agent_name: str) -> str:
    """Normalize agent name for filesystem use."""
    s = (agent_name or "unknown").lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_") or "unknown"


def utc_timestamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# RunWriter
# =============================================================================

class RunWriter:
    """Owns one run directory. Construct, emit events, finalize.

    Usage::

        rw = RunWriter.create(
            output_root=Path(".logs/2026-05-13"),
            agent_name="claude-code",
            model="claude-opus-4-7",
            task_path="demo/hello",
            variant_index=0,
        )
        rw.emit_event("run_started", agent="claude-code", model="claude-opus-4-7")
        ...
        rw.write_trajectory(traj)
        rw.write_eval_result({"score": 1.0})
        rw.write_run_json(meta_dict)
        rw.close()
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        run_dir: Path,
        agent_name: str,
        model: str,
        task_path: str,
        variant_index: int,
        run_id: str,
    ):
        self.run_dir = run_dir
        self.agent_name = agent_name
        self.model = model
        self.task_path = task_path
        self.variant_index = variant_index
        self.run_id = run_id
        self._events_fp = (run_dir / "events.jsonl").open("a", encoding="utf-8")

    # ---- factory ----

    @classmethod
    def create(
        cls,
        *,
        output_root: Path,
        agent_name: str,
        model: str,
        task_path: str,
        variant_index: int,
        timestamp: datetime | None = None,
    ) -> "RunWriter":
        ts = utc_timestamp(timestamp)
        run_dir = (
            output_root
            / slug_agent(agent_name)
            / slug_model(model)
            / slug_task(task_path)
            / f"v{variant_index}"
            / ts
        )
        if run_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run dir: {run_dir}. "
                f"same (agent, model, task, variant, ts) collided."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "output").mkdir()
        (run_dir / "origin_log").mkdir()

        run_id = (
            f"{slug_agent(agent_name)}__{slug_model(model)}__"
            f"{slug_task(task_path)}__v{variant_index}__{ts}"
        )
        rw = cls(
            run_dir=run_dir,
            agent_name=agent_name,
            model=model,
            task_path=task_path,
            variant_index=variant_index,
            run_id=run_id,
        )
        return rw

    # ---- structured events ----

    def emit_event(self, event_type: str, **fields: Any) -> None:
        """Append one event to events.jsonl. Always called from orchestrator code."""
        record = {
            "ts": utc_iso(),
            "type": event_type,
            "run_id": self.run_id,
        }
        if fields:
            record["data"] = fields
        self._events_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events_fp.flush()

    # ---- file artifacts ----

    def write_trajectory(self, traj: BaseModel) -> None:
        """Dump a :class:`Trajectory` to trajectory.json."""
        (self.run_dir / "trajectory.json").write_text(
            traj.model_dump_json(indent=2), encoding="utf-8",
        )

    def write_eval_result(
        self,
        *,
        eval_status: str,
        score: float | None,
        eval_duration_s: float | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """Write eval_result.json with the canonical 4-field schema.

        ``eval_status``: ``"success"`` | ``"failed"`` | ``"not_executed"``.
        ``score``:       float when status==success, else None.
        ``eval_duration_s``: wall time of ``task.evaluate()``. None if not_executed.
        ``error``:       ``{"type", "message", "traceback"}`` when failed.
        """
        payload: dict[str, Any] = {
            "eval_status": eval_status,
            "score": score,
            "eval_duration_s": eval_duration_s,
            "error": error,
        }
        (self.run_dir / "eval_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def write_run_json(self, run_meta: dict[str, Any]) -> None:
        """Write canonical run.json. Caller assembles the dict per spec §run.json."""
        run_meta = dict(run_meta)  # shallow copy
        run_meta.setdefault("schema_version", self.SCHEMA_VERSION)
        run_meta.setdefault("run_id", self.run_id)
        (self.run_dir / "run.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def write_text_artifact(self, name: str, text: str) -> None:
        """Write any UTF-8 string into the run dir (e.g. stdout.log / stderr.log)."""
        if "/" in name or "\\" in name:
            raise ValueError(f"name must be a single filename, got {name!r}")
        (self.run_dir / name).write_text(text, encoding="utf-8")

    def write_bytes_artifact(self, name: str, data: bytes) -> None:
        if "/" in name or "\\" in name:
            raise ValueError(f"name must be a single filename, got {name!r}")
        (self.run_dir / name).write_bytes(data)

    # ---- output mirror ----

    async def mirror_vm_output(
        self,
        session: Any,
        vm_output_dir: str,
        *,
        max_files: int = 256,
    ) -> int:
        """Pull files from the VM's output dir into ``run_dir/output/``.

        Best-effort; returns count of files pulled. Uses ``session.list_dir`` +
        ``session.read_bytes`` (or ``read_file``).
        """
        try:
            names = await session.list_dir(vm_output_dir)
        except Exception as exc:                # noqa: BLE001
            self.note(f"mirror_vm_output: list_dir({vm_output_dir}) failed: {exc}")
            return 0
        local = self.run_dir / "output"
        pulled = 0
        for name in sorted(names)[:max_files]:
            sep = "\\" if "\\" in vm_output_dir or vm_output_dir[1:2] == ":" else "/"
            remote_path = vm_output_dir + sep + name
            try:
                if hasattr(session, "read_bytes"):
                    data = await session.read_bytes(remote_path)
                else:
                    txt = await session.read_file(remote_path)
                    data = txt.encode("utf-8") if isinstance(txt, str) else txt
            except Exception as exc:            # noqa: BLE001
                self.note(f"mirror_vm_output: failed to read {remote_path}: {exc}")
                continue
            (local / name).write_bytes(data)
            pulled += 1
        return pulled

    # ---- lifecycle ----

    def close(self) -> None:
        try:
            self._events_fp.close()
        except Exception:                       # noqa: BLE001
            pass

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
