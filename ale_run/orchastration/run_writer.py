"""RunWriter: owns one run's directory + events.jsonl + finalize files.

LOG_SPEC.md is the source of truth; this class is its only writer. Layout:

    <output_root>/<agent_id>/<model_slug>/<task_slug>/v<i>/<YYYYMMDD_HHMMSS>/
        events.jsonl       append-only, fsync per line
        run.json           schema_version=2, written once at finalize
        trajectory.json    ATIF-v1.0 from Trajectory.model_dump_json(indent=2)
        eval_result.json   {eval_status, score, eval_duration_s, error}
        origin_log/<agent_name>/    deployer work_dir pulled from VM
        output/                     stub: empty + output_gather_skipped event

The constructor refuses to overwrite an existing run dir
(``FileExistsError``). Each finalize write is wrapped in try/except; the
``events.jsonl`` is the authoritative trace even if one of the other writes
fails.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .slug import build_run_id, slug_agent, slug_model, slug_task

logger = logging.getLogger(__name__)


class RunWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        agent_id: str,
        model: str,
        task_path: str,
        variant_index: int,
    ):
        self._ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self._slug_agent = slug_agent(agent_id)
        self._slug_model = slug_model(model)
        self._slug_task = slug_task(task_path)
        self._variant_index = variant_index

        self._run_dir = (
            output_root
            / self._slug_agent
            / self._slug_model
            / self._slug_task
            / f"v{variant_index}"
            / self._ts
        )
        # Refuse to overwrite — LOG_SPEC §1 collision policy.
        if self._run_dir.exists():
            raise FileExistsError(f"run dir already exists: {self._run_dir}")
        self._run_dir.mkdir(parents=True, exist_ok=False)
        (self._run_dir / "origin_log").mkdir(parents=True, exist_ok=True)
        (self._run_dir / "output").mkdir(parents=True, exist_ok=True)

        self._run_id = build_run_id(
            agent_id=agent_id,
            model=model,
            task_path=task_path,
            variant_index=variant_index,
            ts=self._ts,
        )

        self._events_path = self._run_dir / "events.jsonl"
        # Line-buffered append; fsync after each write for SIGTERM safety.
        self._events_fh = self._events_path.open("a", buffering=1, encoding="utf-8")

    # ------------------------------------------------------------------ props

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def ts(self) -> str:
        return self._ts

    # ----------------------------------------------------------------- events

    def emit_event(self, event_type: str, **data: Any) -> None:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": event_type,
            "run_id": self._run_id,
        }
        if data:
            payload["data"] = data
        line = json.dumps(payload, default=str, ensure_ascii=False)
        try:
            self._events_fh.write(line + "\n")
            self._events_fh.flush()
            os.fsync(self._events_fh.fileno())
        except (OSError, ValueError) as e:
            # ValueError when fh is closed; OSError on fsync against a bad fd.
            logger.warning("emit_event(%s) failed: %s", event_type, e)

    # --------------------------------------------------------------- finalize

    def write_run_json(self, meta: dict[str, Any]) -> None:
        path = self._run_dir / "run.json"
        try:
            path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("write_run_json failed: %s", e)

    def write_trajectory(self, traj: Any) -> None:
        path = self._run_dir / "trajectory.json"
        try:
            blob = traj.model_dump_json(indent=2)
        except AttributeError:
            blob = json.dumps(traj, indent=2, ensure_ascii=False, default=str)
        try:
            path.write_text(blob, encoding="utf-8")
        except OSError as e:
            logger.warning("write_trajectory failed: %s", e)

    def write_eval_result(
        self,
        *,
        eval_status: str,
        score: float | None,
        eval_duration_s: float | None,
        error: dict[str, Any] | None,
    ) -> None:
        path = self._run_dir / "eval_result.json"
        payload = {
            "eval_status": eval_status,
            "score": score,
            "eval_duration_s": eval_duration_s,
            "error": error,
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("write_eval_result failed: %s", e)

    def close(self) -> None:
        try:
            self._events_fh.close()
        except OSError:
            pass
