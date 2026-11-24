"""TrajectorySaverCallback turn-bump back-port.

For CUA SDK pins that predate the openclaw fork, ``on_responses``
increments ``self.current_turn`` at the end — immediately after saving
the LLM response artifact, before the computer action that consumes the
response runs. The early increment causes ``cb._get_turn_dir()`` to
return the next turn's empty dir for any caller reading turn state
between ``on_responses`` and the action that follows.

Upstream ``trycua/cua`` agrees with this fix: commits ``1f73713d`` and
``6275ea8d`` ("fix(trajectory_saver): Fix trajectory turn increment to
keep agent decision and execution in same turn") landed on main after
the older target pin, so older pins still have the bug.

The fork's behavioural diff at commit ``b420c6e8`` is one hunk: comment
out ``self.current_turn += 1`` at the tail of ``on_responses``. The
other two increment sites (inside ``on_computer_call_end``) are unchanged.

This subclass duplicates ``on_responses``'s body minus the trailing
increment. Other callbacks delegate to ``super()``. At pins that already
include the fix, the override is a no-op.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from agent.callbacks.trajectory_saver import TrajectorySaverCallback
from typing_extensions import override


class OpenClawTrajectorySaverCallback(TrajectorySaverCallback):
    """TrajectorySaverCallback without the ``on_responses`` turn bump."""

    _SECRET_FIELDS = frozenset(
        {"api_key", "apikey", "authorization", "proxy_authorization", "token", "access_token"}
    )

    def _remember_secrets(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                field = str(key).lower().replace("-", "_")
                if (
                    (field in self._SECRET_FIELDS or field.endswith("_api_key"))
                    and isinstance(item, str)
                    and item
                ):
                    self._secret_values.add(item)
                else:
                    self._remember_secrets(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._remember_secrets(item)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if str(key).lower().replace("-", "_") in self._SECRET_FIELDS
                or str(key).lower().replace("-", "_").endswith("_api_key")
                else self._redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact(item) for item in value)
        if isinstance(value, str):
            for secret in getattr(self, "_secret_values", ()):
                value = value.replace(secret, "[REDACTED]")
        return value

    def _save_artifact(self, name: str, artifact: Any) -> None:
        super()._save_artifact(name, self._redact(artifact))

    @override
    async def on_run_start(self, kwargs: Dict[str, Any], old_items: list[dict[str, Any]]) -> None:
        self._secret_values: set[str] = set()
        self._remember_secrets(kwargs)
        await super().on_run_start(self._redact(kwargs), self._redact(old_items))

    @override
    async def on_run_end(
        self,
        kwargs: Dict[str, Any],
        old_items: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
    ) -> None:
        await super().on_run_end(
            self._redact(kwargs), self._redact(old_items), self._redact(new_items)
        )
        if self.trajectory_id and kwargs.get("ale_claw_status") not in (None, "completed"):
            path = self.trajectory_dir / self.trajectory_id / "metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["status"] = kwargs.get("ale_claw_status", "failed")
            path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @override
    async def on_responses(self, kwargs: Dict[str, Any], responses: Dict[str, Any]) -> None:
        if not self.trajectory_id:
            return

        self._get_turn_dir()
        response_data = {
            "timestamp": str(uuid.uuid1().time),
            "model": self.model,
            "kwargs": kwargs,
            "response": responses,
        }
        self._save_artifact("agent_response", response_data)
