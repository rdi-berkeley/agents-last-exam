"""F — TrajectorySaverCallback turn-bump back-port.

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

import uuid
from typing import Any, Dict

from agent.callbacks.trajectory_saver import TrajectorySaverCallback
from typing_extensions import override


class OpenClawTrajectorySaverCallback(TrajectorySaverCallback):
    """TrajectorySaverCallback without the ``on_responses`` turn bump."""

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
