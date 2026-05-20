"""SDK back-port subclasses for older CUA pins.

These three classes re-implement, on an older CUA SDK pin, behavior that
the openclaw fork already ships in its SDK source. They exist for
consumers (notably the orchestration repo at pin ``4b23b5b7``) whose CUA
SDK predates the relevant fork commits. At the harness's own pin
(``b420c6e8``) the overrides match the SDK's native behavior — i.e. they
are no-ops here. We keep them in the package so a single fix propagates
to all consumers via a normal sparse submodule pull.

Consumer responsibilities:
  - Decide whether to install these classes (typically by reassigning the
    SDK module globals: ``agent.callbacks.trajectory_saver.TrajectorySaverCallback``,
    etc.). The harness package does NOT auto-monkey-patch — that wiring
    stays at the consumer's import site.

SDK-private symbols depended on (review on every SDK pin bump in any
consumer):
  - ``ImageRetentionCallback._apply_image_retention`` body shape;
    ``self.only_n_most_recent_images``.
  - ``TrajectorySaverCallback.on_responses`` body; ``self.trajectory_id``,
    ``self._get_turn_dir``, ``self._save_artifact``, ``self.model``.
  - ``ComputerAgent._handle_item`` body and dispatch helpers
    (``_on_function_call_start/end``, ``_on_screenshot``, ``_get_tool``,
    ``screenshot_delay``, ``telemetry_enabled``, ``_dispatch_function_call``).

Note that ``OpenClawComputerHandler`` lives at the package root, not here:
its keypress override is a real fix needed at every pin (chord-vs-sequence
ambiguity in OpenAI's computer-use spec), not a back-port.
"""

from .image_retention import OpenClawImageRetentionCallback
from .trajectory_saver import OpenClawTrajectorySaverCallback

__all__ = [
    "OpenClawImageRetentionCallback",
    "OpenClawTrajectorySaverCallback",
]
