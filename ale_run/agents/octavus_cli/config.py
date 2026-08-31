"""OctavusCliConfig: per-episode knobs for the Octavus Agent CLI deployer.

The agent under test is one cloud Octavus agent, selected entirely by its
``oct_agt_*`` key. The single credential is ``api_key`` (typically
``api_key: ${env:OCTAVUS_AGENT_API_KEY}`` in the agent yaml, resolved host-side):
the framework treats any field named ``api_key`` as a secret, so it travels into
the sandbox via the read-once sidecar and never lands in gathered host logs.

Everything else about the harness (prompt, tools, workers, skills, memory,
model) lives in the agent's configuration on the Octavus platform. ``model`` /
``backup_model`` / ``thinking`` / ``capabilities`` here are per-run overrides so
one deployer can express many harness variants without touching the stored agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class OctavusCliConfig:
    """Tunables for :class:`OctavusCliDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; there is no agent-side ``timeout_s``.
    """

    name: ClassVar[str] = "octavus-cli"

    # ---- platform target (the hosted Octavus platform, https://octavus.ai) ----
    operator_url: str | None = None
    """Optional operator WebSocket override (``--operator-url``). Normally
    unset: the platform returns a reachable operator URL per run."""

    api_key: str | None = None
    """The agent ``oct_agt_*`` key. Passed to the run via ``OCTAVUS_API_KEY``
    (kept out of argv). Set it as ``${env:OCTAVUS_AGENT_API_KEY}`` in the agent
    yaml; the ``api_key`` name makes the framework carry it as a secret."""

    # ---- per-run harness overrides (variants) ----
    model: str | None = None
    """Per-run primary model ``provider/model-id`` (``--model``). Also settable
    as the agent-yaml ``model:`` sugar. ``None`` inherits the agent's default."""

    backup_model: str | None = None
    """Per-run backup model ``provider/model-id`` (``--backup-model``)."""

    thinking: str | None = None
    """Per-run thinking/reasoning effort (``--thinking``): one of ``off`` /
    ``low`` / ``medium`` / ``high`` / ``max``. ``max`` is each provider's
    maximum; ``off`` disables thinking. ``None`` (or empty) inherits the agent's
    default, omitting the flag."""

    capabilities: dict[str, bool] = field(default_factory=dict)
    """Per-run capability toggles (repeated ``--capability slug=on|off``), e.g.
    ``{"memory": false}`` for an ablation. Unlisted inherits the agent default.
    A toggle only applies to a capability the agent's protocol declares; toggling
    one it does not declare is rejected by the platform (HTTP 400)."""

    # ---- recording ----
    record: bool = False
    """When true, record each task's execution view (the streamed working process
    beside the live computer) to a single shareable video (``--record``). Off by
    default; the platform gates it to funded tiers, so the benchmark org must be
    on a paid/trial/internal plan for it to take effect."""

    record_visibility: str = "private"
    """Where a recording is stored/served: ``private`` (signed, org-only playback)
    or ``public`` (a permanent, unguessable, shareable URL, via ``--record-public``).
    Benchmark content is non-sensitive, so set ``public`` to get a paste-anywhere
    link per task. Ignored when ``record`` is false."""

    # ---- computer wiring ----
    display: str = ":0"
    """X display the CLI acts on (the desktop ALE's graders screenshot). The
    ALE Linux box runs cua-server on ``:0``; the CLI reuses an existing display,
    so setting this makes screen-graded tasks see the agent's actions. File-
    graded tasks (the majority) are unaffected."""

    chrome_path: str | None = None
    """Explicit browser binary (``--chrome-path``). ``None`` installs Chrome for
    Testing at setup and points the CLI at it. Branded Chrome 137+ cannot load the
    automation extension (``--load-extension`` was removed), so the deployer never
    uses a baked ``google-chrome``; set this only to pin your own CfT/Chromium."""

    # ---- install ----
    cli_version: str | None = "@octavus/agent@1.0.10"
    """npm spec installed globally at setup. Pin a version for reproducibility;
    ``None`` installs the latest ``@octavus/agent``. ``--thinking`` needs
    ``>=1.0.10``; older CLIs ignore the flag."""

    install_prereqs: bool = True
    """When true, ``install()`` best-effort ``apt-get``s the computer's display
    stack (Xvfb, AT-SPI + its python bindings, xdotool, scrot, Chrome libraries)
    unless the box already has both the screenshot/automation tools and the
    AT-SPI2 python bindings the computer-use ``label`` tool needs (checking only
    the former misses images that bake a desktop but omit python3-gi). Set false
    on an image that already bakes the full stack, or for shell/filesystem-only
    runs that need no display."""

    def __post_init__(self) -> None:
        # Only the exact literal "public" enables --record-public; validate here
        # so a typo fails loud at load time instead of silently downgrading the
        # recording to private.
        if self.record_visibility not in ("private", "public"):
            raise ValueError(
                f"record_visibility must be 'private' or 'public', got {self.record_visibility!r}"
            )
        # Validate the thinking effort at load time (empty/None inherits the
        # agent default); a typo would otherwise reach the platform and be
        # rejected only at session creation.
        if self.thinking and self.thinking not in ("off", "low", "medium", "high", "max"):
            raise ValueError(
                "thinking must be one of off/low/medium/high/max (or empty to "
                f"inherit the agent default), got {self.thinking!r}"
            )
