"""Per-episode knobs for Google's Antigravity CLI (``agy``).

The default ALE path reuses an OAuth credential the operator obtains once by
logging in interactively on the host:

    ~/.gemini/antigravity-cli/antigravity-oauth-token   (carries a refresh_token)

That file's content is forwarded into the sandbox by the lifecycle env
passthrough (``ANTIGRAVITY_OAUTH_TOKEN`` inline, or ``ANTIGRAVITY_OAUTH_TOKEN_PATH``
pointing at the host file) and written back into place by the deployer, after
which ``agy`` silent-auths headlessly. See the module docstring on the deployer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AntigravityCliConfig:
    """Tunables for :class:`AntigravityCliDeployer`. Standalone (no shared base)."""

    name: ClassVar[str] = "antigravity-cli"

    # Model display name as printed by ``agy models`` (accepted verbatim).
    # Empty => let agy use its configured default.
    model: str = "Gemini 3.7 Flash (High)"

    # Bypass all tool-permission prompts (required headless). Maps to
    # ``--dangerously-skip-permissions``.
    dangerously_skip_permissions: bool = True

    # Pinned CLI version. The deployer enforces this exact version.
    cli_version: str = "1.1.25"

    # Override the install source. Empty => resolve the exact version from the
    # official updater manifest. A direct binary/tarball URL may be supplied for
    # reproducibility or pre-release validation.
    download_url: str = ""
    # Required SHA-512 hex digest when download_url is set. The official
    # manifest supplies its own checksum when download_url is empty.
    download_sha512: str = ""

    # agy print-mode wait, passed as ``--print-timeout``. agy's OWN default is
    # only 5m, which silently cuts off any longer task (e.g. a slow thinking
    # model, or a multi-step task) before it finishes — the run then has no
    # output. Set it well above any task's wall budget so the orchestration's
    # ``wall_time_s`` is the real cap, not agy's internal timer.
    print_timeout: str = "120m"
