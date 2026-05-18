"""Env-var passthrough policy for VM + docker runtimes.

ALE convention: API keys + routing config live in the operator's shell
environment (``.env`` / ``.envrc`` / ``source``), not in experiment yaml
or config dataclasses. The framework reads ``os.environ`` at unit-start
and propagates the same names into the VM's Python process (via
:mod:`._vm_entry`) and the docker container (via ``--env-file``).

Why a fixed list rather than "propagate all": keeps the surface area
explicit + auditable, prevents accidental leak of host shell state
(``PATH``, ``HOME``, ``USER``, ...) into substrate processes.

To add a new propagated var (e.g. ``DEEPSEEK_API_KEY``), append it here.
"""
from __future__ import annotations

import os


PROPAGATED_ENV_VARS: tuple[str, ...] = (
    # Anthropic CLI (claude-code) — direct + OpenRouter remap.
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    # OpenRouter — both as primary key (ale_claw / litellm) and as the
    # source for the ANTHROPIC_AUTH_TOKEN remap (claude-code in VM).
    "OPENROUTER_API_KEY",
    # OpenAI / Azure (ale_claw via litellm).
    "OPENAI_API_KEY",
    # Search tool (ale_claw's web_search tool — disabled by default).
    "BRAVE_API_KEY",
)


def collect_host_env() -> dict[str, str]:
    """Snapshot the propagated env vars currently set in ``os.environ``.

    Returns only vars that are non-empty — empty strings are dropped so
    downstream doesn't see them as "set" when they're effectively unset.
    """
    return {
        k: os.environ[k]
        for k in PROPAGATED_ENV_VARS
        if os.environ.get(k)
    }
