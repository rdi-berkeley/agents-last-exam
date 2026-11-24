"""OpenClaw prompt-cache policy — sliding cache_control breakpoints.

Reproduces OpenClaw's anthropic-payload-policy.ts caching strategy:
  - Cache the system prompt block so it's reused across turns within a run.
  - Apply cache_control on the trailing user/tool_result message so the
    cached prefix grows turn-by-turn (sliding-window pattern). Each new
    turn reads the previous turn's content from cache and writes only
    the small new delta.
  - Optionally split the system prompt at ``OPENCLAW_CACHE_BOUNDARY`` so
    any per-turn dynamic content (timestamps, runtime context) sits
    below the cached prefix.

References:
  - openclaw/src/agents/anthropic-payload-policy.ts (sliding breakpoint)
  - openclaw/src/agents/system-prompt-cache-boundary.ts (boundary marker)
  - openclaw/src/agents/anthropic-family-cache-semantics.ts (model gating)
  - openclaw/src/agents/proxy-stream-wrappers.ts (OpenRouter detection)

Why a callback (not an upstream patch):
The upstream CUA loop (anthropic.py:_add_cache_control) marks the FIRST 4
messages, which pins cache breakpoints at the conversation start and
prevents the cached prefix from extending across turns. We strip those
markers and re-apply them in the right places via ``on_api_start``.
Keeping the change in a callback avoids forking upstream CUA.
"""

from __future__ import annotations

from typing import Any, Dict, List


OPENCLAW_CACHE_BOUNDARY = "<!-- OPENCLAW_CACHE_BOUNDARY -->"
"""Marker inserted by PromptBuilder between stable and dynamic prompt sections.

Content above the marker is cached; content below is not. Mirrors OpenClaw's
SYSTEM_PROMPT_CACHE_BOUNDARY in src/agents/system-prompt-cache-boundary.ts.
"""

_EPHEMERAL: Dict[str, str] = {"type": "ephemeral"}


def supports_anthropic_cache(model: str | None) -> bool:
    """Return True iff Anthropic ``cache_control`` markers are honored for ``model``.

    Mirrors OpenClaw's ``isAnthropicModelRef`` + ``isOpenRouterAnthropicModelRef``:
    only Anthropic-family models (served by Anthropic, Vertex, Bedrock, or
    OpenRouter's anthropic/* route) accept ``{type: "ephemeral"}``. Routing
    a non-Anthropic model (e.g. ``openrouter/openai/gpt-5.4``) through the
    same code path would either no-op or be rejected, so we strip markers
    in those cases.
    """
    if not model:
        return False
    m = model.lower()
    if m.startswith("anthropic/"):
        return True
    if m.startswith("openrouter/anthropic/"):
        return True
    if m.startswith("vertex_ai/claude") or m.startswith("vertex_ai/anthropic"):
        return True
    if m.startswith("bedrock/anthropic"):
        return True
    # liteLLM also accepts bare "claude-..." for direct Anthropic.
    if m.startswith("claude-"):
        return True
    return False


def apply_openclaw_cache_markers(
    messages: List[Dict[str, Any]] | None,
    model: str | None,
) -> None:
    """Apply OpenClaw's sliding-breakpoint cache_control pattern in-place.

    Mutates ``messages``:
      1. Strip any pre-existing message-level markers (e.g. CUA's broken
         first-4 marking from ``anthropic.py:_add_cache_control``).
      2. If ``model`` isn't Anthropic-family, also strip block-level markers
         and return (no caching available).
      3. Otherwise mark the system prompt (splitting at
         ``OPENCLAW_CACHE_BOUNDARY`` if present) and the trailing
         user/tool_result message — sliding breakpoint so the cached prefix
         grows turn by turn.

    Call this from inside the loop's ``predict_step``, before building
    ``api_kwargs``. A callback-hook path won't work — ``agent.py``'s
    ``_on_api_start`` deep-copies via ``get_json`` before invoking
    callbacks, so any mutation made in a hook never reaches the actual
    ``litellm.acompletion`` call.
    """
    if not messages:
        return

    # Always strip pre-existing message-level markers so we have a clean slate.
    for msg in messages:
        msg.pop("cache_control", None)

    if not supports_anthropic_cache(model or ""):
        # Non-Anthropic provider — also strip any block-level markers
        # in case the system prompt was already split before this turn.
        for msg in messages:
            _strip_block_cache_control(msg)
        return

    _apply_system_cache(messages[0])

    # Trailing breakpoint — slides forward each turn so the cached prefix
    # extends to include the previous turn's tool result.
    last = messages[-1]
    if last is not messages[0]:
        last["cache_control"] = dict(_EPHEMERAL)


def _apply_system_cache(msg: Dict[str, Any]) -> None:
    """Mark the system prompt for caching, splitting at the boundary if present.

    The boundary marker (``OPENCLAW_CACHE_BOUNDARY``) is consumed (removed
    from the text sent to the model). When present, the system prompt is
    split into two text blocks; only the stable (above-boundary) block
    receives ``cache_control``.
    """
    content = msg.get("content")

    if isinstance(content, str):
        if OPENCLAW_CACHE_BOUNDARY in content:
            stable, dynamic = content.split(OPENCLAW_CACHE_BOUNDARY, 1)
            blocks: List[Dict[str, Any]] = []
            stable = stable.rstrip()
            if stable:
                blocks.append(
                    {"type": "text", "text": stable, "cache_control": dict(_EPHEMERAL)}
                )
            dynamic = dynamic.lstrip()
            if dynamic:
                blocks.append({"type": "text", "text": dynamic})
            msg["content"] = blocks
        else:
            msg["cache_control"] = dict(_EPHEMERAL)
        return

    if isinstance(content, list):
        for i, block in enumerate(content):
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and OPENCLAW_CACHE_BOUNDARY in block.get("text", "")
            ):
                text = block["text"]
                stable, dynamic = text.split(OPENCLAW_CACHE_BOUNDARY, 1)
                stable = stable.rstrip()
                dynamic = dynamic.lstrip()
                new_blocks: List[Dict[str, Any]] = []
                if stable:
                    new_blocks.append(
                        {
                            "type": "text",
                            "text": stable,
                            "cache_control": dict(_EPHEMERAL),
                        }
                    )
                if dynamic:
                    new_blocks.append({"type": "text", "text": dynamic})
                content[i : i + 1] = new_blocks
                return
        msg["cache_control"] = dict(_EPHEMERAL)
        return

    # Unknown content shape — fall back to message-level marker.
    msg["cache_control"] = dict(_EPHEMERAL)


def _strip_block_cache_control(msg: Dict[str, Any]) -> None:
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)
