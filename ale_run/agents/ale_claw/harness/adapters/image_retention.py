"""E — ImageRetentionCallback extensions for the openclaw harness.

Three extensions on top of the SDK's ``ImageRetentionCallback``:

1. **call_id-based pairing** (legacy back-port for SDK pins predating the
   openclaw fork). The SDK assumed the ``computer_call`` that produced a
   ``computer_call_output`` was at ``idx - 1`` — broken when models emit
   interleaved ``function_call`` and ``computer_call`` items in the same
   turn (Opus 4.6 / ``computer_20251124`` class). Fork commit
   ``b420c6e8`` replaced the immediate-predecessor lookup with a
   backward scan keyed on ``call_id``. This subclass mirrors that.

2. **function_call shim coverage** (this is the live fix at every pin).
   The SDK's matcher only finds screenshots inside ``computer_call_output``
   items. Models that don't speak the native ``computer_call`` item
   (Claude, GPT-5.4, anything via OpenRouter) reach the computer tool
   through a function-call shim — the screenshot lands in a *separate
   user message* with ``image_url`` / ``input_image`` content blocks,
   not inside any ``*_output`` item. The SDK retention silently no-ops
   for that path. This subclass also walks user-message content lists,
   strips older image blocks, and replaces them with a stable text
   placeholder when the message is image-only.

3. **Sticky placeholder + selectable threshold mode** (US-OC-072).
   When the SDK's per-block strip empties a user message, the original
   behavior dropped the message entirely. That deletion shifts every
   subsequent message index, busting Anthropic's prefix cache from the
   deletion point onward — verified at ~45% cache hit rate on
   GUI-heavy tasks (see cache-thrash-image-retention.md). Fix:

   - **Sticky placeholder (always on)**: replace the image with a fixed
     text block (`PRUNED_HISTORY_IMAGE_MARKER`). The placeholder is
     byte-stable across calls, so a given message gets mutated at most
     once (when its image ages out). After that mutation, the placeholder
     becomes part of the cached prefix on subsequent turns. Cache prefix
     extension recovers from "pinned at system prompt" to monotonic.

   - **Mode = "openclaw" (default)**: OpenClaw-parity — keep all images
     from the last ``only_n_most_recent_images`` *completed turns*. A
     turn boundary is a transition into an assistant/tool-emitting
     message. Mirrors ``pruneProcessedHistoryImages`` in
     ``openclaw/src/agents/pi-embedded-runner/run/history-image-prune.ts``.
     Multi-screenshot turns stay intact; on-task verification showed
     ~89% cache hit rate vs ~55% for "cua" on hardware/Analog_Active.

   - **Mode = "cua" (opt-in)**: CUA-default behavior — keep the last
     ``only_n_most_recent_images`` images by count. Each new image past
     the budget ages out the oldest one. With sticky-placeholder this
     no longer pins the cache to the system prompt, but the per-turn
     image-aging still triggers one cache invalidation per displaced
     image (vs. one-per-turn-boundary in openclaw mode).

   Native ``computer_call_output`` images still get the original
   remove-the-triple treatment (no placeholder there) — those are only
   produced by ``computer-use-preview`` models which we don't currently
   target, and the triple-removal would need a different placeholder
   shape that we'd want to validate against the API contract first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from agent.callbacks.image_retention import ImageRetentionCallback


PRUNED_HISTORY_IMAGE_MARKER = "[image data removed - already processed by model]"
"""Placeholder text inserted in place of pruned image blocks.

Mirrors OpenClaw's ``PRUNED_HISTORY_IMAGE_MARKER`` from
``src/agents/pi-embedded-runner/run/history-image-prune.ts``. Short
(~8 tokens) and byte-stable so subsequent turns can reuse it as part
of the cached prefix without paying re-write cost.
"""


RetentionMode = Literal["cua", "openclaw"]
"""Mode names track the source benchmark whose retention policy each one
mirrors:
    "cua"      — CUA-default last-N-images-by-count threshold.
    "openclaw" — OpenClaw-parity last-N-completed-turns threshold
                 (mirrors ``pruneProcessedHistoryImages``).

Both modes share the sticky-placeholder fix from US-OC-072. The OpenClaw
threshold became the default after on-task verification (see
``develop-doc/cache-thrash-image-retention.md`` in the agenthle repo).
"""


def _is_image_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") in ("image_url", "input_image")
    )


def _make_placeholder_block() -> Dict[str, Any]:
    return {"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}


class OpenClawImageRetentionCallback(ImageRetentionCallback):
    """ImageRetentionCallback that prunes both native and function-call shim screenshots.

    Args:
        only_n_most_recent_images: Retention budget. In ``mode="cua"``,
            this is the max number of images to keep. In ``mode="openclaw"``,
            this is the max number of completed turns whose images are kept
            (analogous to OpenClaw's ``PRESERVE_RECENT_COMPLETED_TURNS``).
            Pass ``None`` to disable pruning entirely.
        mode: ``"openclaw"`` (default, OpenClaw-parity) or ``"cua"``
            (CUA-compatible). Default flipped from ``"cua"`` to ``"openclaw"``
            after US-OC-072 verification — see
            ``develop-doc/cache-thrash-image-retention.md``.
    """

    def __init__(
        self,
        only_n_most_recent_images: int | None = None,
        mode: RetentionMode = "openclaw",
    ):
        super().__init__(only_n_most_recent_images=only_n_most_recent_images)
        if mode not in ("cua", "openclaw"):
            raise ValueError(f"mode must be 'cua' or 'openclaw', got {mode!r}")
        self.mode = mode

    def _apply_image_retention(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.only_n_most_recent_images is None:
            return messages
        if self.mode == "openclaw":
            return self._apply_openclaw_retention(messages)
        return self._apply_cua_retention(messages)

    # ------------------------------------------------------------------
    # cua mode (CUA-compatible behavior + sticky placeholder)
    # ------------------------------------------------------------------

    def _apply_cua_retention(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = self.only_n_most_recent_images

        # Index every image location across both paths, in message order.
        # Each entry: (msg_idx, kind, block_idx_or_None).
        # kind == "native_output"  → the entire computer_call_output is the image.
        # kind == "user_block"     → one image block within a user-message content list.
        locs: list[tuple[int, str, int | None]] = []
        for idx, msg in enumerate(messages):
            if msg.get("type") == "computer_call_output":
                out = msg.get("output")
                if isinstance(out, dict) and "image_url" in out:
                    locs.append((idx, "native_output", None))
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for bidx, block in enumerate(content):
                    if _is_image_block(block):
                        locs.append((idx, "user_block", bidx))

        if len(locs) <= n:
            return messages

        drop = locs[:-n]
        return self._apply_drops(messages, drop)

    # ------------------------------------------------------------------
    # openclaw mode (OpenClaw-parity — default since US-OC-072 verification)
    # ------------------------------------------------------------------

    def _apply_openclaw_retention(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep images from the last N completed turns; placeholder-replace older ones.

        Mirrors ``pruneProcessedHistoryImages`` in OpenClaw's
        ``history-image-prune.ts``. A "completed turn" is counted by
        transitions into an assistant-emitting message (``role=assistant``
        or ``type in {"reasoning", "function_call", "computer_call"}``).
        """
        n = self.only_n_most_recent_images
        turn_starts = self._find_turn_starts(messages)
        if len(turn_starts) <= n:
            return messages
        # Cutoff index: prune images in messages[0..cutoff). Everything
        # at or after cutoff is in the "recent N turns" window.
        cutoff = turn_starts[-n]

        # Find image locations strictly before the cutoff.
        locs: list[tuple[int, str, int | None]] = []
        for idx in range(cutoff):
            msg = messages[idx]
            if msg.get("type") == "computer_call_output":
                out = msg.get("output")
                if isinstance(out, dict) and "image_url" in out:
                    locs.append((idx, "native_output", None))
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for bidx, block in enumerate(content):
                    if _is_image_block(block):
                        locs.append((idx, "user_block", bidx))

        if not locs:
            return messages
        return self._apply_drops(messages, locs)

    @staticmethod
    def _find_turn_starts(messages: List[Dict[str, Any]]) -> List[int]:
        """Return indices where a new assistant-emitting turn begins.

        A new turn begins on the transition from a non-assistant message
        (user / tool result) to an assistant-emitting one (assistant role,
        reasoning, function_call, computer_call). Consecutive assistant-
        emitting messages within the same turn don't count as new turns.
        """
        ASSISTANT_TYPES = {"reasoning", "function_call", "computer_call"}
        starts: List[int] = []
        prev_was_assistant = False
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            msg_type = msg.get("type")
            is_assistant = (role == "assistant") or (msg_type in ASSISTANT_TYPES)
            if is_assistant and not prev_was_assistant:
                starts.append(idx)
            prev_was_assistant = is_assistant
        return starts

    # ------------------------------------------------------------------
    # Shared drop-application — placeholder for user_block, triple-remove
    # for native_output (preserves API contract for native path).
    # ------------------------------------------------------------------

    def _apply_drops(
        self,
        messages: List[Dict[str, Any]],
        drop: List[tuple[int, str, int | None]],
    ) -> List[Dict[str, Any]]:
        # Native: remove the computer_call_output, its producing computer_call
        # (matched by call_id), and any preceding reasoning block.
        drop_native_indices = {idx for idx, kind, _ in drop if kind == "native_output"}
        to_remove: set[int] = set()
        for idx in drop_native_indices:
            to_remove.add(idx)
            output_call_id = messages[idx].get("call_id")
            for search_idx in range(idx - 1, -1, -1):
                if (
                    messages[search_idx].get("type") == "computer_call"
                    and messages[search_idx].get("call_id") == output_call_id
                ):
                    to_remove.add(search_idx)
                    r_idx = search_idx - 1
                    if r_idx >= 0 and messages[r_idx].get("type") == "reasoning":
                        to_remove.add(r_idx)
                    break

        # Shim: strip per-block from user messages. Messages whose content
        # was image-only become a placeholder text block — NOT deleted.
        # Deletion would shift every subsequent message index, busting the
        # cache prefix from the deletion point onward (see module docstring).
        drop_blocks_by_msg: dict[int, set[int]] = {}
        for idx, kind, bidx in drop:
            if kind == "user_block":
                drop_blocks_by_msg.setdefault(idx, set()).add(bidx)

        out: List[Dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i in to_remove:
                continue
            if i in drop_blocks_by_msg:
                stripped = [
                    b
                    for bi, b in enumerate(msg.get("content", []))
                    if bi not in drop_blocks_by_msg[i]
                ]
                if not stripped:
                    stripped = [_make_placeholder_block()]
                out.append({**msg, "content": stripped})
            else:
                out.append(msg)
        return out
