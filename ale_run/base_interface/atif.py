"""Convert ALE's internal Trajectory model to ATIF-v1.8 for serialization.

ALE's internal model (see :mod:`ale_run.base_interface.trajectory`) uses
ATIF's own field names directly (``tool_call_id``, ``function_name``,
``reasoning_content``, ``prompt_tokens``, ...), so this is mostly a
pass-through. Two structural gaps remain, both because ALE's model is
intentionally a little more permissive than strict ATIF:

- ALE's ``source`` includes ``"environment"`` (a separate step carrying only
  an Observation, right after the agent step that issued the tool_calls).
  ATIF has no such source and requires an observation's results to resolve
  against ``tool_calls`` on the *same* step. :func:`_merge_split_observations`
  re-attaches each result to the step that actually issued the matching
  tool call, dropping the now-empty carrier step.
- ATIF forbids ``metrics`` on non-``"agent"`` steps; ALE's own ``Step`` model
  already enforces this (see ``trajectory.py``'s ``_metrics_only_on_agent_steps``
  validator), so this is just a defensive drop, never expected to trigger.

See the RFC: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
"""
from __future__ import annotations

from typing import Any

ATIF_VERSION = "ATIF-v1.8"


def _convert_content(content: Any) -> Any:
    """ALE's ContentPart list -> ATIF's ContentPart list (or pass a string through)."""
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            out.append({"type": "text", "text": part.get("text") or ""})
        elif ptype == "image":
            img = part.get("source") or {}
            media_type = img.get("media_type") or "image/png"
            path = img.get("path") or img.get("url")
            if not path and img.get("data"):
                path = f"data:{media_type};base64,{img['data']}"
            out.append({
                "type": "image",
                "source": {"media_type": media_type, "path": path or ""},
            })
        # other/unknown part types dropped -- not emitted by any deployer.
    return out


def _convert_tool_call(tc: dict) -> dict:
    out = {
        "tool_call_id": tc.get("tool_call_id") or "",
        "function_name": tc.get("function_name") or "",
        "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
    }
    if tc.get("extra"):
        out["extra"] = tc["extra"]
    return out


def _convert_observation(obs: dict) -> dict:
    results = []
    for r in obs.get("results") or []:
        out: dict[str, Any] = {}
        if r.get("source_call_id") is not None:
            out["source_call_id"] = r["source_call_id"]
        if r.get("content") is not None:
            out["content"] = _convert_content(r["content"])
        extra = dict(r.get("extra") or {})
        if r.get("is_error"):
            extra["is_error"] = True
        if extra:
            out["extra"] = extra
        results.append(out)
    return {"results": results}


def _convert_metrics(m: dict) -> dict:
    out: dict[str, Any] = {}
    if m.get("prompt_tokens") is not None:
        out["prompt_tokens"] = m["prompt_tokens"]
    if m.get("completion_tokens") is not None:
        out["completion_tokens"] = m["completion_tokens"]
    if m.get("cached_tokens") is not None:
        out["cached_tokens"] = m["cached_tokens"]
    if m.get("cost_usd") is not None:
        out["cost_usd"] = m["cost_usd"]
    extra = {}
    for k in ("cache_creation_tokens", "duration_ms"):
        if m.get(k) is not None:
            extra[k] = m[k]
    if m.get("extra"):
        extra.update(m["extra"])
    if extra:
        out["extra"] = extra
    return out


def _convert_step(s: dict) -> dict:
    """Convert one ALE step. ``step_id`` is assigned later, after merging."""
    source = s.get("source") or "agent"
    if source not in ("system", "user", "agent"):
        source = "system"  # ALE's "environment"

    out: dict[str, Any] = {
        "source": source,
        "message": _convert_content(s["message"]) if s.get("message") is not None else "",
    }
    if s.get("timestamp"):
        out["timestamp"] = s["timestamp"]
    if s.get("reasoning_content"):
        out["reasoning_content"] = s["reasoning_content"]
    if s.get("tool_calls"):
        out["tool_calls"] = [_convert_tool_call(tc) for tc in s["tool_calls"]]
    if s.get("observation"):
        out["observation"] = _convert_observation(s["observation"])
    # ATIF: metrics only valid on source="agent" steps (defensive; ALE's own
    # Step model already forbids this combination at construction time).
    if s.get("metrics") and source == "agent":
        metrics = _convert_metrics(s["metrics"])
        if metrics:
            out["metrics"] = metrics
    extra = dict(s.get("extra") or {})
    # ATIF's Observation has no top-level `error` field (only per-result
    # is_error) -- ALE's does, for an environment-level failure with no
    # specific result to attach it to. Observation forbids unknown keys, but
    # Step's own extra doesn't, so lift it there rather than drop it.
    obs_error = (s.get("observation") or {}).get("error")
    if obs_error:
        extra["observation_error"] = obs_error
    if extra:
        out["extra"] = extra
    return out


def _merge_split_observations(steps: list[dict]) -> list[dict]:
    """Re-attach each observation result to the step whose tool_calls it answers.

    ATIF requires an observation's results to resolve against tool_calls on
    the SAME step. Some deployers (e.g. openclaw_cli) instead emit the
    tool_calls on one step and the matching observation on the next step
    (source="environment", merged to "system" by :func:`_convert_step`).
    """
    call_id_to_step: dict[str, int] = {}
    for i, st in enumerate(steps):
        for tc in st.get("tool_calls") or []:
            call_id_to_step[tc["tool_call_id"]] = i

    drop = set()
    for i, st in enumerate(steps):
        obs = st.get("observation")
        if not obs:
            continue
        own_ids = {tc["tool_call_id"] for tc in st.get("tool_calls") or []}
        leftover = []
        orphan_text = []
        for result in obs.get("results") or []:
            target = result.get("source_call_id")
            if target in own_ids:
                leftover.append(result)
                continue
            if target in call_id_to_step:
                dest = steps[call_id_to_step[target]]
                dest.setdefault("observation", {}).setdefault("results", []).append(result)
                continue
            # No tool_call anywhere in the trajectory matches this result --
            # the deployer logged an error for a call it never recorded.
            # ATIF forbids an unresolvable source_call_id, so fold the
            # content into this step's message instead of dropping it.
            content = result.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        parts.append(p.get("text") or "")
                    elif p.get("type") == "image":
                        path = (p.get("source") or {}).get("path") or ""
                        parts.append(f"[image: {path}]" if path else "[image]")
                text = " ".join(parts)
            else:
                text = None
            orphan_text.append(f"[orphaned observation for {target!r}] {text or ''}".strip())
        if leftover:
            st["observation"]["results"] = leftover
        else:
            st.pop("observation", None)
        if orphan_text:
            prefix = "\n".join(orphan_text)
            existing = st.get("message")
            if isinstance(existing, str) and existing:
                st["message"] = f"{prefix}\n{existing}"
            else:
                st["message"] = prefix
        if "observation" not in st and (
            not st.get("message")
            and not st.get("reasoning_content")
            and not st.get("tool_calls")
            and not st.get("metrics")
            and not st.get("extra")
        ):
            drop.add(i)

    merged = [st for i, st in enumerate(steps) if i not in drop]
    for i, st in enumerate(merged, start=1):
        st["step_id"] = i
    return merged


def _convert_agent(agent: dict) -> dict:
    known = {"name", "version", "model", "extra"}
    out = {
        "name": agent.get("name") or "unknown",
        "version": str(agent.get("version") or "unknown"),
    }
    if agent.get("model"):
        out["model_name"] = agent["model"]
    extra = dict(agent.get("extra") or {})
    extra.update({k: v for k, v in agent.items() if k not in known})
    if extra:
        out["extra"] = extra
    return out


def _convert_final_metrics(fm: dict) -> dict:
    out: dict[str, Any] = {}
    if fm.get("total_prompt_tokens") is not None:
        out["total_prompt_tokens"] = fm["total_prompt_tokens"]
    if fm.get("total_completion_tokens") is not None:
        out["total_completion_tokens"] = fm["total_completion_tokens"]
    if fm.get("total_cached_tokens") is not None:
        out["total_cached_tokens"] = fm["total_cached_tokens"]
    if fm.get("total_cost_usd") is not None:
        out["total_cost_usd"] = fm["total_cost_usd"]
    if fm.get("total_steps") is not None:
        out["total_steps"] = fm["total_steps"]
    extra = {}
    for k in ("total_cache_creation_tokens", "total_duration_ms", "reward", "status"):
        if fm.get(k) is not None:
            extra[k] = fm[k]
    if extra:
        out["extra"] = extra
    return out


def to_atif(trajectory: Any) -> dict:
    """Convert an ALE ``Trajectory`` (model or plain dict) to an ATIF-v1.8 dict.

    Pass the result to ``json.dumps`` to get the bytes written as
    ``trajectory.json``. Accepts either a :class:`~ale_run.base_interface.
    trajectory.Trajectory` instance or its ``model_dump(mode="json")`` dict,
    so it can also be used standalone on already-written files.
    """
    d = trajectory.model_dump(mode="json") if hasattr(trajectory, "model_dump") else trajectory

    steps = [_convert_step(s) for s in (d.get("steps") or [])]
    steps = _merge_split_observations(steps)

    out: dict[str, Any] = {
        "schema_version": ATIF_VERSION,
        "agent": _convert_agent(d.get("agent") or {}),
        "steps": steps,
    }

    episode_id = d.get("episode_id")
    if episode_id:
        out["session_id"] = episode_id
        out["trajectory_id"] = episode_id

    if d.get("final_metrics"):
        fm = _convert_final_metrics(d["final_metrics"])
        if fm:
            out["final_metrics"] = fm

    if d.get("continued_trajectory_ref"):
        out["continued_trajectory_ref"] = d["continued_trajectory_ref"]

    if d.get("subagent_trajectories"):
        out["subagent_trajectories"] = [to_atif(sub) for sub in d["subagent_trajectories"]]

    root_extra = dict(d.get("extra") or {})
    for k in ("task_path", "variant_index", "instruction", "started_at", "ended_at"):
        if d.get(k) is not None:
            root_extra[k] = d[k]
    if root_extra:
        out["extra"] = root_extra

    return out
