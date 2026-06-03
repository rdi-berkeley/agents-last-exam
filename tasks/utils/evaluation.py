import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

if TYPE_CHECKING:
    from cua_bench.computers.base import DesktopSession

logger = logging.getLogger(__name__)


def eval_credentials_dir() -> Path:
    """Directory of evaluator-side per-service env files (`<repo>/secret/eval_time`).
    Override with ``AGENTHLE_EVAL_CREDENTIALS_DIR``. This module lives at
    ``<repo>/tasks/utils/evaluation.py`` so ``parents[2]`` is the repo root."""
    env = os.environ.get("AGENTHLE_EVAL_CREDENTIALS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "secret" / "eval_time"


def load_eval_env(*, override: bool = True) -> list[str]:
    """Export every ``secret/eval_time/*.env`` (one file per service) into
    ``os.environ`` so evaluator judges/scorers find their keys no matter which
    runner launched the eval — no fragile manual ``source`` step.

    ``override=True`` by default: ``secret/eval_time/*.env`` is the authoritative
    source for *evaluator-side* service keys and MUST win over whatever the run's
    ``secret_file`` (e.g. ``secret/.env``, loaded with override) put into
    ``os.environ``. Otherwise a stale ``OPENAI_API_KEY`` in ``secret/.env`` would
    permanently shadow the per-service eval key and judges would auth with the
    wrong credential. Pass ``override=False`` only to fill gaps without clobbering.

    Only real ``*.env`` files are read (``.example`` templates are skipped).
    Idempotent; returns the variable names it set."""
    d = eval_credentials_dir()
    if not d.is_dir():
        return []
    loaded: list[str] = []
    for f in sorted(d.glob("*.env")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, val = line.partition("=")
            if not sep:
                continue
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
                val = val[1:-1]
            if key and (override or key not in os.environ):
                os.environ[key] = val
                loaded.append(key)
    if loaded:
        logger.info("load_eval_env: set %d var(s) from %s", len(loaded), d)
    return loaded


# Make evaluator credentials available regardless of entrypoint (simprun / ale_run
# / tests / ad-hoc): load the lumped dev .env (back-compat) then every per-service
# secret/eval_time/*.env. Runs once at import, before the DEFAULT_* reads below.
load_dotenv(override=False)
load_eval_env()

DEFAULT_LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "gpt-5.4")
DEFAULT_GEMINI_VIDEO_JUDGE_MODEL = os.environ.get(
    "GEMINI_EVAL_MODEL",
    # NOTE: do not pin a soon-to-be-retired preview build. `gemini-3-pro-preview`
    # was retired by Google and started returning 404 NOT_FOUND, which silently
    # zeroed every run. Track the current pro tier and override via env if needed.
    os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
)


def resolve_llm_judge_model(*, env_var: str | None = None, default: str | None = None) -> str:
    """Resolve the model used by LLM-based judges.

    Priority:
    1. task-specific environment variable, if provided
    2. explicit default passed by the caller
    3. shared `LLM_JUDGE_MODEL`
    4. hard-coded repository fallback
    """
    if env_var:
        value = os.environ.get(env_var)
        if value:
            return value
    if default:
        return default
    return DEFAULT_LLM_JUDGE_MODEL


def _resolve_client_kwargs(api_key: str | None = None) -> dict[str, str]:
    load_eval_env()  # ensure secret/eval_time/*.env is loaded at the point creds are read
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    kwargs = {"api_key": resolved_key}
    api_base = os.environ.get("OPENAI_API_BASE")
    if api_base:
        kwargs["base_url"] = api_base
    return kwargs


def build_vision_content(
    prompt: str,
    image_bytes_list: Sequence[bytes],
    *,
    image_media_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Build multimodal user content from text plus one or more images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_bytes in image_bytes_list:
        payload_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{payload_b64}"},
            }
        )
    return content


def build_vision_image_content(
    image_bytes_list: Sequence[bytes],
    *,
    image_media_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Build multimodal content containing only images."""
    content: list[dict[str, Any]] = []
    for image_bytes in image_bytes_list:
        payload_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{payload_b64}"},
            }
        )
    return content


def _content_with_response_format_candidates(
    *, response_format: Optional[dict[str, Any]]
) -> list[Optional[dict[str, Any]]]:
    return [response_format, None] if response_format is not None else [None]


def _extract_message_text(response: Any) -> str:
    return (response.choices[0].message.content or "").strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("Judge returned an empty response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Judge response is not valid JSON: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Judge JSON payload must be an object")
    return parsed


def _extract_genai_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                chunks.append(part_text)
    return "".join(chunks).strip()


def _resolve_gemini_auth(
    *,
    api_key: str | None = None,
    auth_mode: str | None = None,
) -> dict[str, Any]:
    load_eval_env()  # ensure secret/eval_time/*.env is loaded at the point creds are read
    raw_mode = (auth_mode or os.environ.get("GEMINI_AUTH_MODE") or "auto").strip().lower()
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )
    location = (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or os.environ.get("GOOGLE_VERTEX_LOCATION")
        or "us-central1"
    )

    if raw_mode in {"auto", ""}:
        if key:
            raw_mode = "developer_api"
        elif project:
            raw_mode = "vertex_ai"
        else:
            raise RuntimeError(
                "No Gemini credentials found: set GOOGLE_API_KEY/GEMINI_API_KEY "
                "or GOOGLE_CLOUD_PROJECT for Vertex AI"
            )

    if raw_mode in {"developer", "developer_api", "google_ai", "api_key"}:
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini developer API auth"
            )
        return {
            "client_kwargs": {"api_key": key},
            "auth_mode": "developer_api",
            "project": None,
            "location": None,
        }

    if raw_mode in {"vertex", "vertex_ai", "google_cloud", "adc"}:
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for Gemini Vertex AI auth")
        return {
            "client_kwargs": {"vertexai": True, "project": project, "location": location},
            "auth_mode": "vertex_ai",
            "project": project,
            "location": location,
        }

    raise RuntimeError(f"Unsupported GEMINI_AUTH_MODE: {auth_mode!r}")


def _normalize_binary_response(raw: str) -> str | None:
    normalized = re.sub(r"\s+", " ", raw.strip()).upper()
    if normalized in {"YES", "PASS"}:
        return "YES"
    if normalized in {"NO", "NO PASS"}:
        return "NO"
    return None


def _binary_score_from_response(raw: str) -> float:
    return 1.0 if _normalize_binary_response(raw) == "YES" else 0.0


def _vision_token_param_candidates(client: AsyncOpenAI, max_tokens: int) -> list[dict]:
    """Compat shim for different OpenAI SDK/backends.

    Some local SDK versions only accept ``max_tokens``.
    Some newer model backends reject ``max_tokens`` and require
    ``max_completion_tokens``.
    We try both in a deterministic order based on the local client signature,
    then fall back to the alternate spelling on error.
    """
    create = client.chat.completions.create
    code = getattr(create, "__code__", None)
    varnames = set(code.co_varnames if code else ())
    preferred = (
        ["max_completion_tokens", "max_tokens"]
        if "max_completion_tokens" in varnames
        else ["max_tokens", "max_completion_tokens"]
    )
    return [{name: max_tokens} for name in preferred]


def _sync_token_param_candidates(client: OpenAI, max_tokens: int) -> list[dict[str, int]]:
    create = client.chat.completions.create
    code = getattr(create, "__code__", None)
    varnames = set(code.co_varnames if code else ())
    preferred = (
        ["max_completion_tokens", "max_tokens"]
        if "max_completion_tokens" in varnames
        else ["max_tokens", "max_completion_tokens"]
    )
    return [{name: max_tokens} for name in preferred]


async def llm_multimodal_text(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Shared async multimodal completion entry for LLM judges."""
    client = AsyncOpenAI(**_resolve_client_kwargs(api_key))
    resolved_model = model or resolve_llm_judge_model()
    last_exc: Exception | None = None
    response = None

    for format_kwargs in _content_with_response_format_candidates(response_format=response_format):
        extra_kwargs: dict[str, Any] = {}
        if temperature is not None:
            extra_kwargs["temperature"] = temperature
        if format_kwargs is not None:
            extra_kwargs["response_format"] = format_kwargs

        for token_kwargs in _vision_token_param_candidates(client, max_tokens):
            try:
                response = await client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": content}],
                    **extra_kwargs,
                    **token_kwargs,
                )
                return _extract_message_text(response)
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                msg = str(exc)
                if any(
                    marker in msg
                    for marker in (
                        "max_tokens",
                        "max_completion_tokens",
                        "unsupported_parameter",
                        "response_format",
                    )
                ):
                    last_exc = exc
                    continue
                raise

    raise last_exc or RuntimeError("multimodal completion request failed")


def llm_multimodal_text_sync(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Shared sync multimodal completion entry for local LLM judges."""
    client = OpenAI(**_resolve_client_kwargs(api_key))
    resolved_model = model or resolve_llm_judge_model()
    last_exc: Exception | None = None

    for format_kwargs in _content_with_response_format_candidates(response_format=response_format):
        extra_kwargs: dict[str, Any] = {}
        if temperature is not None:
            extra_kwargs["temperature"] = temperature
        if format_kwargs is not None:
            extra_kwargs["response_format"] = format_kwargs

        for token_kwargs in _sync_token_param_candidates(client, max_tokens):
            try:
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": content}],
                    **extra_kwargs,
                    **token_kwargs,
                )
                return _extract_message_text(response)
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception as exc:
                msg = str(exc)
                if any(
                    marker in msg
                    for marker in (
                        "max_tokens",
                        "max_completion_tokens",
                        "unsupported_parameter",
                        "response_format",
                    )
                ):
                    last_exc = exc
                    continue
                raise

    raise last_exc or RuntimeError("multimodal completion request failed")


async def llm_multimodal_json(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    raw = await llm_multimodal_text(
        content=content,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        response_format={"type": "json_object"},
    )
    return _extract_json_object(raw)


def llm_multimodal_json_sync(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    raw = llm_multimodal_text_sync(
        content=content,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        response_format={"type": "json_object"},
    )
    return _extract_json_object(raw)


async def llm_vision_judge(
    prompt: str,
    image_bytes: bytes,
    reference_image_bytes: Optional[bytes] = None,
    model: str = DEFAULT_LLM_JUDGE_MODEL,
    max_tokens: int = 2048,
    return_binary_score: bool = False,
    api_key: Optional[str] = None,
    return_details: bool = False,
    eval_context: Optional["EvaluationContext"] = None,
    identifier: Optional[str] = None,
) -> Union[str, float, dict]:
    """
    General-purpose LLM vision evaluation function supporting both single and dual image modes.

    Args:
        prompt: The question or instruction to send to the LLM
        image_bytes: Primary image to evaluate (required)
        reference_image_bytes: Optional reference image for comparison mode.
                              If provided, the LLM will see both images.
        model: OpenAI model to use (default: "gpt-5.4")
        max_tokens: Maximum tokens for the response
        return_binary_score: If True, parses response for YES/NO and returns 1.0/0.0.
                            If False, returns the raw text response.
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY from environment.
        return_details: If True, returns a dict with full details including VLM response,
                       score, prompt, model, etc. Overrides return_binary_score.
        eval_context: Optional EvaluationContext for automatic logging. When provided,
                     the result will be automatically logged to the context.
        identifier: Identifier for logging (required if eval_context is provided)

    Returns:
        - dict with full evaluation details if return_details=True
        - float (0.0-1.0) if return_binary_score=True
        - str with LLM response otherwise
    """
    result = None
    error_msg = None
    resolved_model = resolve_llm_judge_model(default=model)

    try:
        images = [image_bytes]
        mode = "single"
        if reference_image_bytes is not None:
            images.append(reference_image_bytes)
            mode = "comparison"

        answer = await llm_multimodal_text(
            content=build_vision_content(prompt, images),
            model=resolved_model,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        logger.info(f"LLM vision judge ({mode} mode): {answer}")

        # Calculate score if needed via the shared binary parser.
        score = (
            _binary_score_from_response(answer)
            if (return_binary_score or return_details or eval_context)
            else None
        )

        result = {
            "vlm_response": answer,
            "score": score,
            "prompt": prompt,
            "model": resolved_model,
            "mode": mode,
            "max_tokens": max_tokens,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error in llm_vision_judge: {e}")
        error_msg = f"Error: {str(e)}"
        mode = "comparison" if reference_image_bytes else "single"

        result = {
            "vlm_response": None,
            "score": 0.0,
            "prompt": prompt,
            "model": resolved_model,
            "mode": mode,
            "max_tokens": max_tokens,
            "error": error_msg,
        }

    # Auto-log to EvaluationContext if provided
    if eval_context is not None and identifier is not None:
        eval_context.log_evaluation(
            identifier=identifier,
            score=result["score"],
            vlm_response=result["vlm_response"],
            prompt=result["prompt"],
            model=result["model"],
            error=result["error"],
        )

    # Return based on requested format
    if return_details:
        return result
    elif return_binary_score:
        return result["score"]
    else:
        return result["vlm_response"] if result["vlm_response"] else error_msg


async def llm_vision_yes_no_judge(
    *,
    prompt: str,
    image_bytes: bytes,
    reference_image_bytes: Optional[bytes] = None,
    model: str | None = None,
    max_tokens: int = 10,
    api_key: Optional[str] = None,
    return_details: bool = True,
    eval_context: Optional["EvaluationContext"] = None,
    identifier: Optional[str] = None,
) -> Union[float, dict]:
    """Explicit YES/NO vision judge wrapper with shared defaults."""
    return await llm_vision_judge(
        prompt=prompt,
        image_bytes=image_bytes,
        reference_image_bytes=reference_image_bytes,
        model=resolve_llm_judge_model(default=model),
        max_tokens=max_tokens,
        return_binary_score=not return_details,
        api_key=api_key,
        return_details=return_details,
        eval_context=eval_context,
        identifier=identifier,
    )


async def llm_vision_json_judge(
    *,
    prompt: str,
    image_bytes_list: Sequence[bytes],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Async JSON-returning vision judge using the shared multimodal entry."""
    return await llm_multimodal_json(
        content=build_vision_content(prompt, image_bytes_list),
        model=resolve_llm_judge_model(default=model),
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )


async def gemini_video_json_judge(
    *,
    prompt: str,
    video_bytes: bytes,
    model: str | None = None,
    api_key: str | None = None,
    auth_mode: str | None = None,
    temperature: float = 0,
    # gemini-3.x spends "thinking" tokens before the JSON body; 2048 truncated
    # the response mid-object and broke JSON parsing (→ ok=False → score 0).
    max_tokens: int = 8192,
    video_mime_type: str = "video/mp4",
) -> dict[str, Any]:
    """JSON-returning Gemini video judge using google-genai.

    The return shape mirrors the historical task contract used by
    media/video_reconstruction: successful calls include ``parsed`` and
    ``raw_text``; failures return a structured payload with ``ok=False`` (and an
    ``error`` string) instead of raising, so imports never crash.

    IMPORTANT: ``ok=False`` means the judge itself failed (retired model, auth,
    truncated/invalid JSON, network) — it does NOT mean the video scored zero.
    Callers must distinguish the two: treat ``ok=False`` as an evaluation error
    (fail/retry the run), not as a legitimate 0.0 score, or a broken judge will
    silently zero every submission including a correct answer.
    """
    resolved_model = model or DEFAULT_GEMINI_VIDEO_JUDGE_MODEL
    resolved_auth_mode: str | None = None
    project: str | None = None
    location: str | None = None

    def _call_gemini() -> dict[str, Any]:
        nonlocal resolved_auth_mode, project, location
        from google import genai
        from google.genai import types

        auth = _resolve_gemini_auth(api_key=api_key, auth_mode=auth_mode)
        resolved_auth_mode = auth["auth_mode"]
        project = auth["project"]
        location = auth["location"]
        client = genai.Client(**auth["client_kwargs"])
        response = client.models.generate_content(
            model=resolved_model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=video_bytes, mime_type=video_mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                maxOutputTokens=max_tokens,
                responseMimeType="application/json",
            ),
        )
        raw_text = _extract_genai_text(response)
        try:
            parsed = _extract_json_object(raw_text)
        except Exception as exc:
            return {
                "ok": False,
                "parsed": None,
                "raw_text": raw_text,
                "model_used": resolved_model,
                "auth_mode": resolved_auth_mode,
                "project": project,
                "location": location,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "ok": True,
            "parsed": parsed,
            "raw_text": raw_text,
            "model_used": resolved_model,
            "auth_mode": resolved_auth_mode,
            "project": project,
            "location": location,
            "error": None,
        }

    try:
        return await asyncio.to_thread(_call_gemini)
    except Exception as exc:
        logger.error("Error in gemini_video_json_judge: %s", exc)
        return {
            "ok": False,
            "parsed": None,
            "raw_text": "",
            "model_used": resolved_model,
            "auth_mode": resolved_auth_mode
            or auth_mode
            or os.environ.get("GEMINI_AUTH_MODE", "auto"),
            "project": project,
            "location": location,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_binary_answer(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
    text = str(value or "").strip().upper()
    if text in {"YES", "Y", "TRUE", "1", "PASS"}:
        return 1.0
    if text in {"NO", "N", "FALSE", "0", "FAIL"}:
        return 0.0
    return 0.0


async def llm_vision_binary_checklist_judge(
    *,
    prompt_intro: str,
    checklist_items: Sequence[tuple[str, str]],
    image_bytes: bytes,
    reference_image_bytes: Optional[bytes] = None,
    model: str | None = None,
    max_tokens: int = 512,
    api_key: Optional[str] = None,
    return_details: bool = True,
    eval_context: Optional["EvaluationContext"] = None,
    identifier: Optional[str] = None,
) -> Union[float, dict[str, Any]]:
    """Vision judge that scores a checklist of binary YES/NO items and averages them."""
    if not checklist_items:
        raise ValueError("checklist_items must not be empty")

    lines = [
        prompt_intro.rstrip(),
        "",
        "Return a JSON object with this exact shape:",
        '{"answers": {"item_key": "YES or NO"}, "summary": "one short sentence"}',
        "",
        "Checklist items:",
    ]
    for key, question in checklist_items:
        lines.append(f'- "{key}": {question}')
    lines.extend(
        [
            "",
            "Rules:",
            '- Every checklist answer must be exactly "YES" or "NO".',
            "- Do not use fractional or numeric scores.",
            "- If you are unsure, answer NO for that checklist item.",
        ]
    )
    prompt = "\n".join(lines)

    image_bytes_list = [image_bytes]
    mode = "single"
    if reference_image_bytes is not None:
        image_bytes_list.append(reference_image_bytes)
        mode = "comparison"

    try:
        raw = await llm_vision_json_judge(
            prompt=prompt,
            image_bytes_list=image_bytes_list,
            model=resolve_llm_judge_model(default=model),
            max_tokens=max_tokens,
            temperature=0,
            api_key=api_key,
        )
        answers_payload = raw.get("answers", {}) if isinstance(raw, dict) else {}
        checklist_scores: dict[str, float] = {}
        normalized_answers: dict[str, str] = {}
        for key, _question in checklist_items:
            score = _parse_binary_answer(answers_payload.get(key, "NO"))
            checklist_scores[key] = score
            normalized_answers[key] = "YES" if score >= 1.0 else "NO"
        score = sum(checklist_scores.values()) / len(checklist_items)
        result = {
            "vlm_response": json.dumps(raw, ensure_ascii=False),
            "score": score,
            "prompt": prompt,
            "model": resolve_llm_judge_model(default=model),
            "mode": mode,
            "max_tokens": max_tokens,
            "error": None,
            "checklist_answers": normalized_answers,
            "checklist_scores": checklist_scores,
            "summary": raw.get("summary") if isinstance(raw, dict) else None,
        }
    except Exception as e:
        logger.error(f"Error in llm_vision_binary_checklist_judge: {e}")
        result = {
            "vlm_response": None,
            "score": 0.0,
            "prompt": prompt,
            "model": resolve_llm_judge_model(default=model),
            "mode": mode,
            "max_tokens": max_tokens,
            "error": f"Error: {str(e)}",
            "checklist_answers": {key: "NO" for key, _question in checklist_items},
            "checklist_scores": {key: 0.0 for key, _question in checklist_items},
            "summary": None,
        }

    if eval_context is not None and identifier is not None:
        eval_context.log_evaluation(
            identifier=identifier,
            score=result["score"],
            vlm_response=result["vlm_response"],
            prompt=result["prompt"],
            model=result["model"],
            error=result["error"],
            checklist_answers=result["checklist_answers"],
            checklist_scores=result["checklist_scores"],
            summary=result["summary"],
        )

    if return_details:
        return result
    return result["score"]


def llm_vision_json_judge_sync(
    *,
    prompt: str,
    image_bytes_list: Sequence[bytes],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Sync JSON-returning vision judge using the shared multimodal entry."""
    return llm_multimodal_json_sync(
        content=build_vision_content(prompt, image_bytes_list),
        model=resolve_llm_judge_model(default=model),
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )


def llm_multimodal_binary_questions_sync(
    *,
    prompt_context: str,
    questions: Sequence[str],
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 32,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a family-specific set of YES / NO multimodal questions."""
    if not questions:
        raise ValueError("questions must not be empty")

    resolved_model = resolve_llm_judge_model(default=model)
    results: list[dict[str, Any]] = []
    raw_responses: list[str] = []

    for index, question in enumerate(questions, start=1):
        raw_response = llm_multimodal_text_sync(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"{prompt_context.strip()}\n\n"
                        f"Question {index}/{len(questions)}: {question}\n"
                        "Respond with ONLY YES or NO."
                    ),
                },
                *content,
            ],
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
        )
        parsed = _normalize_binary_response(raw_response) or "NO"
        score = 1.0 if parsed == "YES" else 0.0
        raw_responses.append(raw_response)
        results.append(
            {
                "question": question,
                "result": parsed,
                "score": score,
                "raw_response": raw_response,
            }
        )

    yes_count = sum(1 for item in results if item["result"] == "YES")
    question_count = len(results)
    final_score = float(yes_count / question_count)
    return {
        "results": results,
        "yes_count": yes_count,
        "no_count": question_count - yes_count,
        "question_count": question_count,
        "final_score": final_score,
        "raw_responses": raw_responses,
    }


def llm_vision_binary_questions_sync(
    *,
    prompt_context: str,
    questions: Sequence[str],
    image_bytes_list: Sequence[bytes],
    model: str | None = None,
    max_tokens: int = 32,
    temperature: float = 0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a family-specific set of YES / NO questions over images."""
    return llm_multimodal_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        content=build_vision_image_content(image_bytes_list),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )


async def llm_vision_judge_single(
    prompt: str,
    image_bytes: bytes,
    eval_context: Optional["EvaluationContext"] = None,
    identifier: Optional[str] = None,
    **kwargs,
) -> Union[str, float, dict]:
    """Simplified single-image LLM vision evaluation function."""
    return await llm_vision_judge(
        prompt=prompt,
        image_bytes=image_bytes,
        reference_image_bytes=None,
        eval_context=eval_context,
        identifier=identifier,
        **kwargs,
    )


async def compare_screenshots_game(
    target_image_bytes: bytes,
    reference_image_bytes: bytes,
    context_description: str,
    comparison_criteria: Optional[str] = None,
) -> dict:
    """
    Compare target and reference screenshots using VLM.

    Args:
        target_image_bytes: The screenshot to evaluate
        reference_image_bytes: The reference screenshot
        context_description: Description of what's being compared (e.g., "floor 3")
        comparison_criteria: Optional additional criteria for comparison

    Returns:
        Dictionary with evaluation details (score, vlm_response, prompt, etc.)
    """
    criteria = comparison_criteria or ""

    prompt = f"""You are evaluating a game screenshot.

Compare these two images:
1. First image: A screenshot from the agent's playthrough
2. Second image: A reference screenshot showing the correct state ({context_description})

Question: Does the first image show that the player has successfully reached the same state as the reference image for {context_description}?

Please analyze:
{criteria}

Answer with ONLY "YES" or "NO"."""

    return await llm_vision_judge(
        prompt=prompt,
        image_bytes=target_image_bytes,
        reference_image_bytes=reference_image_bytes,
        return_details=True,
        max_tokens=10,
    )


async def collect_matching_files(
    session: "DesktopSession", target_path: str, reference_path: str
) -> tuple[list[str], list[str]]:
    """
    Collect files from target and reference directories.

    Args:
        session: Desktop session for file operations
        target_path: Path to target directory
        reference_path: Path to reference directory

    Returns:
        Tuple of (target_files, reference_files)
    """
    target_files = await session.list_dir(target_path)
    reference_files = await session.list_dir(reference_path)
    return target_files, reference_files


def save_evaluation_results(
    evaluation_details: dict, task_tag: str, output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Save evaluation results to a JSON file.

    Args:
        evaluation_details: Dictionary containing all evaluation details
        task_tag: Tag identifying the task
        output_dir: Optional directory to save results (defaults to ./trycua/cua-bench/)

    Returns:
        Path to saved JSON file, or None if saving failed
    """
    try:
        output_dir = output_dir or os.environ.get("EVALUATION_OUTPUT_DIR", "./trycua/cua-bench/")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{task_tag}_evaluation_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)

        def _json_default(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    pass
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except TypeError:
                    pass
            raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

        payload = json.dumps(
            evaluation_details,
            indent=2,
            default=_json_default,
        )
        with open(json_filepath, "w", encoding="utf-8") as f:
            f.write(payload)

        logger.info(f"Evaluation details saved to: {json_filepath}")
        return json_filepath
    except Exception as e:
        logger.error(f"Failed to save evaluation details to JSON: {e}")
        return None


class EvaluationContext:
    """
    Context manager for tracking and logging evaluation results automatically.

    Usage:
        async with EvaluationContext(task_tag="my_task", mode="custom") as ctx:
            for file in files:
                result = await llm_vision_judge(...)
                ctx.log_evaluation(
                    identifier=file,
                    score=result["score"],
                    vlm_response=result["vlm_response"],
                    # ... any additional fields
                )
                ctx.add_score(result["score"] * weight)

            return [ctx.get_final_score(num_items=len(files))]
    """

    def __init__(
        self,
        task_tag: str,
        mode: str = "custom",
        output_dir: Optional[str] = None,
        auto_save: bool = True,
        **extra_metadata,
    ):
        """
        Initialize evaluation context.

        Args:
            task_tag: Identifier for the task
            mode: Evaluation mode name (e.g., "milestone", "custom", "deliverable")
            output_dir: Directory for saving results
            auto_save: Whether to automatically save results on context exit
            **extra_metadata: Additional metadata to include in evaluation details
        """
        self.task_tag = task_tag
        self.mode = mode
        self.output_dir = output_dir
        self.auto_save = auto_save

        self.evaluation_details = {
            "mode": mode,
            "task_tag": task_tag,
            "timestamp": datetime.now().isoformat(),
            "evaluations": [],
            **extra_metadata,
        }
        self._total_score = 0.0
        self._num_evaluated = 0
        self._finalized = False

    def log_evaluation(
        self,
        identifier: str,
        score: float,
        vlm_response: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        error: Optional[str] = None,
        **extra_fields,
    ) -> None:
        """
        Log a single evaluation result with automatic logging.

        Args:
            identifier: Unique identifier for this evaluation (e.g., filename, milestone)
            score: Score for this evaluation (0.0-1.0)
            vlm_response: Optional VLM response text
            prompt: Optional prompt used
            model: Optional model name
            error: Optional error message
            **extra_fields: Any additional fields to store
        """
        eval_entry = {
            "identifier": identifier,
            "score": score,
            "vlm_response": vlm_response,
            "prompt": prompt,
            "model": model,
            "error": error,
            **extra_fields,
        }
        # Remove None values for cleaner output
        eval_entry = {k: v for k, v in eval_entry.items() if v is not None}

        self.evaluation_details["evaluations"].append(eval_entry)
        self._num_evaluated += 1

        # Automatic logging
        if vlm_response:
            logger.info(f"Identifier '{identifier}' VLM response: {vlm_response}")
        logger.info(f"Identifier '{identifier}' judgment score: {score}")
        if error:
            logger.error(f"Identifier '{identifier}' error: {error}")

    def log_error(self, identifier: str, error: Union[str, Exception], score: float = 0.0) -> None:
        """Log an evaluation error."""
        error_msg = str(error)
        self.log_evaluation(identifier=identifier, score=score, error=error_msg)
        logger.error(f"Error evaluating identifier '{identifier}': {error_msg}")

    def add_score(self, score: float) -> None:
        """Add to the cumulative total score."""
        self._total_score += score

    def get_final_score(self, num_items: Optional[int] = None) -> float:
        """
        Get the final normalized score.

        Args:
            num_items: If provided, divides total_score by this number.
                      If None, returns raw total_score.
        """
        if num_items and num_items > 0:
            return self._total_score / num_items
        return self._total_score

    @property
    def total_score(self) -> float:
        """Get the raw cumulative score."""
        return self._total_score

    @property
    def num_evaluated(self) -> int:
        """Get the number of evaluations logged."""
        return self._num_evaluated

    def finalize(self, **extra_summary) -> tuple[float, dict]:
        """
        Finalize evaluation, add summary, and save results.

        Args:
            **extra_summary: Additional fields to include in summary

        Returns:
            Tuple of (final_score, evaluation_details)
        """
        if self._finalized:
            return self._total_score, self.evaluation_details

        self.evaluation_details["summary"] = {
            "total_score": self._total_score,
            "num_evaluated": self._num_evaluated,
            **extra_summary,
        }

        logger.info(
            f"Evaluation complete. Total score: {self._total_score} ({self._num_evaluated} evaluated)"
        )

        if self.auto_save:
            save_evaluation_results(self.evaluation_details, self.task_tag, self.output_dir)

        self._finalized = True
        return self._total_score, self.evaluation_details

    async def __aenter__(self) -> "EvaluationContext":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit - auto-finalize on success."""
        if exc_type is None and not self._finalized:
            self.finalize()
        return False

    def __enter__(self) -> "EvaluationContext":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Sync context manager exit - auto-finalize on success."""
        if exc_type is None and not self._finalized:
            self.finalize()
        return False


async def evaluate_milestone_mode(
    session: "DesktopSession",
    target_path: str,
    reference_path: str,
    task_tag: str,
    comparison_fn: callable,
    output_dir: Optional[str] = None,
) -> tuple[float, dict]:
    """
    Evaluate using milestone mode: compare agent-saved screenshots with references.
    """
    # Check if target directory exists
    exists = (await session.file_exists(target_path) or await session.directory_exists(target_path))
    if not exists:
        logger.info(f"Evaluation: File NOT found at {target_path}")
        return 0.0, {"error": f"Target path not found: {target_path}"}

    # Collect files
    target_files, reference_files = await collect_matching_files(
        session, target_path, reference_path
    )

    async with EvaluationContext(
        task_tag=task_tag,
        mode="milestone",
        output_dir=output_dir,
        target_path=target_path,
        reference_path=reference_path,
    ) as ctx:
        # Evaluate matching files
        for file in target_files:
            if file in reference_files:
                try:
                    target_file_path = os.path.join(target_path, file)
                    reference_file_path = os.path.join(reference_path, file)
                    identifier = os.path.splitext(file)[0]

                    logger.info(f"Evaluating milestone: {file}")

                    # Download images from remote server
                    target_image_bytes = await session.read_bytes(target_file_path)
                    reference_image_bytes = await session.read_bytes(reference_file_path)

                    # Compare screenshots
                    eval_result = await comparison_fn(
                        target_image_bytes, reference_image_bytes, identifier
                    )

                    score = eval_result["score"]
                    ctx.log_evaluation(
                        identifier=identifier,
                        score=score,
                        vlm_response=eval_result["vlm_response"],
                        prompt=eval_result["prompt"],
                        model=eval_result["model"],
                        mode=eval_result["mode"],
                        error=eval_result["error"],
                        target_file_path=target_file_path,
                        reference_file_path=reference_file_path,
                        file=file,
                    )
                    ctx.add_score(score / len(reference_files))

                except Exception as e:
                    ctx.log_error(identifier=file, error=e)

        return ctx.finalize(
            num_reference_files=len(reference_files), num_target_files=len(target_files)
        )


async def evaluate_deliverable_mode(
    session: "DesktopSession",
    trajectory_dir: str,
    reference_path: str,
    task_tag: str,
    comparison_fn: callable,
    screenshot_points: list[int],
    action_delay: float = 0.5,
    output_dir: Optional[str] = None,
) -> tuple[float, dict]:
    """
    Evaluate using deliverable mode: replay trajectory and take screenshots at specified points.
    """
    async with EvaluationContext(
        task_tag=task_tag,
        mode="deliverable",
        output_dir=output_dir,
        trajectory_dir=str(trajectory_dir),
        reference_path=reference_path,
        screenshot_points=screenshot_points,
    ) as ctx:
        try:
            # Get reference files to know what to compare
            reference_files = await session.list_dir(reference_path)

            # Replay trajectory with screenshots at specified points
            logger.info(f"Replaying trajectory from: {trajectory_dir}")

            import json
            from pathlib import Path

            trajectory_path = Path(trajectory_dir)
            if not trajectory_path.exists():
                raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

            # Find latest agent response file
            response_files = sorted(trajectory_path.rglob("*_agent_response.json"))
            if not response_files:
                raise ValueError(f"No agent_response.json files found in {trajectory_dir}")

            latest_response_file = response_files[-1]
            logger.info(f"Using trajectory file: {latest_response_file.name}")

            # Load and extract actions
            with open(latest_response_file, "r") as f:
                data = json.load(f)

            messages = data.get("kwargs", {}).get("messages", [])
            actions_to_execute = []
            for item in messages:
                if isinstance(item, dict) and item.get("type") == "computer_call":
                    action = item.get("action", {})
                    action_type = action.get("type")
                    if action_type and action_type != "screenshot":
                        actions_to_execute.append(action)

            logger.info(f"Found {len(actions_to_execute)} actions to replay")

            # Import computer handler
            from agent.computers import cuaComputerHandler

            handler = cuaComputerHandler(session._computer)
            await handler._initialize()

            # Replay actions and take screenshots at specified points
            screenshots_taken = {}

            for i, action in enumerate(actions_to_execute):
                action_type = action.get("type")
                action_args = {k: v for k, v in action.items() if k != "type"}

                logger.info(
                    f"[{i+1}/{len(actions_to_execute)}] Executing: {action_type}({action_args})"
                )

                method = getattr(handler, action_type, None)
                if method:
                    try:
                        await method(**action_args)
                    except Exception as e:
                        logger.error(f"Action {action_type} failed: {e}")

                # Take screenshot if at a screenshot point
                if i + 1 in screenshot_points:
                    try:
                        screenshot_bytes = await session.screenshot()
                        # Map this screenshot to corresponding reference file
                        point_index = screenshot_points.index(i + 1)
                        if point_index < len(reference_files):
                            identifier = os.path.splitext(reference_files[point_index])[0]
                            screenshots_taken[identifier] = screenshot_bytes
                            logger.info(
                                f"Screenshot taken at action {i+1} for identifier '{identifier}'"
                            )
                    except Exception as e:
                        logger.error(f"Failed to take screenshot at action {i+1}: {e}")

                await asyncio.sleep(action_delay)

            # Now compare screenshots with references
            for ref_file in reference_files:
                identifier = os.path.splitext(ref_file)[0]

                if identifier in screenshots_taken:
                    try:
                        reference_file_path = os.path.join(reference_path, ref_file)
                        reference_image_bytes = await session.read_bytes(reference_file_path)
                        target_image_bytes = screenshots_taken[identifier]

                        logger.info(f"Evaluating deliverable: {identifier}")

                        # Compare screenshots
                        eval_result = await comparison_fn(
                            target_image_bytes, reference_image_bytes, identifier
                        )

                        score = eval_result["score"]
                        ctx.log_evaluation(
                            identifier=identifier,
                            score=score,
                            vlm_response=eval_result["vlm_response"],
                            prompt=eval_result["prompt"],
                            model=eval_result["model"],
                            mode=eval_result["mode"],
                            error=eval_result["error"],
                            reference_file=ref_file,
                            reference_file_path=reference_file_path,
                        )
                        ctx.add_score(score / len(reference_files))

                    except Exception as e:
                        ctx.log_error(identifier=identifier, error=e)
                else:
                    ctx.log_evaluation(
                        identifier=identifier,
                        score=0.0,
                        error="No screenshot taken at corresponding point",
                    )

            return ctx.finalize(
                num_reference_files=len(reference_files),
                num_screenshots_taken=len(screenshots_taken),
                total_actions_replayed=len(actions_to_execute),
            )

        except Exception as e:
            logger.error(f"Error in deliverable evaluation: {e}")
            ctx.evaluation_details["error"] = str(e)
            return ctx.finalize(error=str(e))
