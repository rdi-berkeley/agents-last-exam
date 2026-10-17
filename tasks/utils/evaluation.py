import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import httpx
from openai import AsyncOpenAI, BadRequestError, OpenAI
from dotenv import dotenv_values

if TYPE_CHECKING:
    from cua_bench.computers.base import DesktopSession

logger = logging.getLogger(__name__)


class JudgeInfrastructureError(RuntimeError):
    """Raised when an external judge cannot produce a valid evaluation."""


def eval_credentials_dir() -> Path:
    """Directory of evaluator-side per-service env files (`<repo>/secret/eval_time`).
    Override with ``AGENTHLE_EVAL_CREDENTIALS_DIR``. This module lives at
    ``<repo>/tasks/utils/evaluation.py`` so ``parents[2]`` is the repo root."""
    env = os.environ.get("AGENTHLE_EVAL_CREDENTIALS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "secret" / "eval_time"


def load_eval_env() -> dict[str, str]:
    """Read evaluator configuration privately; never modify the process environment.

    Real ``secret/eval_time/*.env`` files override environment values. Templates
    and solver ``.env`` files are not loaded or interpolated.
    """
    values = dict(os.environ)
    for path in sorted(eval_credentials_dir().glob("*.env")):
        values.update(
            {
                key: value
                for key, value in dotenv_values(path, interpolate=False).items()
                if value is not None
            }
        )
    return values


DEFAULT_LLM_JUDGE_MODEL = "gpt-5.4"
DEFAULT_AUDIO_JUDGE_MODEL = "gpt-audio-1.5"
DEFAULT_GEMINI_VIDEO_JUDGE_MODEL = "gemini-3.1-pro-preview"


def resolve_llm_judge_model(*, env_var: str | None = None, default: str | None = None) -> str:
    """Resolve the shared judge model at call time, with an optional explicit override."""
    values = load_eval_env()
    return (
        values.get("JUDGE_MODEL")
        or (values.get(env_var) if env_var else None)
        or default
        or DEFAULT_LLM_JUDGE_MODEL
    )


def judge_config(*, capability: str = "vision") -> dict[str, str]:
    """Return canonical judge routing with a capability-specific audio model."""
    if capability not in {"vision", "audio"}:
        raise ValueError(f"Unknown judge capability: {capability}")
    values = load_eval_env()
    return {
        "api_key": values.get("JUDGE_API_KEY", ""),
        "base_url": values.get("JUDGE_BASE_URL") or "https://api.openai.com/v1",
        "model": values.get("AUDIO_JUDGE_MODEL" if capability == "audio" else "JUDGE_MODEL")
        or (DEFAULT_AUDIO_JUDGE_MODEL if capability == "audio" else DEFAULT_LLM_JUDGE_MODEL),
        "reasoning_effort": values.get("JUDGE_REASONING_EFFORT", "")
        if capability != "audio"
        else "",
        "max_completion_tokens": values.get("JUDGE_MAX_COMPLETION_TOKENS", "0")
        if capability != "audio"
        else "0",
    }


def _resolve_client_kwargs(
    api_key: str | None = None, *, capability: str = "vision"
) -> dict[str, Any]:
    settings = judge_config(capability=capability)
    resolved_key = api_key or settings["api_key"]
    if not resolved_key:
        raise JudgeInfrastructureError("JUDGE_API_KEY is not set")
    return {
        "api_key": resolved_key,
        "base_url": settings["base_url"],
        "organization": "",
        "project": "",
        "timeout": 120,
        "max_retries": 2,
    }


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


def _extract_message_text(response: Any) -> str:
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise JudgeInfrastructureError("Judge returned an incomplete response")
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise JudgeInfrastructureError("Judge refused the scoring request")
    text = (message.content or "").strip()
    if not text:
        raise JudgeInfrastructureError("Judge returned an empty response")
    return text


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

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if not fenced:
        raise JudgeInfrastructureError("Judge response is not a JSON object")
    parsed = json.loads(fenced.group(1))
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
    values = load_eval_env()
    raw_mode = (auth_mode or values.get("GEMINI_AUTH_MODE") or "auto").strip().lower()
    key = api_key or values.get("GOOGLE_API_KEY") or values.get("GEMINI_API_KEY")
    project = (
        values.get("GOOGLE_CLOUD_PROJECT")
        or values.get("GCLOUD_PROJECT")
        or values.get("GCP_PROJECT")
    )
    location = (
        values.get("GOOGLE_CLOUD_LOCATION")
        or values.get("GOOGLE_CLOUD_REGION")
        or values.get("GOOGLE_VERTEX_LOCATION")
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
        client_kwargs = {"vertexai": True, "project": project, "location": location}
        if values.get("GOOGLE_APPLICATION_CREDENTIALS"):
            import google.auth

            credentials, _project = google.auth.load_credentials_from_file(
                values["GOOGLE_APPLICATION_CREDENTIALS"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client_kwargs["credentials"] = credentials
        return {
            "client_kwargs": client_kwargs,
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
    normalized = _normalize_binary_response(raw)
    if normalized is None:
        raise ValueError(f"Judge returned a non-binary response: {raw[:200]!r}")
    return 1.0 if normalized == "YES" else 0.0


def _judge_parameters(
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    capability: str = "vision",
    modalities: list[str] | None = None,
    store: bool | None = None,
) -> dict[str, Any]:
    settings = judge_config(capability=capability)
    effort = settings["reasoning_effort"] or reasoning_effort
    budget = max(max_tokens, int(settings["max_completion_tokens"]), 4096 if effort else 0)
    parameters: dict[str, Any] = {
        "messages": messages,
        "model": model or settings["model"],
        "max_completion_tokens": budget,
    }
    for name, value in (
        ("temperature", temperature),
        ("response_format", response_format),
        ("reasoning_effort", effort or None),
        ("modalities", modalities),
        ("store", store),
    ):
        if value is not None:
            parameters[name] = value
    return parameters


def _adapt_judge_parameters(error: BadRequestError, parameters: dict[str, Any]) -> bool:
    body = error.body if isinstance(error.body, dict) else {}
    body = body.get("error", body)
    if not isinstance(body, dict):
        return False
    message = str(body.get("message", "")).lower()
    if not any(marker in message for marker in ("unsupported", "not supported", "not support")):
        return False
    parameter = body.get("param")
    if not parameter:
        parameter = next(
            (
                name
                for name in ("temperature", "response_format", "max_completion_tokens")
                if f"'{name}'" in message or f'"{name}"' in message
            ),
            None,
        )
    if parameter == "max_completion_tokens" and parameter in parameters:
        parameters["max_tokens"] = parameters.pop(parameter)
    elif parameter in {"temperature", "response_format"} and parameter in parameters:
        parameters.pop(parameter)
    else:
        return False
    logger.info("Judge backend rejected %s; retrying with supported parameters", parameter)
    return True


async def llm_judge_completion(
    *,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    capability: str = "vision",
    transport: httpx.AsyncBaseTransport | None = None,
    event_hooks: dict[str, list[Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Shared async scoring transport, preserving messages and raw completion evidence."""
    parameters = _judge_parameters(messages=messages, capability=capability, **kwargs)
    async with AsyncOpenAI(
        **_resolve_client_kwargs(api_key, capability=capability),
        http_client=httpx.AsyncClient(transport=transport, event_hooks=event_hooks),
    ) as client:
        while True:
            try:
                return await client.chat.completions.create(**parameters)
            except BadRequestError as error:
                if not _adapt_judge_parameters(error, parameters):
                    raise


def llm_judge_completion_sync(
    *,
    messages: list[dict[str, Any]],
    api_key: str | None = None,
    capability: str = "vision",
    **kwargs: Any,
) -> Any:
    """Shared sync scoring transport, preserving messages and raw completion evidence."""
    parameters = _judge_parameters(messages=messages, capability=capability, **kwargs)
    with OpenAI(**_resolve_client_kwargs(api_key, capability=capability)) as client:
        while True:
            try:
                return client.chat.completions.create(**parameters)
            except BadRequestError as error:
                if not _adapt_judge_parameters(error, parameters):
                    raise


async def llm_multimodal_text(
    *,
    content: str | list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Shared async multimodal completion entry for LLM judges."""
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    response = await llm_judge_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    return _extract_message_text(response)


def llm_multimodal_text_sync(
    *,
    content: str | list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Shared sync multimodal completion entry for local LLM judges."""
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    response = llm_judge_completion_sync(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    return _extract_message_text(response)


async def llm_multimodal_json(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = 0,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    for attempt in range(2):
        raw = await llm_multimodal_text(
            content=content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        try:
            return _extract_json_object(raw)
        except (ValueError, RuntimeError) as error:
            if attempt:
                raise JudgeInfrastructureError("Judge returned invalid JSON twice") from error


def llm_multimodal_json_sync(
    *,
    content: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float | None = 0,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    for attempt in range(2):
        raw = llm_multimodal_text_sync(
            content=content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        try:
            return _extract_json_object(raw)
        except (ValueError, RuntimeError) as error:
            if attempt:
                raise JudgeInfrastructureError("Judge returned invalid JSON twice") from error


async def llm_vision_judge(
    prompt: str,
    image_bytes: bytes,
    reference_image_bytes: Optional[bytes] = None,
    model: str | None = None,
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
        model: Judge model override. Defaults to JUDGE_MODEL.
        max_tokens: Maximum tokens for the response
        return_binary_score: If True, parses response for YES/NO and returns 1.0/0.0.
                            If False, returns the raw text response.
        api_key: Judge API key override. Defaults to private JUDGE_API_KEY.
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
        logger.exception("Error in llm_vision_judge")
        if eval_context is not None and identifier is not None:
            eval_context.log_error(identifier, e)
        raise JudgeInfrastructureError(f"llm_vision_judge failed: {e}") from e

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
        return result["vlm_response"]


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
    values = load_eval_env()
    resolved_model = (
        model
        or values.get("GEMINI_EVAL_MODEL")
        or values.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_VIDEO_JUDGE_MODEL
    )
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
            "auth_mode": resolved_auth_mode or auth_mode or values.get("GEMINI_AUTH_MODE", "auto"),
            "project": project,
            "location": location,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_binary_answer(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if value not in (0, 1):
            raise ValueError(f"Judge returned a non-binary checklist answer: {value!r}")
        return float(value)
    text = str(value or "").strip().upper()
    if text in {"YES", "Y", "TRUE", "1", "PASS"}:
        return 1.0
    if text in {"NO", "N", "FALSE", "0", "FAIL"}:
        return 0.0
    raise ValueError(f"Judge returned a non-binary checklist answer: {value!r}")


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
            if key not in answers_payload:
                raise ValueError(f"Judge response is missing checklist answer {key!r}")
            score = _parse_binary_answer(answers_payload[key])
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
        logger.exception("Error in llm_vision_binary_checklist_judge")
        if eval_context is not None and identifier is not None:
            eval_context.log_error(identifier, e)
        raise JudgeInfrastructureError(f"llm_vision_binary_checklist_judge failed: {e}") from e

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
        parsed = _normalize_binary_response(raw_response)
        if parsed is None:
            raise JudgeInfrastructureError(
                f"Judge returned a non-binary response for question {index}: {raw_response[:200]!r}"
            )
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
        score: float | None,
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
        eval_entry = {k: v for k, v in eval_entry.items() if v is not None or k == "score"}

        self.evaluation_details["evaluations"].append(eval_entry)
        self._num_evaluated += 1

        # Automatic logging
        if vlm_response:
            logger.info(f"Identifier '{identifier}' VLM response: {vlm_response}")
        logger.info(f"Identifier '{identifier}' judgment score: {score}")
        if error:
            logger.error(f"Identifier '{identifier}' error: {error}")

    def log_error(self, identifier: str, error: Union[str, Exception], score: float | None = None) -> None:
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
    exists = await session.file_exists(target_path) or await session.directory_exists(target_path)
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

                except JudgeInfrastructureError:
                    raise
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
                    f"[{i + 1}/{len(actions_to_execute)}] Executing: {action_type}({action_args})"
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
                                f"Screenshot taken at action {i + 1} for identifier '{identifier}'"
                            )
                    except Exception as e:
                        logger.error(f"Failed to take screenshot at action {i + 1}: {e}")

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

                    except JudgeInfrastructureError:
                        raise
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

        except JudgeInfrastructureError:
            raise
        except Exception as e:
            logger.error(f"Error in deliverable evaluation: {e}")
            ctx.evaluation_details["error"] = str(e)
            return ctx.finalize(error=str(e))
