"""Minimal OpenAI-compatible chat client used by the rubric judge (vision optional).

Configuration (environment): OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1),
ALE_JUDGE_MODEL (default gpt-5.6-terra), ALE_JUDGE_VOTES (default 3). Every call asks for a JSON object.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request

DEFAULT_MODEL = "gpt-5.6-terra"


def _img_part(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}}


def chat_json(system: str, user_text: str, images: list[str] | None = None, model: str | None = None, retries: int = 4) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set for the rubric judge")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = model or os.environ.get("ALE_JUDGE_MODEL", DEFAULT_MODEL)
    content: list[dict] = [{"type": "text", "text": user_text}] + [_img_part(p) for p in (images or [])]
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}], "response_format": {"type": "json_object"}}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
            text = resp["choices"][0]["message"]["content"]
            return json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception as exc:  # noqa: BLE001 - retry any transport / parse failure
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"judge call failed after {retries} attempts: {last}")


def votes() -> int:
    return max(1, int(os.environ.get("ALE_JUDGE_VOTES", "3")))
