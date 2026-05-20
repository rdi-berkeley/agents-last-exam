"""Web search + web fetch tools (US-OC-056).

Two BaseTool subclasses:
  - :class:`WebSearchTool` — Brave Search API (one provider, env-var key).
  - :class:`WebFetchTool` — HTTP(S) fetch with SSRF guard, Readability-based
    extraction, basic-HTML fallback, HTML→markdown conversion, and a
    per-process TTL cache.

Adapted from OpenClaw's ``web-search.ts`` / ``web-fetch.ts`` /
``web-guarded-fetch.ts`` / ``web-fetch-utils.ts`` / ``web-shared.ts``.

Kept:
  - Brave provider (``web-search-provider-common.ts``), schema params
    ``query``/``count``/``freshness``/``country``/``date_after``.
  - SSRF guard: http(s) only, reject private/loopback/link-local/multicast/
    reserved/unspecified on every resolved IP before fetch.
  - Redirect + timeout + max-response-bytes caps.
  - maxChars + truncation marker.
  - Readability → basic-HTML fallback → raw — matches OpenClaw's 3-tier
    extraction (``extractReadableContent`` → ``extractBasicHtmlContent``).
  - ``htmlToMarkdown`` conversion via ``markdownify``.
  - Per-process ``FETCH_CACHE`` + ``SEARCH_CACHE`` with TTL (5m search,
    10m fetch), matches ``web-shared.ts::CacheEntry``.

Dropped:
  - Multi-provider framework, runtime credential scoping, plugin manifest.
  - ``wrapWebContent`` untrusted-content wrapper (benchmark harness; single
    tenant; revisit when we host untrusted tasks).
  - Cloudflare Markdown-for-Agents header branch.
  - Provider-fallback on extraction failure (single provider).

Known limitation: SSRF guard resolves DNS once and then trusts aiohttp to
connect to a fresh resolution. DNS-rebinding attacks between check and
connect are not prevented. Benchmark harness; accepted. OpenClaw's
``fetchWithSsrFGuard`` pins the resolved address into the socket via a
custom ``LookupFn`` — follow-up story if we ever ingest untrusted task
authors.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlparse

from agent.tools.base import BaseTool, register_tool

from .tools_fs import _run_async

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (match OpenClaw web-fetch.ts:43-51 / web-shared.ts defaults)
# ---------------------------------------------------------------------------

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

_DEFAULT_SEARCH_COUNT = 5
_MAX_SEARCH_COUNT = 20

_DEFAULT_FETCH_MAX_CHARS = 20_000
_MAX_FETCH_MAX_CHARS = 100_000
_DEFAULT_FETCH_MAX_RESPONSE_BYTES = 750_000  # matches OpenClaw DEFAULT_FETCH_MAX_RESPONSE_BYTES
_DEFAULT_FETCH_MAX_REDIRECTS = 3              # matches DEFAULT_FETCH_MAX_REDIRECTS
_DEFAULT_FETCH_TIMEOUT_SECONDS = 30
_DEFAULT_SEARCH_TIMEOUT_SECONDS = 10

# Matches OpenClaw DEFAULT_FETCH_USER_AGENT (web-fetch.ts:50-51)
_DEFAULT_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_SEARCH_CACHE_TTL_SECONDS = 5 * 60
_FETCH_CACHE_TTL_SECONDS = 10 * 60

_VALID_FRESHNESS = {"pd", "pw", "pm", "py"}
_DATE_AFTER_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    value: dict
    expires_at: float


class _TTLCache:
    """Simple per-process TTL dict. Thread-safe.

    Mirrors OpenClaw ``web-shared.ts::CacheEntry`` — a flat key→(value, exp)
    map with lazy expiry on read.
    """

    def __init__(self) -> None:
        self._data: dict[Any, _CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[dict]:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._data.pop(key, None)
                return None
            return entry.value

    def set(self, key: Any, value: dict, ttl_seconds: float) -> None:
        with self._lock:
            self._data[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + ttl_seconds,
            )

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_SEARCH_CACHE = _TTLCache()
_FETCH_CACHE = _TTLCache()


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

# Order matters: Python's ``is_private`` returns True for loopback /
# link-local / multicast / unspecified too, so specific predicates must be
# checked first to surface the most informative label in the error message.
_BLOCKED_IP_PREDICATES = (
    "is_loopback",
    "is_link_local",
    "is_multicast",
    "is_unspecified",
    "is_reserved",
    "is_private",
)


def _assert_url_safe(url: str) -> None:
    """Reject the URL if it is unsafe to fetch.

    Rules (match OpenClaw ``infra/net/ssrf.ts`` policy):
      - Must have a parseable URL.
      - Scheme must be ``http`` or ``https``.
      - Must have a host.
      - Every resolved IP must pass: not private, not loopback, not
        link-local (covers 169.254.169.254 cloud-metadata), not multicast,
        not reserved, not unspecified.

    Raises :class:`ValueError` on rejection so ``call()`` turns it into a
    structured tool error.
    """
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise ValueError(f"Invalid URL {url!r}: {e}") from e
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme {scheme!r} is not allowed (only http/https)."
        )
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL {url!r} is missing a host.")

    # Bare-IP URLs: check the literal first — cheaper and no DNS involved.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _assert_ip_safe(literal, url, host)
        return

    # Hostname → resolve and inspect every returned IP.
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for host {host!r}: {e}") from e

    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0] if sockaddr else ""
        if not ip_str or ip_str in seen:
            continue
        seen.add(ip_str)
        # IPv6 scope-id suffix: "fe80::1%eth0" — strip before parsing.
        ip_clean = ip_str.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(ip_clean)
        except ValueError as e:
            raise ValueError(
                f"Could not parse resolved IP {ip_str!r} for {host!r}: {e}"
            ) from e
        _assert_ip_safe(ip, url, host)


def _assert_ip_safe(
    ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
    url: str,
    host: str,
) -> None:
    for pred in _BLOCKED_IP_PREDICATES:
        if getattr(ip, pred, False):
            raise ValueError(
                f"URL {url!r} resolves to blocked address {ip.compressed} "
                f"({pred.removeprefix('is_')}) for host {host!r}."
            )


# ---------------------------------------------------------------------------
# HTML extraction helpers
# ---------------------------------------------------------------------------


def _extract_with_readability(html: str, url: str) -> Optional[dict]:
    """Run ``readability-lxml`` and return ``{"title", "html"}`` on success.

    Returns ``None`` if readability is unavailable or returns empty content.
    """
    try:
        from readability import Document  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("readability-lxml not installed; skipping to basic fallback")
        return None
    try:
        doc = Document(html, url=url)
        title = (doc.short_title() or "").strip() or None
        summary_html = doc.summary(html_partial=True) or ""
    except Exception as e:  # noqa: BLE001 — readability is lenient; protect the path
        logger.info("readability failed on %s: %s", url, e)
        return None
    if not summary_html.strip():
        return None
    return {"title": title, "html": summary_html}


def _extract_basic_html(html: str) -> Optional[dict]:
    """Fallback extractor using bs4 + html5lib (both already core deps).

    Strips script/style/nav/footer/aside/header then ``get_text`` with a
    newline separator, collapses runs of blank lines. Returns
    ``{"title", "text"}`` or ``None`` if nothing useful was extracted.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("beautifulsoup4 not installed; basic-HTML fallback unavailable")
        return None
    try:
        soup = BeautifulSoup(html, "html5lib")
    except Exception as e:  # noqa: BLE001
        logger.info("bs4 parse failed: %s", e)
        return None
    for selector in ("script", "style", "nav", "footer", "aside", "header", "noscript"):
        for node in soup.find_all(selector):
            node.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    text = soup.get_text(separator="\n")
    # Collapse runs of 2+ blank lines to a single blank line.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    if not text:
        return None
    return {"title": title or None, "text": text}


def _html_to_markdown(html: str) -> str:
    """HTML → Markdown via ``markdownify``.

    Falls back to bs4's ``get_text`` (or the raw HTML) if ``markdownify``
    isn't installed so the fetch path keeps working.
    """
    try:
        from markdownify import markdownify as md  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("markdownify not installed; returning text extraction")
        basic = _extract_basic_html(html)
        return (basic or {}).get("text", html)
    try:
        return md(html, heading_style="ATX").strip()
    except Exception as e:  # noqa: BLE001
        logger.info("markdownify failed: %s", e)
        return html


def _truncate_with_marker(text: str, max_chars: int) -> tuple[str, bool]:
    """Hard-cap ``text`` to ``max_chars`` with a tail marker.

    Returns ``(truncated_text, was_truncated)``.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "\n\n... [truncated {} chars]"
    # Reserve room for the marker so the final string fits under max_chars.
    marker_len = len(marker.format(10**9))  # over-reserve
    keep = max(0, max_chars - marker_len)
    omitted = len(text) - keep
    return text[:keep] + marker.format(omitted), True


# ---------------------------------------------------------------------------
# Param validation helpers
# ---------------------------------------------------------------------------


def _get_required_str(params: dict, key: str, tool_name: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{tool_name}: required parameter "{key}" is missing or empty')
    return value


def _resolve_int(raw: object, default: int, *, min_: int, max_: int) -> int:
    """Clamp ``raw`` to ``[min_, max_]`` if it's a finite number; else default."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    if raw <= 0:
        return default
    val = int(raw)
    return max(min_, min(max_, val))


def _normalize_freshness_and_date_after(
    freshness_raw: object,
    date_after_raw: object,
) -> Optional[str]:
    """Return the ``freshness`` query param Brave expects, or ``None``.

    ``freshness`` accepts ``pd|pw|pm|py`` natively. ``date_after`` is
    ``YYYY-MM-DD``; when set alone it's mapped to Brave's range syntax
    ``YYYY-MM-DDto<today>`` (see ``web-search-provider-common.ts:261``).
    Explicit ``freshness`` wins when both are supplied.
    """
    if isinstance(freshness_raw, str) and freshness_raw.strip():
        candidate = freshness_raw.strip().lower()
        if candidate not in _VALID_FRESHNESS:
            raise ValueError(
                f'freshness must be one of {sorted(_VALID_FRESHNESS)}, got {freshness_raw!r}'
            )
        return candidate
    if isinstance(date_after_raw, str) and date_after_raw.strip():
        candidate = date_after_raw.strip()
        if not _DATE_AFTER_RE.match(candidate):
            raise ValueError(
                f'date_after must be YYYY-MM-DD, got {date_after_raw!r}'
            )
        # Brave range syntax expects "YYYY-MM-DDtoYYYY-MM-DD".
        today = _dt.date.today().isoformat()
        return f"{candidate}to{today}"
    return None


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


@register_tool("web_search")
class WebSearchTool(BaseTool):
    """Search the web via the Brave Search API.

    Requires ``BRAVE_API_KEY`` (env var) or an explicit ``api_key`` kwarg.
    Errors are returned as ``{"success": False, "error": "..."}`` to keep
    the contract aligned with the other OpenClaw tools.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        cfg: Optional[dict] = None,
    ):
        self._api_key_override = api_key
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Search the web (Brave API). Returns ranked results with title, "
            "url, and description."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "count": {
                    "type": "integer",
                    "description": (
                        f"Results to return "
                        f"(1-{_MAX_SEARCH_COUNT}, default {_DEFAULT_SEARCH_COUNT})."
                    ),
                    "minimum": 1,
                    "maximum": _MAX_SEARCH_COUNT,
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Time filter: 'pd' (past day), 'pw' (past week), "
                        "'pm' (past month), 'py' (past year)."
                    ),
                    "enum": sorted(_VALID_FRESHNESS),
                },
                "country": {
                    "type": "string",
                    "description": "ISO country code (e.g. 'US', 'JP'). Biases results.",
                },
                "date_after": {
                    "type": "string",
                    "description": (
                        "Only results newer than YYYY-MM-DD. Mapped to Brave "
                        "range-freshness syntax. Ignored if freshness is set."
                    ),
                },
            },
            "required": ["query"],
        }

    def _resolve_api_key(self) -> str:
        key = self._api_key_override or os.environ.get("BRAVE_API_KEY") or ""
        key = key.strip()
        if not key:
            raise ValueError(
                "web_search requires BRAVE_API_KEY (env var) or an api_key kwarg."
            )
        return key

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            query = _get_required_str(parsed, "query", "web_search")
            count = _resolve_int(
                parsed.get("count"),
                default=_DEFAULT_SEARCH_COUNT,
                min_=1,
                max_=_MAX_SEARCH_COUNT,
            )
            freshness_param = _normalize_freshness_and_date_after(
                parsed.get("freshness"),
                parsed.get("date_after"),
            )
            country_raw = parsed.get("country")
            country: Optional[str] = None
            if country_raw is not None:
                if not isinstance(country_raw, str) or not country_raw.strip():
                    raise ValueError('web_search: "country" must be a non-empty string')
                country = country_raw.strip().upper()
            api_key = self._resolve_api_key()
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        cache_key = (query, count, freshness_param, country)
        hit = _SEARCH_CACHE.get(cache_key)
        if hit is not None:
            return {**hit, "cached": True}

        try:
            result = _run_async(
                self._search(api_key, query, count, freshness_param, country)
            )
        except Exception as e:  # noqa: BLE001 — surface HTTP errors as tool errors
            logger.error("web_search failure on %r: %s", query, e)
            return {"success": False, "error": f"Error: {e}"}

        _SEARCH_CACHE.set(cache_key, result, ttl_seconds=_SEARCH_CACHE_TTL_SECONDS)
        return result

    async def _search(
        self,
        api_key: str,
        query: str,
        count: int,
        freshness: Optional[str],
        country: Optional[str],
    ) -> dict:
        import aiohttp

        params: dict[str, Any] = {"q": query, "count": count}
        if freshness:
            params["freshness"] = freshness
        if country:
            params["country"] = country
        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        timeout = aiohttp.ClientTimeout(total=_DEFAULT_SEARCH_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(BRAVE_SEARCH_URL, params=params, headers=headers) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After", "unknown")
                    raise RuntimeError(
                        f"web_search rate-limited by Brave (HTTP 429, Retry-After={retry_after})"
                    )
                if resp.status >= 400:
                    body = (await resp.text())[:1000]
                    raise RuntimeError(
                        f"web_search failed (HTTP {resp.status}): {body!r}"
                    )
                payload = await resp.json(content_type=None)

        web = payload.get("web") or {}
        raw_results = web.get("results") or []
        results: list[dict[str, Any]] = []
        for r in raw_results[:count]:
            if not isinstance(r, dict):
                continue
            results.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "description": r.get("description") or "",
                }
            )
        return {
            "success": True,
            "provider": "brave",
            "query": query,
            "count": len(results),
            "results": results,
        }


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------


@register_tool("web_fetch")
class WebFetchTool(BaseTool):
    """Fetch an HTTP(S) URL and extract readable text.

    Pipeline: SSRF guard → aiohttp GET (redirect + size cap) → content-type
    routing → readability / basic-HTML / raw → optional markdownify →
    truncate.
    """

    def __init__(
        self,
        *,
        user_agent: Optional[str] = None,
        max_response_bytes: Optional[int] = None,
        cfg: Optional[dict] = None,
    ):
        self.user_agent = user_agent or _DEFAULT_FETCH_USER_AGENT
        self.max_response_bytes = (
            int(max_response_bytes)
            if isinstance(max_response_bytes, (int, float)) and max_response_bytes > 0
            else _DEFAULT_FETCH_MAX_RESPONSE_BYTES
        )
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Fetch and extract readable text from an HTTP(S) URL. Use for "
            "lightweight page access without browser automation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL.",
                },
                "extractMode": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "description": "Output format (default 'markdown').",
                },
                "maxChars": {
                    "type": "integer",
                    "description": (
                        f"Character cap on returned text "
                        f"(default {_DEFAULT_FETCH_MAX_CHARS}, "
                        f"max {_MAX_FETCH_MAX_CHARS})."
                    ),
                    "minimum": 100,
                    "maximum": _MAX_FETCH_MAX_CHARS,
                },
            },
            "required": ["url"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            url = _get_required_str(parsed, "url", "web_fetch")
            extract_mode_raw = parsed.get("extractMode", "markdown")
            if extract_mode_raw not in ("markdown", "text"):
                raise ValueError(
                    f'extractMode must be "markdown" or "text", got {extract_mode_raw!r}'
                )
            extract_mode = extract_mode_raw
            max_chars = _resolve_int(
                parsed.get("maxChars"),
                default=_DEFAULT_FETCH_MAX_CHARS,
                min_=100,
                max_=_MAX_FETCH_MAX_CHARS,
            )
            _assert_url_safe(url)
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        cache_key = (url, extract_mode, max_chars)
        hit = _FETCH_CACHE.get(cache_key)
        if hit is not None:
            return {**hit, "cached": True}

        try:
            result = _run_async(self._fetch(url, extract_mode, max_chars))
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}
        except Exception as e:  # noqa: BLE001 — surface HTTP errors as tool errors
            logger.error("web_fetch failure on %r: %s", url, e)
            return {"success": False, "error": f"Error: {e}"}

        _FETCH_CACHE.set(cache_key, result, ttl_seconds=_FETCH_CACHE_TTL_SECONDS)
        return result

    async def _fetch(
        self,
        url: str,
        extract_mode: str,
        max_chars: int,
    ) -> dict:
        import aiohttp

        t0 = time.monotonic()
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/markdown, text/html;q=0.9, */*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
        }
        timeout = aiohttp.ClientTimeout(total=_DEFAULT_FETCH_TIMEOUT_SECONDS)
        connector = aiohttp.TCPConnector(limit=4)
        body_bytes = bytearray()
        truncated_body = False
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            async with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                max_redirects=_DEFAULT_FETCH_MAX_REDIRECTS,
            ) as resp:
                final_url = str(resp.url)
                # Re-check the final URL after redirects — a server can
                # redirect a public URL to an internal one.
                _assert_url_safe(final_url)
                status = resp.status
                content_type = resp.headers.get("Content-Type", "application/octet-stream")
                async for chunk in resp.content.iter_chunked(32 * 1024):
                    if not chunk:
                        continue
                    remaining = self.max_response_bytes - len(body_bytes)
                    if remaining <= 0:
                        truncated_body = True
                        break
                    if len(chunk) > remaining:
                        body_bytes.extend(chunk[:remaining])
                        truncated_body = True
                        break
                    body_bytes.extend(chunk)

                if status >= 400:
                    detail_html = _decode_best_effort(bytes(body_bytes))
                    detail_text = _html_to_markdown(detail_html) if _looks_like_html(
                        detail_html, content_type
                    ) else detail_html
                    detail_truncated, _ = _truncate_with_marker(detail_text, 4000)
                    raise RuntimeError(
                        f"web_fetch failed (HTTP {status}) for {url}: {detail_truncated}"
                    )

        body_text = _decode_best_effort(bytes(body_bytes))
        normalized_ct = content_type.split(";", 1)[0].strip().lower()

        title: Optional[str] = None
        extractor = "raw"
        text = body_text

        if _is_html_content_type(normalized_ct) or _looks_like_html(body_text, content_type):
            extracted = _extract_with_readability(body_text, final_url)
            if extracted is not None:
                title = extracted["title"]
                html_fragment = extracted["html"]
                if extract_mode == "markdown":
                    text = _html_to_markdown(html_fragment)
                else:
                    basic = _extract_basic_html(html_fragment)
                    text = (basic or {}).get("text", body_text)
                extractor = "readability"
            else:
                basic = _extract_basic_html(body_text)
                if basic is not None:
                    title = basic["title"]
                    text = basic["text"]
                    extractor = "basic-html"
                else:
                    raise RuntimeError(
                        "web_fetch extraction failed: readability and basic-HTML "
                        "fallback both returned empty content."
                    )
        elif normalized_ct == "application/json" or normalized_ct.endswith("+json"):
            try:
                text = json.dumps(json.loads(body_text), indent=2)
                extractor = "json"
            except (ValueError, json.JSONDecodeError):
                text = body_text
                extractor = "raw"
        elif normalized_ct in ("text/markdown", "text/plain") or normalized_ct.startswith("text/"):
            text = body_text
            extractor = "raw"
        else:
            text = body_text
            extractor = "raw"

        truncated_text_flag = False
        if truncated_body:
            truncated_text_flag = True
        truncated_output, marker_applied = _truncate_with_marker(text, max_chars)
        truncated_text_flag = truncated_text_flag or marker_applied

        took_ms = int((time.monotonic() - t0) * 1000)
        return {
            "success": True,
            "url": url,
            "finalUrl": final_url,
            "status": status,
            "contentType": normalized_ct,
            "title": title,
            "extractMode": extract_mode,
            "extractor": extractor,
            "text": truncated_output,
            "truncated": truncated_text_flag,
            "length": len(truncated_output),
            "rawLength": len(body_text),
            "fetchedAt": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "tookMs": took_ms,
        }


def _decode_best_effort(data: bytes) -> str:
    """Decode ``data`` trying utf-8 first, then latin-1 (which never fails).

    aiohttp can surface mis-declared encodings; latin-1 gives us a lossless
    fallback that ``readability``/``bs4`` can still parse.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _is_html_content_type(ct: str) -> bool:
    return ct in ("text/html", "application/xhtml+xml")


def _looks_like_html(body: str, content_type: str) -> bool:
    head = body.lstrip()[:256].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return True
    return "text/html" in (content_type or "").lower()
