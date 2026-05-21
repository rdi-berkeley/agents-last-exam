"""Recursive remote → host directory mirror.

Used after the agent exits to pull ``remote_work_dir`` (deployer artifacts) into
``<run_dir>/origin_log/<agent_name>/``. Single transport: CUA HTTP via the
``environments.remote`` primitives. No GCS bridge, no incremental puller —
those land in a later iteration.

Returns a report dict suitable for the ``origin_log_gather_done`` event:

    {"transport": "cua", "files": <int>, "error": <str|None>}
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..base_interface import EnvHandle
from ..environments.remote import download_file, list_remote_dir

logger = logging.getLogger(__name__)

_PER_FILE_RETRIES = 3
_RETRY_BACKOFFS_S = (1.0, 3.0, 9.0)


async def pull_dir(session: Any, *, src: str, dst: Path, os_type: str) -> dict:
    """Pull ``src`` (on the remote env) into ``dst`` (on the host).

    Walks ``src`` via ``list_remote_dir`` then downloads each file via
    ``download_file`` with bounded retries. The session's underlying
    endpoint is inferred from ``session._api_host`` / ``session._api_port``
    (RemoteDesktopSession's stored attrs).
    """
    api_host = getattr(session, "_api_host", None)
    api_port = getattr(session, "_api_port", None)
    if not api_host or not api_port:
        return {"transport": "cua", "files": 0, "error": "session has no api_host/api_port"}

    env_cfg = EnvHandle(
        id="",
        endpoint=f"http://{api_host}:{api_port}",
        os=os_type,
    )

    dst.mkdir(parents=True, exist_ok=True)

    try:
        entries = await asyncio.to_thread(list_remote_dir, env_cfg, src)
    except Exception as e:
        logger.warning("list_remote_dir failed for %s: %s", src, e)
        return {"transport": "cua", "files": 0, "error": str(e)}

    if not entries:
        logger.info("gather.pull_dir: no entries at %s", src)
        return {"transport": "cua", "files": 0, "error": None}

    file_count = 0
    last_error: str | None = None
    sep = "/" if os_type == "linux" else "\\"

    for entry in entries:
        rel = entry["relpath"]
        local = dst / rel
        if entry["is_dir"]:
            local.mkdir(parents=True, exist_ok=True)
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        remote_path = f"{src.rstrip(sep)}{sep}{rel.replace('/', sep)}"
        ok = await _download_with_retry(env_cfg, remote_path, local)
        if ok:
            file_count += 1
        else:
            last_error = f"download failed: {rel}"
            logger.warning("gather.pull_dir: %s", last_error)

    return {"transport": "cua", "files": file_count, "error": last_error}


async def _download_with_retry(env_cfg: EnvHandle, remote_path: str, local: Path) -> bool:
    for attempt in range(_PER_FILE_RETRIES):
        try:
            ok = await asyncio.to_thread(download_file, env_cfg, remote_path, str(local), 120)
        except Exception as e:
            logger.debug("download_file raised for %s (attempt %d): %s", remote_path, attempt + 1, e)
            ok = False
        if ok:
            return True
        if attempt < _PER_FILE_RETRIES - 1:
            await asyncio.sleep(_RETRY_BACKOFFS_S[attempt])
    return False
