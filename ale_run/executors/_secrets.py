"""Secret handling shared by the executors and their entry points.

API keys (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, ...) must NEVER be
serialized into ``_spec.json`` — that file is gathered back to the host
into ``.logs/.../origin_log/<agent>/_spec.json`` and would persist
plaintext keys on host disk (committable / shareable).

Instead the framework writes the env into a sibling ``_secrets.json``
that the in-sandbox / in-container entry reads ONCE, injects into the
process environment, then deletes immediately. The gather step also
excludes this filename by name as defense-in-depth, so even a racing
or failed delete cannot leak it to host logs.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import fields
from pathlib import Path
from typing import Any

# Basename of the transient secrets sidecar. Lives next to ``_spec.json``
# in the deployer work_dir; read-once, then deleted by the entry.
SECRETS_FILE = "_secrets.json"

# Control files that carry secrets and must never reach host logs. The
# gather paths exclude these by basename as a belt-and-suspenders guard.
SECRET_GATHER_EXCLUDES = frozenset({SECRETS_FILE, "_env"})

# Config values resolved from `${env:...}` are ordinary dataclass fields by the
# time an executor sees them. Keep those values in the same read-once sidecar
# as process env instead of serializing them into the gathered `_spec.json`.
_CONFIG_SECRET_PREFIX = "_ALE_CONFIG_SECRET_"


def _is_config_secret_field(name: str, metadata: Any) -> bool:
    return bool(metadata.get("secret")) or name == "api_key" or name.endswith("_api_key")


def config_to_kwargs_and_secrets(
    config: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Serialize a dataclass config while extracting secret-valued fields."""
    kwargs: dict[str, Any] = {}
    secrets: dict[str, str] = {}
    for config_field in fields(config):
        value = getattr(config, config_field.name)
        if not isinstance(
            value,
            (str, int, float, bool, type(None), list, dict, tuple),
        ):
            continue
        if (
            isinstance(value, str)
            and value
            and _is_config_secret_field(config_field.name, config_field.metadata)
        ):
            secret_name = f"{_CONFIG_SECRET_PREFIX}{config_field.name.upper()}"
            kwargs[config_field.name] = None
            secrets[secret_name] = value
        else:
            kwargs[config_field.name] = value
    return kwargs, secrets


def restore_config_secrets(
    config_kwargs: dict[str, Any],
    env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Restore extracted config fields and remove private sidecar keys."""
    restored = dict(config_kwargs)
    clean_env = dict(env)
    for key in tuple(clean_env):
        if not key.startswith(_CONFIG_SECRET_PREFIX):
            continue
        field_name = key.removeprefix(_CONFIG_SECRET_PREFIX).lower()
        restored[field_name] = clean_env.pop(key)
    return restored, clean_env


def write_secrets(work_dir: Path, env: dict[str, str]) -> Path:
    """Write ``env`` to ``<work_dir>/_secrets.json`` with 0600 perms.

    Returns the path written. An empty env still writes an empty object
    so the entry's read path is uniform.
    """
    path = Path(work_dir) / SECRETS_FILE
    path.write_text(json.dumps(dict(env or {})))
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass
    return path


def read_and_delete_secrets(work_dir: str | Path) -> dict[str, str]:
    """Read ``<work_dir>/_secrets.json``, then delete it immediately.

    Returns the env dict (empty if the file is absent). Deletion is
    best-effort but attempted before the dict is returned so the secret
    sidecar does not outlive the read even on the unhappy path.
    """
    path = Path(work_dir) / SECRETS_FILE
    try:
        raw = path.read_text()
    except (FileNotFoundError, OSError):
        return {}
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in (data or {}).items()}


def inject_env(env: dict[str, str]) -> None:
    """Inject ``env`` into ``os.environ`` (string-coerced)."""
    for k, v in (env or {}).items():
        os.environ[str(k)] = str(v)
