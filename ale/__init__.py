"""agent-last-exam (ale): OpenEnv-aligned benchmark framework."""

__version__ = "0.1.0"

# Top-level public API: registry + the single Env class.
from .registry import auto_discover, list_envs, make, register, unregister

__all__ = [
    "__version__",
    "register",
    "unregister",
    "make",
    "list_envs",
    "auto_discover",
    "AgenthleEnv",
]


def __getattr__(name: str):
    """Lazy attribute access for the heavy Env class — avoids import cycles."""
    if name == "AgenthleEnv":
        from .core.env import AgenthleEnv
        return AgenthleEnv
    raise AttributeError(name)
