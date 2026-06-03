"""Dummy smoke agent — no-LLM pipeline check across all tasks.

Scans the task prompt for input/output paths, verifies inputs exist on the
eval VM, and writes a marker file into the output dir. Used to validate
orchestration / sandbox / data-staging without spending model tokens.
"""

from .config import DummyConfig
from .deployer import DummyDeployer

__all__ = ["DummyConfig", "DummyDeployer"]
