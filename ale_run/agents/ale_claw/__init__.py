"""ale-claw — OpenClaw native (host-side, in-process) agent deployer.

Re-exports the public surface so callers can do:

    from ale_run.agents.ale_claw import AleClawConfig, AleClawDeployer
"""

from .config import AleClawConfig
from .deployer import AleClawDeployer

__all__ = ["AleClawConfig", "AleClawDeployer"]
