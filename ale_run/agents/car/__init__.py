"""Common Agent Runtime (CAR) agent — out-of-sandbox harness over stdio MCP.

CAR is a deterministic execution layer for AI agents: models propose, the runtime
validates and executes. This deployer runs CAR as the system under test by
launching the ``car run-task`` headless runner against the eval VM via the vm
(shell/fs) and cua (GUI) MCP bridges, then converts CAR's transcript to ATIF.
"""

from .config import CarConfig
from .deployer import CarDeployer

__all__ = ["CarConfig", "CarDeployer"]
