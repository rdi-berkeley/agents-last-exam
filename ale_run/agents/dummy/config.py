"""DummyConfig — knobs for the smoke-test agent.

Standalone dataclass (no shared base), per ALE convention. The dummy agent
runs no LLM, so ``model`` is accepted only because the yaml loader maps a
top-level ``model:`` into ``config["model"]`` (and ``build_config`` would
otherwise drop it); it is never used.

What the agent does: scan the prompt for input/output paths, check inputs
exist on the eval VM, then pull each task's *positive reference output*
(``output_test_pos``) from the GCS bucket straight into the VM's output dir —
simulating an agent that produced the correct answer, so evaluation should
score it high. This exercises orchestration + sandbox + data staging + the
output pull + the scorer end-to-end without spending model tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class DummyConfig:
    """Tunables for :class:`DummyDeployer`."""

    name: ClassVar[str] = "dummy"

    model: str = "none"
    """Unused. Present so a ``model:`` line in the agent yaml doesn't surprise
    anyone — the dummy agent makes no model calls."""

    data_bucket: str = "gs://ale-data-all"
    """GCS bucket holding per-task data. The positive reference output is
    ``<bucket>/<domain>/<task>/<variant>/output_test_pos``."""

    pos_subdir: str = "output_test_pos"
    """Name of the positive-output reference dir to pull from GCS."""

    connect_timeout_s: int = 120
    """Seconds to wait for the eval VM's cua-server to become responsive."""

    pull_timeout_s: int = 900
    """Per-output-dir ``gsutil rsync`` timeout (seconds)."""

    fail_on_missing_input: bool = False
    """If True, a missing input path flips the run status to ``failed``.
    Default False: missing inputs are recorded but the run still completes, so
    one broken task doesn't mask the rest of the sweep."""

    write_marker_when_no_pos: bool = False
    """If True, when no ``output_test_pos`` is found on GCS, still write a
    small ``dummy.json`` marker into the output dir (proves the dir is
    writable). Default False: leave the output dir clean and just report the
    miss to the result file."""

    marker_filename: str = "dummy.json"
    """Marker file name used when :attr:`write_marker_when_no_pos` is True."""
