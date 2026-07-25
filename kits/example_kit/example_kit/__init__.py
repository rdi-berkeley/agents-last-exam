"""Worked example of a shared verification library.

Standard library only, and written to run on Python 3.8: a kit lands in whatever image
the task chose, and those interpreters are frequently older than the one you develop on.

Keeping grading logic here rather than duplicating it across tasks is the point — when
a scoring rule changes, it changes once, and the kit's content hash records exactly
which version produced a given result.
"""

import json
import os

__all__ = ["read_params", "write_rewards"]


def read_params():
    """Return the effective parameters for this episode, variant merge already applied."""
    with open(os.environ["ALE_PARAMS_JSON"], encoding="utf-8") as handle:
        return json.load(handle)


def write_rewards(rewards, path=None):
    """Write the rewards a verifier produced.

    The engine reads exactly one file, so writing it through a shared helper keeps the
    format identical across every task in the domain.
    """
    target = path or os.environ["ALE_VERDICT_PATH"]
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"rewards": rewards}, handle)
