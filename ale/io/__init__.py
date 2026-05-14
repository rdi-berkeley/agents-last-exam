"""IO layer: structured log writer + (future) artifact sink."""

from .run_writer import RunWriter, slug_model, slug_task

__all__ = ["RunWriter", "slug_model", "slug_task"]
