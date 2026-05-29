"""Compatibility helpers for importing OpenCV across stage runner images."""

from __future__ import annotations

import importlib
from types import ModuleType


def _ensure_dnn_dict_value(module: ModuleType) -> None:
    dnn = getattr(module, "dnn", None)
    if dnn is not None and not hasattr(dnn, "DictValue"):
        setattr(dnn, "DictValue", object)


def import_cv2() -> ModuleType:
    """Import cv2 while patching older wheels that lack cv2.dnn.DictValue.

    Some VM images carry an OpenCV wheel whose binary module works, but whose
    generated ``cv2.typing`` package references ``cv2.dnn.DictValue`` even
    though that symbol is absent. Patch the native module during OpenCV's own
    bootstrap, before the extra typing module is imported.
    """
    real_import_module = importlib.import_module

    def patched_import_module(name: str, package: str | None = None):
        module = real_import_module(name, package)
        if name == "cv2":
            _ensure_dnn_dict_value(module)
        return module

    importlib.import_module = patched_import_module
    try:
        module = real_import_module("cv2")
        _ensure_dnn_dict_value(module)
        return module
    finally:
        importlib.import_module = real_import_module
