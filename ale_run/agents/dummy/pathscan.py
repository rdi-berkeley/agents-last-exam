"""Prompt path extraction for the dummy smoke agent.

Pure-stdlib, no framework imports — the same heuristic used to scan all 147
tasks offline, so the agent's launch-time view matches the offline report.

The task description renders absolute paths in one of several styles
(``E:\\agenthle\\...``, ``/media/user/data/agenthle/...``,
``/home/user/Desktop/...``, ``/workspace/...``, container-relative
``/input/...``). We don't reconstruct paths from the catalog id — task
configs deliberately diverge from it (``DOMAIN_NAME`` overrides,
``VISIBLE_TASK_NAME`` aliases, per-variant dir names, ``/workspace``
staging). We only trust the literal path tokens the prompt renders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Backtick-quoted spans hold most paths; a span may carry several
# whitespace-separated tokens (e.g. a `cp src dst` example), so split inside.
_BACKTICK = re.compile(r"`([^`]+)`")
# Bare Windows drive paths and POSIX paths that appear outside backticks.
_WINPATH = re.compile(r"[A-Za-z]:\\[^\s`'\"]+")
_UNIXPATH = re.compile(r"/(?:[\w.\-]+/)+[\w.\-]+")


def _looks_like_path(tok: str) -> bool:
    return "\\" in tok or tok.startswith("/")


def extract_paths(text: str) -> set[str]:
    """All path-like tokens in ``text`` (trailing separators stripped)."""
    cands: set[str] = set()
    for span in _BACKTICK.findall(text):
        for tok in re.split(r"\s+", span):
            tok = tok.strip().strip("\"'")
            if _looks_like_path(tok):
                cands.add(tok.rstrip("\\/"))
    for m in _WINPATH.findall(text):
        cands.add(m.rstrip("\\/"))
    for m in _UNIXPATH.findall(text):
        cands.add(m.rstrip("\\/"))
    return cands


def _sep_of(path: str) -> str:
    return "\\" if "\\" in path else "/"


def output_dirs(output_paths: list[str]) -> list[str]:
    """Map output path tokens to the directory to create.

    A token may be the output dir itself (``...\\base\\output``) or a file
    under it (``...\\output\\result.gpkg``); in both cases we want the dir
    ending at the ``output`` segment. Prefer a segment whose name is exactly
    ``output``; fall back to any segment merely containing ``output`` (e.g.
    ``output_test_pos``) only when no exact match exists.
    """
    exact: set[str] = set()
    fuzzy: set[str] = set()
    for p in output_paths:
        sep = _sep_of(p)
        segs = p.split(sep)
        exact_idx = [i for i, s in enumerate(segs) if s.lower() == "output"]
        if exact_idx:
            i = exact_idx[-1]
            exact.add(sep.join(segs[: i + 1]))
            continue
        fuzzy_idx = [i for i, s in enumerate(segs) if "output" in s.lower()]
        if fuzzy_idx:
            i = fuzzy_idx[-1]
            fuzzy.add(sep.join(segs[: i + 1]))
    return sorted(exact) if exact else sorted(fuzzy)


def join_path(directory: str, name: str) -> str:
    return f"{directory}{_sep_of(directory)}{name}"


def gcs_pos_candidates(
    output_dir: str,
    task_data_root: str | None,
    *,
    bucket: str = "gs://ale-data-all",
    pos_name: str = "output_test_pos",
) -> list[str]:
    """Candidate ``gs://`` URLs for a task's positive reference output.

    The GCS bucket mirrors the on-VM layout: data lives at
    ``<bucket>/<domain>/<task>/<variant>/<subdir>`` exactly as the framework
    stages it under ``<task_data_root>/<domain>/<task>/<variant>/<subdir>``.
    So the positive output is the agent's output dir taken relative to
    ``task_data_root`` with the trailing ``output`` segment swapped for
    ``output_test_pos``.

    A few tasks render an output root that differs from ``task_data_root``
    (e.g. a symlinked ``/home/user/Desktop`` tree, or a task whose VM path
    drops a level); for those we add a best-effort fallback using the last
    three path segments before ``output``. Callers probe each candidate with
    ``gsutil ls`` and use the first that exists, reporting when none do.
    """
    bucket = bucket.rstrip("/")
    out = output_dir.replace("\\", "/").rstrip("/")
    segs = out.split("/")
    if not segs or "output" not in segs[-1].lower():
        return []
    cands: list[str] = []
    root = (task_data_root or "").replace("\\", "/").rstrip("/")
    if root and out.lower().startswith(root.lower() + "/"):
        rel = out[len(root):].strip("/").split("/")
        rel[-1] = pos_name
        cands.append(f"{bucket}/" + "/".join(rel))
    if len(segs) >= 4:
        fallback = f"{bucket}/" + "/".join(segs[-4:-1] + [pos_name])
        if fallback not in cands:
            cands.append(fallback)
    return cands


@dataclass
class ScanResult:
    all_paths: list[str] = field(default_factory=list)
    input_paths: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    output_dirs: list[str] = field(default_factory=list)

    @property
    def has_input(self) -> bool:
        return bool(self.input_paths)

    @property
    def has_output(self) -> bool:
        return bool(self.output_dirs)


def scan(prompt: str) -> ScanResult:
    """Parse a task prompt into input/output path findings."""
    paths = extract_paths(prompt)
    inputs = sorted(p for p in paths if "input" in p.lower())
    outputs = sorted(p for p in paths if "output" in p.lower())
    return ScanResult(
        all_paths=sorted(paths),
        input_paths=inputs,
        output_paths=outputs,
        output_dirs=output_dirs(outputs),
    )
