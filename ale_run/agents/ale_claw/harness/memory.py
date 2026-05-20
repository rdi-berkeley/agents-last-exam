"""MemoryStore — task-workspace storage for the OpenClaw agent harness.

Manages per-task persistent memory with the layout:

    <base_dir>/tasks/<task_id>/
    ├── TASK_MEMORY.md              # Curated task knowledge (bootstrap-injected)
    └── memory/
        ├── session-001.md          # Session log (append-only)
        ├── session-002.md
        └── ...

Reference implementation: memory/store.py (AgentHLE prototype)
Design rationale: docs/plan/US-OC-002-memory-store.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

from agent.tools.base import BaseTool, register_tool


@dataclass
class SearchResult:
    """A single search result from memory files."""

    file_path: str
    line_number: int
    content: str
    score: float

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "content": self.content,
            "score": self.score,
        }


class MemoryStore:
    """Task-workspace storage backend for persistent agent memory.

    Each task gets an isolated workspace under ``base_dir/tasks/<task_id>/``
    containing a curated knowledge file (``TASK_MEMORY.md``) and append-only
    session logs (``memory/session-NNN.md``).

    Args:
        base_dir: Root directory for all task workspaces.
        task_id: Required task identifier scoping the workspace.
    """

    TASKS_DIR = "tasks"
    TASK_MEMORY_FILE = "TASK_MEMORY.md"
    MEMORY_SUBDIR = "memory"

    DEFAULT_BASE_DIR = "openclaw_memory"

    def __init__(self, task_id: str, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir if base_dir is not None else self.DEFAULT_BASE_DIR)
        self.task_id = task_id
        self._current_session_path: Path | None = None

    @property
    def task_dir(self) -> Path:
        """Path to the task-scoped directory: ``base_dir/tasks/<task_id>``."""
        return self.base_dir / self.TASKS_DIR / self.task_id

    @property
    def memory_dir(self) -> Path:
        """Path to the session logs directory: ``task_dir/memory``."""
        return self.task_dir / self.MEMORY_SUBDIR

    def init_session(self) -> str:
        """Create a new session log file with the next sequential number.

        Returns:
            Relative path from base_dir (e.g. ``tasks/my_task/memory/session-001.md``).
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing session files to determine next number
        existing = sorted(self.memory_dir.glob("session-*.md"))
        next_num = 1
        if existing:
            for f in existing:
                match = re.match(r"session-(\d+)\.md$", f.name)
                if match:
                    next_num = max(next_num, int(match.group(1)) + 1)

        session_file = self.memory_dir / f"session-{next_num:03d}.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"# Session {next_num:03d} — {timestamp}\n\n"
        session_file.write_text(header, encoding="utf-8")
        self._current_session_path = session_file

        return str(session_file.relative_to(self.base_dir))

    def append_to_session_log(self, content: str) -> str:
        """Append a timestamped entry to the current session file.

        Returns:
            Relative path to the session file.

        Raises:
            RuntimeError: If ``init_session()`` has not been called.
        """
        if self._current_session_path is None:
            raise RuntimeError(
                "init_session() must be called before append_to_session_log()"
            )

        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n[{timestamp}] {content}\n"

        with open(self._current_session_path, "a", encoding="utf-8") as f:
            f.write(entry)

        return str(self._current_session_path.relative_to(self.base_dir))

    def write_task_memory(self, content: str) -> None:
        """Overwrite ``TASK_MEMORY.md`` for the current task. Creates dir if absent."""
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / self.TASK_MEMORY_FILE).write_text(content, encoding="utf-8")

    def read_task_memory(self) -> str:
        """Read ``TASK_MEMORY.md`` content. Returns empty string if missing."""
        task_memory_path = self.task_dir / self.TASK_MEMORY_FILE
        if not task_memory_path.exists():
            return ""
        try:
            return task_memory_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def read_file(
        self,
        relative_path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """Read content from a file with optional line range.

        Args:
            relative_path: Path relative to base_dir.
            start_line: 1-based start line (default 1).
            end_line: 1-based end line inclusive (default: read to end).

        Returns:
            File content as string. Empty string if file doesn't exist.
        """
        file_path = self.base_dir / relative_path
        if not file_path.exists():
            return ""

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeDecodeError):
            return ""

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line) if end_line is not None else len(lines)

        return "".join(lines[start_idx:end_idx])

    def search(
        self, keywords: list[str], max_results: int = 10
    ) -> list[SearchResult]:
        """Case-insensitive keyword search across TASK_MEMORY.md and session logs.

        Returns results sorted by score (descending), then file path, then line number.
        """
        if not keywords:
            return []

        keywords_lower = [k.lower() for k in keywords]
        results: list[SearchResult] = []

        md_files: list[Path] = []

        # TASK_MEMORY.md
        task_memory = self.task_dir / self.TASK_MEMORY_FILE
        if task_memory.exists():
            md_files.append(task_memory)

        # Session logs
        if self.memory_dir.exists():
            md_files.extend(sorted(self.memory_dir.glob("session-*.md")))

        for file_path in md_files:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue

            relative_path = str(file_path.relative_to(self.base_dir))

            for line_num, line in enumerate(lines, start=1):
                line_lower = line.lower()
                score = sum(1.0 for kw in keywords_lower if kw in line_lower)
                if score > 0:
                    results.append(
                        SearchResult(
                            file_path=relative_path,
                            line_number=line_num,
                            content=line.strip(),
                            score=score,
                        )
                    )

        results.sort(key=lambda r: (-r.score, r.file_path, r.line_number))
        return results[:max_results]

    def list_session_files(self) -> list[str]:
        """Return sorted list of session-NNN.md relative paths."""
        if not self.memory_dir.exists():
            return []
        files = sorted(self.memory_dir.glob("session-*.md"))
        return [str(f.relative_to(self.base_dir)) for f in files]

    def get_bootstrap_context(self) -> str:
        """Return TASK_MEMORY.md content for ContextFile injection into the system prompt."""
        return self.read_task_memory()


# ---------------------------------------------------------------------------
# Memory tools — BaseTool subclasses for agent file I/O
# ---------------------------------------------------------------------------

_WRITE_TARGETS = ("session", "task_memory")


@register_tool("memory_search")
class MemorySearchTool(BaseTool):
    """Search task-scoped memory files by keywords.

    Reference: openclaw/src/agents/tools/memory-tool.ts (createMemorySearchTool).
    Deviations: keyword-based (not semantic/embedding), accepts both ``query``
    and ``keywords`` params, returns plain text lines for CUA consumption.
    """

    def __init__(self, store: MemoryStore, cfg=None):
        self.store = store
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Search task memory files (TASK_MEMORY.md and session logs) by keywords. "
            "Use this to recall past observations, strategies, mistakes, or patterns. "
            "Returns matched lines with file path and line number."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (split on whitespace into keywords).",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords to search for. Lines matching more keywords rank higher.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10).",
                },
            },
            "required": [],
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params_dict = self._verify_json_format_args(params)

        keywords = params_dict.get("keywords", [])
        query = params_dict.get("query", "")

        # Resolve keywords: prefer explicit keywords, fall back to splitting query
        if not keywords and query:
            keywords = query.strip().split()
        if not keywords:
            raise ValueError(
                "'keywords' or 'query' must be provided. "
                "Pass keywords: [...] or query: 'space separated terms'."
            )

        max_results = params_dict.get("max_results", 10)

        try:
            results = self.store.search(keywords, max_results=max_results)
        except Exception as e:
            return f"Error searching memory: {e}"

        if not results:
            return f"No memory results found for keywords: {keywords}"

        lines = []
        for r in results:
            lines.append(f"[{r.file_path}:{r.line_number}] (score {r.score}) {r.content}")
        return "\n".join(lines)


@register_tool("memory_get")
class MemoryGetTool(BaseTool):
    """Read task-scoped memory files or specific line ranges.

    Reference: openclaw/src/agents/tools/memory-tool.ts (createMemoryGetTool)
    and openclaw/src/memory/manager.ts (readFile). API shape (path/from/lines),
    .md-only restriction, and path traversal checks follow that implementation.
    """

    def __init__(self, store: MemoryStore, cfg=None):
        self.store = store
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Read a memory file (TASK_MEMORY.md or session logs) with optional "
            "line range. Use after memory_search to pull only the needed lines "
            "and keep context small."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to the memory file "
                        "(e.g. 'tasks/my_task/TASK_MEMORY.md' or "
                        "'tasks/my_task/memory/session-001.md')."
                    ),
                },
                "from": {
                    "type": "integer",
                    "description": "1-based starting line number (default: 1).",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read from the starting line (default: entire file).",
                },
            },
            "required": ["path"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params_dict = self._verify_json_format_args(params)

        file_path = params_dict.get("path", "")

        # Security: reject path traversal and absolute paths
        resolved = (self.store.base_dir / file_path).resolve()
        if not str(resolved).startswith(str(self.store.base_dir.resolve())):
            return "Error: path traversal is not allowed. Use a relative path within memory."

        # Only allow .md files
        if not file_path.endswith(".md"):
            return "Error: only .md files can be read."

        start_line = params_dict.get("from", 1)
        num_lines = params_dict.get("lines", None)

        end_line = (start_line + num_lines - 1) if num_lines is not None else None

        content = self.store.read_file(file_path, start_line=start_line, end_line=end_line)

        if not content:
            return f"File '{file_path}' not found or empty."

        return content


@register_tool("memory_write")
class MemoryWriteTool(BaseTool):
    """Write content to task-scoped memory files.

    No direct OpenClaw equivalent. OpenClaw agents write memory via filesystem
    access; this explicit tool enables the same capability for CUA agents that
    lack direct file I/O. Two targets: session log (append) and task memory
    (overwrite).
    """

    def __init__(self, store: MemoryStore, cfg=None):
        self.store = store
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Write content to task memory:\n"
            "- 'session' (default): append to the current session log (timestamped)\n"
            "- 'task_memory': overwrite TASK_MEMORY.md (curated task knowledge)"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text content to write.",
                },
                "target": {
                    "type": "string",
                    "enum": list(_WRITE_TARGETS),
                    "description": "Where to write: 'session' (default) or 'task_memory'.",
                },
            },
            "required": ["content"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params_dict = self._verify_json_format_args(params)

        content = params_dict.get("content", "")
        if not content or not content.strip():
            return "Error: content must be a non-empty string (not blank/whitespace)."

        target = params_dict.get("target", "session")
        if target not in _WRITE_TARGETS:
            return f"Error: target must be one of {_WRITE_TARGETS}, got '{target}'."

        if target == "session":
            try:
                path = self.store.append_to_session_log(content)
            except RuntimeError:
                return (
                    "Error: no session initialized. "
                    "The session must be started (init_session) before writing to it."
                )
        else:  # task_memory
            self.store.write_task_memory(content)
            path = "TASK_MEMORY.md"

        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
