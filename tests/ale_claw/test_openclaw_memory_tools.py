"""Tests for OpenClaw memory tools (MemorySearchTool, MemoryGetTool, MemoryWriteTool).

These tools wrap the task-scoped MemoryStore with BaseTool subclasses
so CUA agents can search, read, and write memory files during task execution.
"""

import pytest

from ale_run.agents.ale_claw.harness.memory.memory import (
    MemoryGetTool,
    MemorySearchTool,
    MemoryStore,
    MemoryWriteTool,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(task_id="test_task", base_dir=tmp_path)


@pytest.fixture
def tool(store):
    return MemorySearchTool(store)


class TestMemorySearchTool:
    def test_name_registered(self, tool):
        assert tool.name == "memory_search"

    def test_description_non_empty(self, tool):
        assert len(tool.description) > 0

    def test_parameters_schema(self, tool):
        params = tool.parameters
        assert params["type"] == "object"
        assert "keywords" in params["properties"]
        assert "query" in params["properties"]
        assert params["required"] == []

    def test_search_basic(self, store, tool):
        store.write_task_memory("The agent got stuck on floor 1\nFloor 2 was reached\n")
        result = tool.call({"keywords": ["floor"]})
        assert "floor" in result.lower()
        assert "TASK_MEMORY.md" in result

    def test_search_no_results(self, store, tool):
        store.write_task_memory("some content\n")
        result = tool.call({"keywords": ["nonexistent"]})
        assert "No memory results found" in result

    def test_search_multiple_keywords(self, store, tool):
        store.write_task_memory("arrow keys stuck floor\nclick worked\narrow keys failed\n")
        result = tool.call({"keywords": ["arrow", "stuck"]})
        # Line with both keywords should appear first (higher score)
        lines = result.strip().split("\n")
        assert "score 2.0" in lines[0]

    def test_search_max_results(self, store, tool):
        store.write_task_memory("\n".join(f"line {i} match" for i in range(20)))
        result = tool.call({"keywords": ["match"], "max_results": 3})
        lines = [line for line in result.strip().split("\n") if line]
        assert len(lines) == 3

    def test_search_json_string_params(self, store, tool):
        store.write_task_memory("test content here\n")
        result = tool.call('{"keywords": ["test"]}')
        assert "test content here" in result

    def test_search_missing_keywords_raises(self, tool):
        with pytest.raises(ValueError, match="keywords"):
            tool.call({"not_keywords": ["x"]})

    def test_search_empty_keywords_raises(self, tool):
        with pytest.raises(ValueError, match="keywords.*query"):
            tool.call({"keywords": []})

    def test_query_string_splitting(self, store, tool):
        store.write_task_memory("arrow keys stuck floor\nclick worked\n")
        result = tool.call({"query": "arrow stuck"})
        assert "arrow keys stuck floor" in result

    def test_keywords_precedence(self, store, tool):
        store.write_task_memory("alpha line\nbeta line\n")
        # keywords provided — query should be ignored
        result = tool.call({"keywords": ["alpha"], "query": "beta"})
        assert "alpha" in result

    def test_store_error_returns_friendly_message(self, store, tool, monkeypatch):
        monkeypatch.setattr(
            store, "search",
            lambda *a, **kw: (_ for _ in ()).throw(IOError("disk error")),
        )
        result = tool.call({"keywords": ["test"]})
        assert "Error searching memory" in result
        assert "disk error" in result

    def test_no_query_no_keywords_raises(self, tool):
        with pytest.raises(ValueError, match="keywords.*query"):
            tool.call({})

    def test_empty_query_and_no_keywords_raises(self, tool):
        with pytest.raises(ValueError, match="keywords.*query"):
            tool.call({"query": ""})


@pytest.fixture
def get_tool(store):
    return MemoryGetTool(store)


class TestMemoryGetTool:
    def test_name_registered(self, get_tool):
        assert get_tool.name == "memory_get"

    def test_parameters_schema(self, get_tool):
        params = get_tool.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "from" in params["properties"]
        assert "lines" in params["properties"]
        assert params["required"] == ["path"]

    def test_read_full_file(self, store, get_tool):
        store.write_task_memory("line one\nline two\nline three\n")
        rel_path = f"tasks/{store.task_id}/TASK_MEMORY.md"
        result = get_tool.call({"path": rel_path})
        assert "line one" in result
        assert "line three" in result

    def test_read_line_range(self, store, get_tool):
        store.write_task_memory("line one\nline two\nline three\nline four\n")
        rel_path = f"tasks/{store.task_id}/TASK_MEMORY.md"
        result = get_tool.call({"path": rel_path, "from": 2, "lines": 2})
        assert "line two" in result
        assert "line three" in result
        assert "line one" not in result
        assert "line four" not in result

    def test_path_traversal_rejected(self, get_tool):
        result = get_tool.call({"path": "../etc/passwd"})
        assert "not allowed" in result

    def test_absolute_path_rejected(self, get_tool):
        result = get_tool.call({"path": "/etc/passwd"})
        assert "not allowed" in result

    def test_non_md_rejected(self, get_tool):
        result = get_tool.call({"path": "secrets.txt"})
        assert "only .md files" in result

    def test_missing_file(self, get_tool):
        result = get_tool.call({"path": "nonexistent.md"})
        assert "not found or empty" in result

    def test_json_string_params(self, store, get_tool):
        store.write_task_memory("test content\n")
        rel_path = f"tasks/{store.task_id}/TASK_MEMORY.md"
        result = get_tool.call(f'{{"path": "{rel_path}"}}')
        assert "test content" in result


@pytest.fixture
def write_tool(store):
    return MemoryWriteTool(store)


class TestMemoryWriteTool:
    def test_name_registered(self, write_tool):
        assert write_tool.name == "memory_write"

    def test_parameters_schema(self, write_tool):
        params = write_tool.parameters
        assert params["type"] == "object"
        assert "content" in params["properties"]
        assert "target" in params["properties"]
        assert params["required"] == ["content"]

    def test_write_session(self, store, write_tool):
        store.init_session()
        result = write_tool.call({"content": "session observation", "target": "session"})
        assert "Wrote" in result
        assert "bytes" in result
        # Verify file content
        session_files = list(store.memory_dir.glob("session-*.md"))
        assert len(session_files) == 1
        assert "session observation" in session_files[0].read_text(encoding="utf-8")

    def test_write_task_memory(self, store, write_tool):
        result = write_tool.call({"content": "task-specific note", "target": "task_memory"})
        assert "Wrote" in result
        assert "TASK_MEMORY.md" in result
        assert "task-specific note" in (store.task_dir / "TASK_MEMORY.md").read_text(
            encoding="utf-8"
        )

    def test_empty_content_rejected(self, write_tool):
        result = write_tool.call({"content": ""})
        assert "Error" in result

    def test_whitespace_content_rejected(self, write_tool):
        result = write_tool.call({"content": "   \n\t  "})
        assert "Error" in result

    def test_default_target_is_session(self, store, write_tool):
        store.init_session()
        result = write_tool.call({"content": "default target test"})
        assert "Wrote" in result
        session_files = list(store.memory_dir.glob("session-*.md"))
        assert any(
            "default target test" in f.read_text(encoding="utf-8") for f in session_files
        )

    def test_no_session_init_error(self, store, write_tool):
        result = write_tool.call({"content": "should fail", "target": "session"})
        assert "Error" in result
        assert "session" in result.lower()

    def test_invalid_target_rejected(self, write_tool):
        result = write_tool.call({"content": "hello", "target": "bogus"})
        assert "Error" in result
        assert "bogus" in result

    def test_json_string_params(self, store, write_tool):
        result = write_tool.call('{"content": "json test", "target": "task_memory"}')
        assert "Wrote" in result
        assert (store.task_dir / "TASK_MEMORY.md").read_text(encoding="utf-8") == "json test"
