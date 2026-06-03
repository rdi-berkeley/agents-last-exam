"""Tests for ale_run.agents.ale_claw.harness.memory.memory (MemoryStore + SearchResult).

Ported from tests/test_memory_store.py, adapted for the task workspace layout
where session logs live in a memory/ subdirectory under the task dir.
"""

import pytest

from ale_run.agents.ale_claw.harness import MemoryStore, SearchResult


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore for a test task."""
    return MemoryStore(task_id="mota_24_easy", base_dir=tmp_path)


class TestTaskDir:
    def test_returns_correct_path(self, store):
        assert store.task_dir == store.base_dir / "tasks" / "mota_24_easy"

    def test_memory_dir(self, store):
        assert store.memory_dir == store.task_dir / "memory"


class TestInitSession:
    def test_creates_dir_and_first_session(self, store):
        path = store.init_session()
        assert path == "tasks/mota_24_easy/memory/session-001.md"
        assert store.memory_dir.exists()
        assert (store.memory_dir / "session-001.md").exists()

    def test_increments_session_number(self, store):
        store.init_session()
        store2 = MemoryStore(task_id="mota_24_easy", base_dir=store.base_dir)
        path = store2.init_session()
        assert path == "tasks/mota_24_easy/memory/session-002.md"

    def test_multiple_calls_on_same_instance(self, store):
        p1 = store.init_session()
        p2 = store.init_session()
        assert p1 == "tasks/mota_24_easy/memory/session-001.md"
        assert p2 == "tasks/mota_24_easy/memory/session-002.md"

    def test_session_file_has_header(self, store):
        store.init_session()
        content = (store.memory_dir / "session-001.md").read_text(encoding="utf-8")
        assert content.startswith("# Session 001")

    def test_non_session_files_ignored_for_numbering(self, store):
        """Non-session .md files in memory/ don't affect numbering."""
        store.memory_dir.mkdir(parents=True)
        (store.memory_dir / "notes.md").write_text("tmp", encoding="utf-8")
        path = store.init_session()
        assert path == "tasks/mota_24_easy/memory/session-001.md"


class TestAppendToSessionLog:
    def test_appends_with_timestamp(self, store):
        store.init_session()
        path = store.append_to_session_log("found a key on floor 2")
        assert path == "tasks/mota_24_easy/memory/session-001.md"
        content = (store.memory_dir / "session-001.md").read_text(encoding="utf-8")
        assert "found a key on floor 2" in content
        assert "[" in content

    def test_raises_without_init_session(self, store):
        with pytest.raises(RuntimeError, match="init_session"):
            store.append_to_session_log("should fail")

    def test_multiple_appends(self, store):
        store.init_session()
        store.append_to_session_log("first observation")
        store.append_to_session_log("second observation")
        content = (store.memory_dir / "session-001.md").read_text(encoding="utf-8")
        assert "first observation" in content
        assert "second observation" in content
        assert content.count("[") >= 2


class TestWriteTaskMemory:
    def test_creates_file(self, store):
        store.write_task_memory("yellow door needs yellow key")
        content = (store.task_dir / "TASK_MEMORY.md").read_text(encoding="utf-8")
        assert content == "yellow door needs yellow key"

    def test_overwrites_existing(self, store):
        store.write_task_memory("old knowledge")
        store.write_task_memory("new knowledge")
        content = (store.task_dir / "TASK_MEMORY.md").read_text(encoding="utf-8")
        assert content == "new knowledge"

    def test_creates_dir_if_absent(self, store):
        assert not store.task_dir.exists()
        store.write_task_memory("content")
        assert store.task_dir.exists()


class TestReadTaskMemory:
    def test_reads_content(self, store):
        store.write_task_memory("floor 3 strategy")
        assert store.read_task_memory() == "floor 3 strategy"

    def test_returns_empty_if_missing(self, store):
        store.task_dir.mkdir(parents=True, exist_ok=True)
        assert store.read_task_memory() == ""

    def test_returns_empty_if_dir_missing(self, store):
        assert store.read_task_memory() == ""


class TestReadFile:
    def test_missing_file_returns_empty(self, store):
        assert store.read_file("nonexistent.md") == ""

    def test_read_full_file(self, store):
        store.write_task_memory("line1\nline2\nline3\n")
        content = store.read_file("tasks/mota_24_easy/TASK_MEMORY.md")
        assert content == "line1\nline2\nline3\n"

    def test_read_line_range(self, store):
        store.write_task_memory("line1\nline2\nline3\nline4\n")
        content = store.read_file(
            "tasks/mota_24_easy/TASK_MEMORY.md", start_line=2, end_line=3
        )
        assert content == "line2\nline3\n"

    def test_start_line_beyond_file(self, store):
        store.write_task_memory("line1\n")
        content = store.read_file(
            "tasks/mota_24_easy/TASK_MEMORY.md", start_line=100
        )
        assert content == ""

    def test_end_line_beyond_file(self, store):
        store.write_task_memory("line1\nline2\n")
        content = store.read_file(
            "tasks/mota_24_easy/TASK_MEMORY.md", start_line=1, end_line=999
        )
        assert content == "line1\nline2\n"

    def test_read_session_file(self, store):
        store.init_session()
        store.append_to_session_log("log entry")
        content = store.read_file("tasks/mota_24_easy/memory/session-001.md")
        assert "log entry" in content


class TestSearch:
    def test_empty_keywords(self, store):
        assert store.search([]) == []

    def test_no_files_returns_empty(self, store):
        assert store.search(["anything"]) == []

    def test_single_keyword_match(self, store):
        store.write_task_memory(
            "The agent got stuck on floor 1\nFloor 2 was never reached\n"
        )
        results = store.search(["floor"])
        assert len(results) == 2
        assert all(r.score == 1.0 for r in results)

    def test_multiple_keyword_scoring(self, store):
        store.write_task_memory(
            "arrow keys stuck floor\nclick worked\narrow keys failed\n"
        )
        results = store.search(["arrow", "stuck"])
        assert results[0].score == 2.0
        assert "arrow keys stuck" in results[0].content

    def test_case_insensitive(self, store):
        store.write_task_memory("MEMORY is Important\n")
        results = store.search(["memory", "important"])
        assert len(results) == 1
        assert results[0].score == 2.0

    def test_max_results(self, store):
        store.write_task_memory("\n".join(f"line {i} match" for i in range(20)))
        results = store.search(["match"], max_results=5)
        assert len(results) == 5

    def test_searches_session_files(self, store):
        store.init_session()
        store.append_to_session_log("found the key")
        results = store.search(["key"])
        assert len(results) >= 1
        assert any("session-001" in r.file_path for r in results)

    def test_search_across_multiple_files(self, store):
        store.write_task_memory("agent stuck on floor 1\n")
        store.init_session()
        store.append_to_session_log("stuck in loop")
        results = store.search(["stuck"])
        assert len(results) == 2

    def test_result_has_line_number(self, store):
        store.write_task_memory("no match\nno match\ntarget line\n")
        results = store.search(["target"])
        assert results[0].line_number == 3

    def test_to_dict(self, store):
        store.write_task_memory("test content\n")
        results = store.search(["test"])
        d = results[0].to_dict()
        assert set(d.keys()) == {"file_path", "line_number", "content", "score"}


class TestListSessionFiles:
    def test_empty_when_no_dir(self, store):
        assert store.list_session_files() == []

    def test_returns_sorted(self, store):
        store.memory_dir.mkdir(parents=True)
        (store.memory_dir / "session-003.md").write_text("", encoding="utf-8")
        (store.memory_dir / "session-001.md").write_text("", encoding="utf-8")
        (store.memory_dir / "session-002.md").write_text("", encoding="utf-8")
        files = store.list_session_files()
        assert files == [
            "tasks/mota_24_easy/memory/session-001.md",
            "tasks/mota_24_easy/memory/session-002.md",
            "tasks/mota_24_easy/memory/session-003.md",
        ]

    def test_ignores_non_session_files(self, store):
        store.memory_dir.mkdir(parents=True)
        (store.memory_dir / "session-001.md").write_text("", encoding="utf-8")
        (store.memory_dir / "notes.md").write_text("", encoding="utf-8")
        files = store.list_session_files()
        assert files == ["tasks/mota_24_easy/memory/session-001.md"]


class TestGetBootstrapContext:
    def test_returns_task_memory_content(self, store):
        store.write_task_memory("bootstrap content")
        assert store.get_bootstrap_context() == "bootstrap content"

    def test_returns_empty_if_missing(self, store):
        assert store.get_bootstrap_context() == ""


class TestSearchResultDataclass:
    def test_fields(self):
        r = SearchResult(
            file_path="test.md", line_number=1, content="hello", score=1.0
        )
        assert r.file_path == "test.md"
        assert r.line_number == 1
        assert r.content == "hello"
        assert r.score == 1.0
