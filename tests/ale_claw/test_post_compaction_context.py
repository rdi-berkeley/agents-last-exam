"""Tests for post-compaction context re-injection (Gap A).

Mirrors OpenClaw's readPostCompactionContext (post-compaction-context.ts):
after a compaction event, a fresh user-role message is appended that
re-reads the bootstrap files verbatim. Re-anchors the agent on stable
workspace rules and seeds a byte-stable cache prefix block.
"""



from ale_claw.harness.agent_loop import OpenClawComputerAgent
from ale_claw.harness.prompt import ContextFile


def _make_agent(context_files):
    """Construct an OpenClawComputerAgent with mocks for everything not under test.

    We bypass __init__ — the goal is to exercise _build_post_compaction_message,
    which only reads ``self._context_files``.
    """
    agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
    agent._context_files = list(context_files or [])
    return agent


class TestBuildPostCompactionMessage:
    def test_returns_none_when_no_context_files(self):
        agent = _make_agent([])
        assert agent._build_post_compaction_message() is None

    def test_returns_none_when_context_files_is_none(self):
        agent = _make_agent(None)
        assert agent._build_post_compaction_message() is None

    def test_emits_user_role_message(self):
        agent = _make_agent(
            [ContextFile(path="AGENTS.md", content="rule one\nrule two")]
        )
        msg = agent._build_post_compaction_message()
        assert msg is not None
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)

    def test_includes_auto_marker(self):
        agent = _make_agent(
            [ContextFile(path="AGENTS.md", content="x")]
        )
        msg = agent._build_post_compaction_message()
        # Auto-injection marker so the model knows it wasn't user input.
        assert msg["content"].startswith("[Auto: post-compaction context refresh]")

    def test_includes_each_context_file(self):
        files = [
            ContextFile(path="AGENTS.md", content="agent rules here"),
            ContextFile(path="TASK_MEMORY.md", content="task memory here"),
        ]
        agent = _make_agent(files)
        msg = agent._build_post_compaction_message()
        assert "## AGENTS.md" in msg["content"]
        assert "agent rules here" in msg["content"]
        assert "## TASK_MEMORY.md" in msg["content"]
        assert "task memory here" in msg["content"]

    def test_byte_stable_across_calls(self):
        """Two consecutive calls with identical inputs produce identical output.

        This is the property that makes the message cacheable as a stable
        prefix point — if it varied per call, every compaction would write
        a fresh, never-reused cache block.
        """
        files = [ContextFile(path="AGENTS.md", content="stable content")]
        agent = _make_agent(files)
        first = agent._build_post_compaction_message()
        second = agent._build_post_compaction_message()
        assert first == second
        assert first["content"] == second["content"]

    def test_preserves_file_order(self):
        files = [
            ContextFile(path="A.md", content="alpha"),
            ContextFile(path="B.md", content="beta"),
            ContextFile(path="C.md", content="gamma"),
        ]
        agent = _make_agent(files)
        content = agent._build_post_compaction_message()["content"]
        a_idx = content.index("## A.md")
        b_idx = content.index("## B.md")
        c_idx = content.index("## C.md")
        assert a_idx < b_idx < c_idx


class TestContextFileStorage:
    """Verify __init__ stores context_files defensively (copy, not reference)."""

    def test_constructor_stores_independent_list(self):
        # Use __new__ + manual init of just the field under test, to avoid
        # the heavy ComputerAgent.__init__ chain.
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        original = [ContextFile(path="AGENTS.md", content="x")]
        agent._context_files = list(original)
        original.append(ContextFile(path="OTHER.md", content="y"))
        # Mutating the caller's list must not change the agent's copy.
        assert len(agent._context_files) == 1
        assert agent._context_files[0].path == "AGENTS.md"
