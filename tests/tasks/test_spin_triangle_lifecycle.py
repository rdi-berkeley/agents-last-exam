"""Real ALE loader and task hooks with an in-memory DesktopSession test double."""

from __future__ import annotations

import hashlib
import json
import shlex
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ale_run.tasks.loader import TaskLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = (
    REPO_ROOT / "tasks" / "physical_sciences" / "frustrated_spin_triangle_yang_baxter_holonomy"
)


class MemorySession:
    def __init__(self):
        self.files = {}
        self.directories = set()
        self.interface = self

    async def create_dir(self, path):
        self.directories.add(path)

    async def write_file(self, path, content):
        self.files[path] = content

    async def read_file(self, path):
        return self.files[path]

    async def file_exists(self, path):
        return path in self.files

    async def directory_exists(self, path):
        return path in self.directories

    async def run_command(self, command, check=False):
        tokens = shlex.split(command)
        if tokens[:2] == ["rm", "-rf"]:
            for root in tokens[2:]:
                self.files = {
                    path: value
                    for path, value in self.files.items()
                    if path != root and not path.startswith(root + "/")
                }
                self.directories = {
                    path
                    for path in self.directories
                    if path != root and not path.startswith(root + "/")
                }
        elif tokens[:2] != ["chmod", "755"]:
            raise AssertionError(f"unexpected setup command: {command}")
        return {"success": True, "return_code": 0, "stdout": "", "stderr": ""}


class SpinTriangleLifecycleTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = TaskLoader(str(TASK_DIR))
        cls.info = cls.loader.load()
        cls.task_module = cls.loader._load_module()
        cls.author_answers = cls.task_module.build_fixtures()[1]

    def setUp(self):
        self.cfg = self.loader.build_task_cfg()
        self.session = MemorySession()
        self.meta = self.cfg.metadata

    def test_real_loader_metadata_and_offline_card(self):
        self.assertEqual(self.info["os_type"], "linux")
        self.assertEqual(self.info["image_category"], "cpu-free-ubuntu")
        self.assertEqual(self.info["timeout_s"], 7200)
        self.assertEqual(self.info["vcpus"], 4)
        self.assertEqual(self.info["memory_gb"], 16)
        self.assertEqual(self.info["machine_type"], "c4-standard-4")
        self.assertFalse(self.info["task_data"].requires_task_data)
        self.assertEqual(self.meta["task_id"], f"physical_sciences/{TASK_DIR.name}")
        self.assertTrue(callable(self.loader.get_setup_fn()))
        self.assertTrue(callable(self.loader.get_evaluate_fn()))
        card = json.loads((TASK_DIR / "task_card.json").read_text())
        self.assertEqual(card["vm"]["network"], {"mode": "off"})

    async def test_public_inputs_match_reviewed_snapshot(self):
        await self.loader.get_setup_fn()(self.cfg, self.session)
        expected = {
            "task_spec_path": "29cfa1f5c52f83f2a706f38dd32ef7212e5f8ad26577c9aeedfec9ef548f00ca",
            "calculus_path": "cfd62822b730b92b86d780aec10792dafb690c0aba7b0e43c27e23bc1553f856",
            "scenarios_path": "b81f4f88c403503d5decce8fcc9365685fca34dc1a2fde3b0ff2751c9e3f993c",
            "check_path": "c398b84f424d46100fd11d02027b02b62905a15caf6dd07cde896d2cb65549b1",
        }
        for key, digest in expected.items():
            content = self.session.files[self.meta[key]].encode("utf-8")
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest, key)

    async def test_setup_is_idempotent_and_stages_only_public_files(self):
        for _ in range(2):
            self.session.files[self.meta["answers_path"]] = "stale output"
            self.session.files[self.meta["reference_dir"] + "/answer.json"] = "secret"
            self.session.directories.add(self.meta["reference_dir"])
            await self.loader.get_setup_fn()(self.cfg, self.session)
            expected = {
                self.meta[key]
                for key in ("task_spec_path", "calculus_path", "scenarios_path", "check_path")
            }
            self.assertEqual(set(self.session.files), expected)
            self.assertNotIn(self.meta["reference_dir"], self.session.directories)
            self.assertEqual(len(json.loads(self.session.files[self.meta["scenarios_path"]])), 12)

    async def test_reference_visibility_is_rejected(self):
        self.session.directory_exists = AsyncMock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, "reference data must not exist"):
            await self.loader.get_setup_fn()(self.cfg, self.session)

    async def test_missing_and_malformed_answers_score_zero(self):
        evaluate = self.loader.get_evaluate_fn()
        self.assertEqual(await evaluate(self.cfg, self.session), [0.0])
        self.session.files[self.meta["answers_path"]] = "not JSON"
        self.assertEqual(await evaluate(self.cfg, self.session), [0.0])

    async def test_unparseable_and_nonobject_json_scores_zero(self):
        for value in ("9" * 10000, "[" * 10000 + "]" * 10000, "null", "[]", '"text"'):
            with self.subTest(value=value[:40]):
                self.session.files[self.meta["answers_path"]] = value
                self.assertEqual(await self.loader.get_evaluate_fn()(self.cfg, self.session), [0.0])

    async def test_author_answer_passes_despite_tampered_public_inputs(self):
        await self.loader.get_setup_fn()(self.cfg, self.session)
        self.session.files[self.meta["answers_path"]] = json.dumps(self.author_answers)
        self.session.files[self.meta["scenarios_path"]] = "{}"
        self.session.files[self.meta["check_path"]] = "raise RuntimeError('tampered')"
        self.assertEqual(await self.loader.get_evaluate_fn()(self.cfg, self.session), [1.0])

    async def test_storage_failure_is_not_a_solver_zero(self):
        self.session.files[self.meta["answers_path"]] = "{}"
        self.session.read_file = AsyncMock(side_effect=OSError("storage unavailable"))
        with self.assertRaisesRegex(OSError, "storage unavailable"):
            await self.loader.get_evaluate_fn()(self.cfg, self.session)

    async def test_internal_grader_failure_is_not_a_solver_zero(self):
        self.session.files[self.meta["answers_path"]] = "{}"
        with (
            patch.object(self.task_module, "score_submission", side_effect=RuntimeError("bug")),
            self.assertRaisesRegex(RuntimeError, "bug"),
        ):
            await self.loader.get_evaluate_fn()(self.cfg, self.session)

    async def test_failed_cleanup_propagates(self):
        self.session.run_command = AsyncMock(side_effect=OSError("cleanup failed"))
        with self.assertRaisesRegex(OSError, "cleanup failed"):
            await self.loader.get_setup_fn()(self.cfg, self.session)
        self.assertTrue(self.session.run_command.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
