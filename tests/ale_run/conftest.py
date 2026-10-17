import pytest


@pytest.fixture
def aliased_task_root(tmp_path):
    task_root = tmp_path / "tasks/fixture_domain/canonical_answer_task"
    task_root.mkdir(parents=True)
    (task_root / "main.py").write_text(
        '''import json
import os
from types import SimpleNamespace


def load():
    root = os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle")
    base = f"{root}/fixture_domain/neutral_workspace/base"
    output = f"{base}/{os.environ.get('REMOTE_OUTPUT_DIR', 'output')}"
    metadata = {
        "task_id": "fixture_domain/canonical_answer_task",
        "domain_name": "fixture_domain",
        "task_name": "canonical_answer_task",
        "variant_name": "base",
        "sandbox_task_name": "neutral_workspace",
        "requires_task_data": True,
        "input_dir": f"{base}/input",
        "reference_dir": f"{base}/reference",
        "remote_output_dir": output,
        "input_file": f"{base}/input/input.txt",
        "settings_json": f"{base}/input/settings.json",
        "runtime_pyproject": f"{base}/input/runtime_env/pyproject.toml",
        "answer_file": f"{output}/answer.json",
        "verification_targets_file": f"{base}/reference/metadata/verification_targets.json",
    }
    return [SimpleNamespace(
        description=f"Write the synthetic answer to {output}/answer.json.",
        metadata=metadata,
        computer={"setup_config": {"os_type": "linux"}},
    )]


async def start(task_cfg, session):
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])


async def evaluate(task_cfg, session):
    metadata = task_cfg.metadata
    if not await session.file_exists(metadata["answer_file"]):
        return [0.0]
    reference = json.loads(await session.read_bytes(metadata["verification_targets_file"]))
    answer = json.loads(await session.read_bytes(metadata["answer_file"]))
    return [float(answer == reference["answer"])]
'''
    )
    return task_root
