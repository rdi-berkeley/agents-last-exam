from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

TASK_MODULES = (
    "tasks.life_sciences.genomic_interval_processing_1",
    "tasks.health_medicine.obermeyer_bias_reproduction",
    "tasks.computing_math.dit_pipeline_cfg_alignment_fid_256_001",
)


@pytest.mark.parametrize("module_order", list(itertools.permutations(TASK_MODULES)))
@pytest.mark.parametrize("unrelated_module", [False, True])
def test_task_scorers_remain_scoped_in_shared_process(module_order, unrelated_module):
    script = f"""
import importlib
import sys
import types

unrelated = types.ModuleType('score_outputs')
if {unrelated_module!r}:
    sys.modules['score_outputs'] = unrelated
before = list(sys.path)
for task in {module_order!r}:
    main = importlib.import_module(task + '.main')
    scorer = importlib.import_module(task + '.scripts.score_outputs')
    if task.endswith('genomic_interval_processing_1'):
        assert main.score_submission is scorer.score_submission
        assert main.INPUT_BED_FILES is scorer.INPUT_BED_FILES
        assert main.REQUIRED_FILES is scorer.REQUIRED_FILES
    elif task.endswith('obermeyer_bias_reproduction'):
        assert main.score_output_bundle is scorer.score_output_bundle
        assert main.ScoreResult is scorer.ScoreResult
    else:
        assert main.score_submission_text is scorer.score_submission_text
        assert main.ScoreResult is scorer.ScoreResult
    assert len(main.load()) >= 1
if {unrelated_module!r}:
    assert sys.modules['score_outputs'] is unrelated
else:
    assert 'score_outputs' not in sys.modules
assert sys.path == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module_order", list(itertools.permutations(TASK_MODULES)))
def test_runner_loads_task_qualified_scorers(module_order):
    script = f"""
from pathlib import Path
from ale_run.tasks.loader import TaskLoader

for task in {module_order!r}:
    loader = TaskLoader(str(Path(*task.split('.'))))
    loaded = loader.load()
    main = loader._load_module()
    if task.endswith('genomic_interval_processing_1'):
        score = main.score_submission
    elif task.endswith('obermeyer_bias_reproduction'):
        score = main.score_output_bundle
    else:
        score = main.score_submission_text
    assert score.__module__ == task + '.scripts.score_outputs'
    assert loaded['metadata'] == main.load()[0].metadata
    assert loaded['description'] == main.config.task_description
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
