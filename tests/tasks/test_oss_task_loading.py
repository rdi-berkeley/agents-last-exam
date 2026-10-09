from pathlib import Path

import pytest

from ale_run.tasks.loader import TaskLoader


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "task_id",
    [
        "engineering/2d_drawings_to_3d_building_model",
        "engineering/inner_support_elevation_optimization",
        "engineering/cailian_road_highway_alignment_2",
        "engineering/gcode",
        "visual_media/music_transcription",
        "visual_media/project_migration",
    ],
)
def test_native_task_loads_through_runner_with_linux_routing(task_id):
    loader = TaskLoader(str(REPOSITORY / "tasks" / task_id))
    loaded = loader.load()
    assert loaded["description"].strip()
    assert loaded["os_type"] == "linux"
    assert loaded["computer"]["setup_config"]["os_type"] == "linux"
    assert loaded["image_category"] == "cpu-free-ubuntu"
    assert callable(loader.get_setup_fn())
    assert callable(loader.get_evaluate_fn())
