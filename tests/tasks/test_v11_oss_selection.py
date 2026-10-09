import json
from pathlib import Path

import pytest

from ale_run.tasks.loader import TaskLoader


ROOT = Path(__file__).resolve().parents[2]
RETIRED_TASK = "engineering/mold-flow"
OSS_VARIANTS = {
    "engineering/2d_drawings_to_3d_building_model": "base",
    "engineering/cailian_road_highway_alignment_2": "base",
    "engineering/gcode": "125162_319",
    "engineering/inner_support_elevation_optimization": "base",
    "visual_media/music_transcription": "dorico_prelude",
    "visual_media/project_migration": "celeste_symphonic_suite",
}
DOCKER_EXCLUSIONS = {
    "engineering/gcode",
    "engineering/cailian_road_highway_alignment_2",
    "engineering/inner_support_elevation_optimization",
    "engineering/openroad_sky130_ibex_pnr_signoff",
    "computing_math/k8s_migration_1",
    "business_finance/bpmn_supply_disruption_l3",
    "business_finance/bpmn_category_governance_restructuring_l3",
    "health_medicine/scene3_skullstrip_qc",
    "psychology_neuro/scene2_resample",
}
CPU_EXCLUSIONS = {
    "engineering/humanoid_velocity_tracking_policy",
    "visual_media/butterfly_flap_animation",
    "visual_media/chroma_key_from_reference",
    "visual_media/compress_3dgs_scene_ply",
    "visual_media/skeletal_animation_reproduction",
}


@pytest.fixture
def selections():
    return {
        path.relative_to(ROOT / "selected_tasks").as_posix(): [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for path in (ROOT / "selected_tasks").rglob("*.txt")
    }


def test_curated_lists_only_select_existing_unique_tasks(selections):
    for name, task_ids in selections.items():
        assert len(task_ids) == len(set(task_ids)), name
        assert RETIRED_TASK not in task_ids, name
        for task_id in task_ids:
            assert (ROOT / "tasks" / task_id / "main.py").is_file(), (name, task_id)


def test_release_selection_matches_task_routing(selections):
    full = set(selections["full.txt"])
    assert len(full) == 151
    assert full == set(selections["full/overall.txt"])
    published = json.loads((ROOT / "tasks/published_tasks.json").read_text())["published"]
    assert full == set(published)
    assert all("licensed" not in name for name in selections)

    snapshots = {
        task_id: json.loads((ROOT / "tasks" / task_id / "task_card.json").read_text())["vm"][
            "snapshot"
        ]
        for task_id in full
    }
    assert set(snapshots.values()) <= {"cpu-free", "cpu-free-ubuntu", "gpu-free"}
    linux = {task_id for task_id, snapshot in snapshots.items() if snapshot == "cpu-free-ubuntu"}
    assert len(linux) == 111
    assert set(selections["ale_cli.txt"]) == linux
    assert set(selections["docker_support.txt"]) == linux - DOCKER_EXCLUSIONS
    assert len(selections["docker_support.txt"]) == 102
    assert set(selections["cpu.txt"]) == full - CPU_EXCLUSIONS
    assert len(selections["cpu.txt"]) == 146
    assert set(selections["qemu_excluded.txt"]) == CPU_EXCLUSIONS
    for name in ("cpu.txt", "full.txt", "ale_cli.txt"):
        assert OSS_VARIANTS.keys() <= set(selections[name]), name
    assert set(selections["docker_support.txt"]) & OSS_VARIANTS.keys() == (
        OSS_VARIANTS.keys() - DOCKER_EXCLUSIONS
    )


@pytest.mark.parametrize(
    "track,count,variants",
    [
        ("overall", 151, set(OSS_VARIANTS)),
        ("near-term", 67, set()),
        (
            "full-spectrum",
            54,
            {"engineering/cailian_road_highway_alignment_2", "visual_media/music_transcription"},
        ),
        (
            "last-exam",
            37,
            {
                "engineering/2d_drawings_to_3d_building_model",
                "engineering/gcode",
                "engineering/inner_support_elevation_optimization",
                "visual_media/project_migration",
            },
        ),
    ],
)
def test_release_tracks_keep_existing_difficulty_assignments(selections, track, count, variants):
    tasks = set(selections[f"full/{track}.txt"])
    assert len(tasks) == count
    assert tasks <= set(selections["full.txt"])
    assert tasks & OSS_VARIANTS.keys() == variants


def test_difficulty_tracks_cover_complete_release(selections):
    tracks = ("near-term", "full-spectrum", "last-exam")
    assert set.union(*(set(selections[f"full/{track}.txt"]) for track in tracks)) == set(
        selections["full.txt"]
    )


@pytest.mark.parametrize(
    "task_id,track",
    [
        ("business_finance/odoo", "full-spectrum"),
        ("visual_media/inkscape_cultural_poster_design", "full-spectrum"),
        ("engineering/robotics_blender_tabletop_reconstruction", "last-exam"),
        ("visual_media/atlas_outpost_graybox_navigation", "last-exam"),
        ("visual_media/human_mesh_animation_reproduction", "last-exam"),
    ],
)
def test_canonical_tracks_keep_all_assigned_tasks(selections, task_id, track):
    assert task_id in selections[f"full/{track}.txt"]
    assert task_id in selections["full/overall.txt"]
    assert task_id in selections["full.txt"]


@pytest.mark.parametrize("task_id,variant", OSS_VARIANTS.items())
def test_published_oss_variant_matches_runner_selection(task_id, variant):
    published = json.loads((ROOT / "tasks/published_tasks.json").read_text())["published"]
    assert published[task_id]["variants"] == [variant]
    loader = TaskLoader(str(ROOT / "tasks" / task_id))
    task = loader.build_task_cfg()
    assert task.metadata["variant_name"] == variant
    assert task.computer["setup_config"]["os_type"] == "linux"
    loaded = loader.load()
    assert loaded["os_type"] == "linux"
    assert loaded["image_category"] == "cpu-free-ubuntu"


def test_retired_task_is_not_published_discoverable_or_loadable():
    published = json.loads((ROOT / "tasks/published_tasks.json").read_text())["published"]
    assert RETIRED_TASK not in published
    discovered = {
        path.parent.relative_to(ROOT / "tasks").as_posix()
        for path in (ROOT / "tasks").glob("*/*/main.py")
    }
    assert RETIRED_TASK not in discovered
    assert not (ROOT / "tasks" / RETIRED_TASK / "task_card.json").exists()
    with pytest.raises(FileNotFoundError, match="main.py not found"):
        TaskLoader(str(ROOT / "tasks" / RETIRED_TASK))
