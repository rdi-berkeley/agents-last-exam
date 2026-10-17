import json
from pathlib import Path

import pytest

from ale_run.tasks.loader import TaskLoader


ROOT = Path(__file__).resolve().parents[2]
RETIRED_TASKS = {
    "agriculture_env/crop_rotation_d02",
    "agriculture_env/ndvi_zonal_statistics_d02",
    "business_finance/financial_stmt_reconstruction_aapl_fy2024",
    "business_finance/pe_screening_memo_1",
    "computing_math/branch_bound_atsp",
    "computing_math/ising_post_measurement_1",
    "computing_math/k3_abelian_extensions",
    "computing_math/os_log_permission_guard_v1",
    "education_info/homework_grading_numerical_pdes_instance_02",
    "education_info/moodle_gradebook_closeout_reconciliation",
    "engineering/abb_irb6700_asset_to_urdf_instance_1",
    "engineering/mold-flow",
    "engineering/sanding_performance_scoring_instance_1",
    "health_medicine/Clinical_Variant_Annotation",
    "health_medicine/crf_sdtm_mapping_4",
    "health_medicine/epidemiology_forecast",
    "health_medicine/healthcare_bias_audit_27a_public_replication_v1",
    "health_medicine/healthcare_variant_annotation_pipeline",
    "health_medicine/replicate_paper_1",
    "health_medicine/scene3_skullstrip_qc",
    "life_sciences/amber_minimization_script_prep_instance_1",
    "life_sciences/gene_expression_differential_analysis_functional_enrichment_analysis_1",
    "life_sciences/genomic_interval_processing_1",
    "life_sciences/hg002_chr22_germline_variant_pipeline",
    "life_sciences/protein_function_annotation_instance_1",
    "life_sciences/pseudotime_de",
    "life_sciences/rgi_mcr1_colistin_v2",
    "physical_sciences/adapt_vqe_molecular_energy",
    "transport_safety/capacitated_vehicle_routing_problems",
    "visual_media/inkscape_cultural_poster_design",
}

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
    "psychology_neuro/scene2_resample",
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
    full = set(selections["full.txt"])
    for name, task_ids in selections.items():
        assert len(task_ids) == len(set(task_ids)), name
        assert RETIRED_TASKS.isdisjoint(task_ids), name
        if name in {"hello_both.txt", "tool_smoke_both.txt"}:
            assert all(task_id.startswith("demo/") for task_id in task_ids), name
        else:
            assert set(task_ids) <= full, name
        for task_id in task_ids:
            assert (ROOT / "tasks" / task_id / "main.py").is_file(), (name, task_id)


def test_release_selection_matches_task_routing(selections):
    full = set(selections["full.txt"])
    assert len(full) == 122
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
    assert len(linux) == 86
    assert set(selections["ale_cli.txt"]) == linux
    assert set(selections["docker_support.txt"]) == linux - DOCKER_EXCLUSIONS
    assert len(selections["docker_support.txt"]) == 78
    gpu = {task_id for task_id, snapshot in snapshots.items() if snapshot == "gpu-free"}
    assert set(selections["gpu.txt"]) == gpu
    assert len(gpu) == 3
    assert set(selections["cpu.txt"]) == full - gpu
    assert len(selections["cpu.txt"]) == 119
    for name in ("cpu.txt", "full.txt", "ale_cli.txt"):
        assert OSS_VARIANTS.keys() <= set(selections[name]), name
    assert set(selections["docker_support.txt"]) & OSS_VARIANTS.keys() == (
        OSS_VARIANTS.keys() - DOCKER_EXCLUSIONS
    )


@pytest.mark.parametrize(
    "track,count,variants",
    [
        ("overall", 122, set(OSS_VARIANTS)),
        ("near-term", 45, set()),
        (
            "full-spectrum",
            46,
            {"engineering/cailian_road_highway_alignment_2", "visual_media/music_transcription"},
        ),
        (
            "last-exam",
            36,
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


@pytest.mark.parametrize("task_id", sorted(RETIRED_TASKS))
def test_retired_task_is_not_published_discoverable_or_loadable(task_id):
    published = json.loads((ROOT / "tasks/published_tasks.json").read_text())["published"]
    assert task_id not in published
    discovered = {
        path.parent.relative_to(ROOT / "tasks").as_posix()
        for path in (ROOT / "tasks").glob("*/*/main.py")
    }
    assert task_id not in discovered
    assert not (ROOT / "tasks" / task_id).exists()
    with pytest.raises(FileNotFoundError, match="main.py not found"):
        TaskLoader(str(ROOT / "tasks" / task_id))
