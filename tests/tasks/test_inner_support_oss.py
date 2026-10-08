import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.inner_support_elevation_optimization import main, verification


@pytest.fixture(scope="module")
def original_sources():
    directory = (
        Path(__file__).resolve().parents[2]
        / "task-data-hf/extracted/engineering/inner_support_elevation_optimization/base/input"
    )
    if not directory.is_dir():
        pytest.skip("Original public input bundle is needed for source-transcription controls")
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def answer():
    result = {}
    for position in range(5):
        result[f"pos_{position}m_max_settlement_mm"] = 150 + position
        result[f"pos_{position}m_max_disp_mm"] = 130 + position
    result.update(
        reduction_0m_to_2m_percent=-1.3,
        increase_2m_to_4m_percent=1.3,
        best_support_position_for_minimum_settlement_m=0,
        best_support_position_for_minimum_disp_m=0,
    )
    return result


def summary(candidate):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["max_disp_mm", "max_settlement_mm", "case_label", "support_position_m"])
    for position in reversed(range(5)):
        case = f"pos_{position}m"
        writer.writerow(
            [
                candidate[f"{case}_max_disp_mm"],
                candidate[f"{case}_max_settlement_mm"],
                case,
                position,
            ]
        )
    return stream.getvalue()


def test_discoverable_task_and_card_validation_status_agree():
    task = main.load()[0]
    assert task.computer["setup_config"]["os_type"] == "linux"
    assert task.metadata["release_status"] == "native_entrypoint_and_agent_review_passed"
    assert task.metadata["input_dir"].startswith("/")
    assert "/opt/ale-kratos/bin/python" in task.description
    assert len(verification.RESPONSES + verification.DERIVED) == 14
    card = json.loads(main.ASSETS.parent.joinpath("task_card.json").read_text())
    prompt = (main.ASSETS / "instruction.md").read_text()
    config = main.InnerSupportElevationConfig()
    assert task.description == prompt.replace("{input_dir}", config.input_dir).replace(
        "{output_dir}", config.remote_output_dir
    )
    assert card["releaseStatus"] == task.metadata["release_status"]
    assert card["evaluatorCredentials"] == []
    assert card["vm"]["snapshot"] == "cpu-free-ubuntu"
    manifest = json.loads((main.ASSETS / "source_manifest.json").read_text())
    assert set(manifest["sha256"]).issubset({entry["name"] for entry in card["inputFiles"]})


def test_task_card_prompt_matches_current_instruction():
    card = json.loads(main.ASSETS.parent.joinpath("task_card.json").read_text())
    prompt = (main.ASSETS / "instruction.md").read_text()
    assert card["taskPrompt"] == prompt.replace("{input_dir}", "base/input").replace(
        "{output_dir}", "base/output"
    )


def test_public_numerical_review_split_preserves_original_refinement_and_weights(original_sources):
    original = original_sources["benchmark_model_spec_en.md"].decode()
    assert (
        "Use local refinement around the excavation, retaining walls, and support zones."
        in original
    )
    assert "mesh must be fine enough to reproduce the benchmark within tolerance" in original
    definition = (main.ASSETS / "model_definition.md").read_text()
    instruction = (main.ASSETS / "instruction.md").read_text()
    delivery = (main.ASSETS / "delivery_schema.md").read_text()
    assert "local-refinement requirement remains mandatory" in definition
    assert "further numerical validation" in definition
    assert "further numerical validation" in instruction
    assert "numerical_validation_followup" not in definition
    assert "numerical_validation_followup" not in instruction
    assert "support-decision" in delivery
    assert "native_model_contract.py" in instruction
    assert "50% numerical and 50% engineering delivery" in instruction
    assert "max(0.30 mm, 5%)" in instruction
    assert "Observed errors and invalid" in definition


def test_original_source_integrity(original_sources):
    verification.verify_source_bundle(original_sources)


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_model_spec_en.md",
        "A_zone_structural_materials_en.md",
        "figure_2_4_2_5_support_layout_and_section.png",
        "figure_2_6_monitoring_layout.png",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_incomplete_or_replaced_sources_are_preparation_errors(original_sources, name, mutation):
    files = dict(original_sources)
    if mutation == "missing":
        files.pop(name)
    else:
        files[name] += b" changed"
    with pytest.raises(verification.SourceBundleError, match=name):
        verification.verify_source_bundle(files)


def test_all_fourteen_material_rows_match_original_source(original_sources):
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    source_rows = {}
    for line in original_sources["benchmark_model_spec_en.md"].decode().splitlines():
        cells = line.split()
        if not cells or cells[0] not in {row[0] for row in benchmark["soils"]}:
            continue
        values = (
            [*map(float, cells[-8:-2]), None]
            if cells[-2:] == ["not", "listed"]
            else list(map(float, cells[-7:]))
        )
        source_rows[cells[0]] = [cells[0], *values]
    assert list(source_rows.values()) == benchmark["soils"]
    assert len(source_rows) == 14
    assert [row[0] for row in benchmark["soils"] if row[-1] is None] == ["4-1", "4-2"]


def test_stage_depths_and_structural_constants_match_original_source(original_sources):
    import re

    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    source = original_sources["benchmark_model_spec_en.md"].decode()
    schedule = [
        (int(stage), float(water), float(depth))
        for stage, water, depth in re.findall(
            r"Stage (\d+): Lower groundwater to -([\d.]+) m,? (?:and )?"
            r"excavate to (?:the (?:outer|inner) pit base at )?-([\d.]+) m",
            source,
        )
    ]
    assert schedule == [
        (stage["stage"], stage["water_depth_m"], stage["excavation_depth_m"])
        for stage in benchmark["stages"]
        if "water_depth_m" in stage
    ]
    assert len(schedule) == 5
    structures = benchmark["structures"]
    rows = []
    for line in original_sources["A_zone_structural_materials_en.md"].decode().splitlines():
        cells = line.split()
        if len(cells) >= 4 and cells[-4] in ("Plate", "Beam"):
            rows.append(list(map(float, cells[-3:])))
    assert len(rows) == 4
    assert all(
        row
        == [
            structures["unit_weight_kN_m3"],
            structures["young_modulus_GPa"],
            structures["poisson_ratio"],
        ]
        for row in rows
    )


def test_choices_are_not_replaced_by_old_fixed_model():
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    for old_requirement in (
        "x_coordinates",
        "y_coordinates",
        "outer_polygon",
        "inner_polygon",
        "beam_width",
        "beam_height",
        "drawdown_transition_m",
        "brace_x_coordinates",
        "longitudinal_brace_y",
        "outer_support_depths",
        "inner_wall_toe_depth",
    ):
        assert old_requirement not in benchmark
    assert benchmark["geometry"]["exact_plan_coordinates"] is None
    assert benchmark["observation_source"]["surveyed_coordinates"] is None
    assert benchmark["structures"]["outer_cross_pit_support_levels"] == 4
    annotations = benchmark["section_annotations"]
    datum = annotations["inferred_depth_datum_absolute_elevation_m"]
    for pit in ("outer", "inner"):
        assert annotations[f"{pit}_base_absolute_elevation_m"] + benchmark["geometry"][
            f"{pit}_excavation_depth_m"
        ] == pytest.approx(datum)
    assert annotations["wall_top_absolute_elevation_m"] != datum


def test_dimensioned_embedment_is_a_declared_decision_not_a_recovered_ratio(original_sources):
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    decisions = benchmark["adaptation_clarifications"]
    embedment = decisions["inner_wall_embedment"]
    annotations = benchmark["section_annotations"]
    geometry = benchmark["geometry"]
    assert decisions["kind"] == "explicit_new_task_decisions_not_recovered_historical_definitions"
    assert (
        "Inner wall embedment ratio n = 0.40"
        in original_sources["task_specific_brief_en.md"].decode()
    )
    assert embedment["legacy_n_label"] == geometry["inner_embedment_ratio_n"] == 0.4
    assert embedment["legacy_n_role"] == "rounded_study_label_only_no_exact_ratio_constraint"
    assert embedment["recovered_n_denominator"] is None
    assert embedment["historical_study_toe_proven"] is False
    assert (
        embedment["inner_base_absolute_elevation_m"]
        == annotations["inner_base_absolute_elevation_m"]
        == -15.1
    )
    assert (
        embedment["inner_wall_toe_absolute_elevation_m"]
        == annotations["drawn_inner_pile_toe_absolute_elevation_m"]
        == -21.6
    )
    assert embedment["embedment_below_inner_base_m"] == pytest.approx(
        embedment["inner_base_absolute_elevation_m"]
        - embedment["inner_wall_toe_absolute_elevation_m"]
    )
    assert embedment["embedment_below_inner_base_m"] == 6.5
    assert embedment["toe_depth_below_depth_datum_m"] == pytest.approx(
        annotations["inferred_depth_datum_absolute_elevation_m"]
        - embedment["inner_wall_toe_absolute_elevation_m"]
    )
    assert embedment["toe_depth_below_depth_datum_m"] == 28.1
    assert embedment["embedment_below_inner_base_m"] / geometry[
        "outer_excavation_depth_m"
    ] != pytest.approx(embedment["legacy_n_label"])
    assert geometry["depth_ratio_beta"] == 0.3
    assert geometry["outer_excavation_depth_m"] == 16.6
    assert geometry["inner_excavation_depth_m"] == 21.6
    assert geometry["inner_equivalent_wall_thickness_m"] == 0.62


@pytest.mark.parametrize("position", range(5))
def test_case_heights_change_only_the_support_not_embedment(original_sources, position):
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    decisions = benchmark["adaptation_clarifications"]
    assert benchmark["case_positions_below_inner_top_m"] == list(range(5))
    assert f"- pos_{position}m" in original_sources["task_specific_brief_en.md"].decode()
    support = decisions["inner_support_installation"]["support_absolute_elevations_m"][position]
    assert support == pytest.approx(
        benchmark["section_annotations"]["outer_base_absolute_elevation_m"] - position
    )
    embedment = decisions["inner_wall_embedment"]
    assert embedment["rule"] == "dimensioned_inner_base_and_bored_pile_toe_control_all_five_cases"
    assert support > embedment["inner_base_absolute_elevation_m"]
    assert embedment["inner_base_absolute_elevation_m"] - embedment[
        "inner_wall_toe_absolute_elevation_m"
    ] == pytest.approx(6.5)


def test_construction_order_correction_preserves_single_inner_system(original_sources):
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    source = original_sources["benchmark_model_spec_en.md"].decode()
    assert "and construct the fifth ring beam/support level\nStage 8: Construct" in source
    assert [stage["stage"] for stage in benchmark["stages"]] == list(range(1, 10))
    assert benchmark["stages"][6]["event"] == (
        "dewater_complete_outer_excavation_transition_to_inner_system"
    )
    assert benchmark["stages"][7]["event"] == (
        "construct_inner_wall_and_crown_ring_beam_install_position_0_support"
    )
    installation = benchmark["adaptation_clarifications"]["inner_support_installation"]
    assert installation["count_per_case"] == benchmark["structures"]["inner_support_levels"] == 1
    assert benchmark["structures"]["outer_cross_pit_support_levels"] == 4
    assert installation["requires_inner_wall_and_exposed_case_level"] is True
    assert installation["requires_equilibrium_before_further_excavation"] is True
    assert (
        installation["position_0_installation"] == "stage_8_after_inner_wall_before_inner_digging"
    )
    assert installation["positions_1_to_4_installation"] == (
        "stage_9_substeps_at_case_level_before_excavating_below_it"
    )


def test_all_monitor_conventions_are_public_and_keep_source_registration_open():
    benchmark = json.loads((main.ASSETS / "benchmark.json").read_text())
    observation = benchmark["observation_source"]
    assert observation["drawing_labels"]["ground"] == [f"DBC{number}" for number in range(1, 11)]
    assert observation["drawing_labels"]["outer_wall"] == [f"CX{number}" for number in range(1, 11)]
    assert observation["surveyed_coordinates"] is None
    monitoring = benchmark["adaptation_clarifications"]["monitoring"]
    assert monitoring["scope"] == "all_drawn_DBC_stations_and_CX_outer_wall_profiles"
    assert monitoring["ground_sampling"] == "DBC_stations_not_unspecified_continuous_transects"
    assert monitoring["wall_profile_extent"] == "modeled_outer_wall_top_to_toe"
    assert monitoring["displacement_baseline"] == "equilibrated_end_of_stage_1_before_construction"
    assert monitoring["construction_displacement_resets_allowed"] is False
    assert monitoring["mapping_fixed_before_viewing_results"] is True
    assert monitoring["same_mapping_across_cases"] is True
    assert monitoring["sampling_convergence_required"] is True
    assert monitoring["report_units"] == "mm"
    for filename in ("instruction.md", "model_definition.md"):
        text = (main.ASSETS / filename).read_text()
        assert monitoring["settlement_formula"] in text
        assert monitoring["wall_movement_formula"] in text
        assert "6.50 m" in text
        assert "rounded legacy study label" in text


def test_declarative_native_checks_remain_explicitly_pending():
    contract = json.loads((main.ASSETS / "compliance_requirements.json").read_text())
    assert contract["status"] == "design_only_no_native_validator"
    assert len({row["id"] for row in contract["requirements"]}) == len(contract["requirements"])
    assert all(row["implementation"] == "pending" for row in contract["requirements"])
    assert not hasattr(verification, "numeric_scores")
    assert not hasattr(verification, "read_fields")


def test_json_and_csv_order_do_not_affect_syntax():
    candidate = dict(reversed(list(answer().items())))
    candidate[verification.DERIVED[-1]] = 0.0
    parsed = verification.read_answer(json.dumps(candidate))
    assert parsed == candidate
    assert len(verification.read_csv(summary(candidate), parsed)) == 5


@pytest.mark.parametrize("value", [True, "0", 0.5, float("nan"), float("inf")])
def test_invalid_best_position_is_rejected(value):
    candidate = answer()
    candidate[verification.DERIVED[-1]] = value
    with pytest.raises(ValueError):
        verification.read_answer(json.dumps(candidate))


@pytest.mark.parametrize("value", [True, "150", None, 0, -1, float("nan"), float("inf")])
def test_invalid_magnitude_is_rejected(value):
    candidate = answer()
    candidate[verification.RESPONSES[0]] = value
    with pytest.raises(ValueError):
        verification.read_answer(json.dumps(candidate))


def test_duplicate_json_keys_are_invalid():
    with pytest.raises(ValueError, match="Duplicate"):
        verification.parse_json('{"a": 1, "a": 2}')


def test_nonfinite_unrecognized_json_is_also_invalid():
    with pytest.raises(ValueError, match="Nonfinite"):
        verification.parse_json('{"extra": NaN}')


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("150", "151"),
        lambda text: text.replace("pos_1m", "pos_0m"),
        lambda text: text.replace("max_disp_mm", "max_settlement_mm"),
        lambda text: text.replace("130,150,pos_0m,0", "nan,150,pos_0m,0"),
        lambda text: text.replace("130,150,pos_0m,0", "130,150,pos_0m"),
        lambda text: text.replace("130,150,pos_0m,0", "130,150,pos_0m,0,extra"),
    ],
)
def test_malformed_or_contradictory_csv_is_rejected(mutation):
    with pytest.raises(ValueError):
        verification.read_csv(mutation(summary(answer())), answer())


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", [main.start, main.evaluate])
async def test_no_submission_reference_or_claim_can_enable_scoring(hook):
    class UntrustedSession:
        def __getattr__(self, name):
            pytest.fail(f"Unreleased hook accessed candidate/reference/runtime through {name}")

    task = SimpleNamespace(
        metadata={
            "remote_output_dir": "/untrusted",
            "reference_dir": "/old-arbitrary-reference",
            "trusted_replay": True,
            "release_status": "ready",
            "compliant": True,
        }
    )
    with pytest.raises(verification.EvaluationUnavailableError, match="not a solver score"):
        await hook(task, UntrustedSession())


@pytest.mark.asyncio
async def test_consistent_reports_are_not_physical_evidence():
    candidate = verification.read_answer(json.dumps(answer()))
    assert len(verification.read_csv(summary(candidate), candidate)) == 5
    with pytest.raises(verification.EvaluationUnavailableError):
        await main.evaluate(SimpleNamespace(metadata={}), None)
