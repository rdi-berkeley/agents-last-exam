from __future__ import annotations

from copy import deepcopy
from itertools import combinations, permutations

import pytest

from tasks.engineering.robotics_blender_tabletop_reconstruction import main


@pytest.fixture
def reference_spec():
    return {
        "objects": [
            {
                "name": "mug",
                "position": [-0.2, -0.1, 0.44],
                "rotation_deg": [0, 0, 120],
                "material": {
                    "base_color_rgb": [0.85, 0.85, 0.85],
                    "roughness": 0.3,
                    "metallic": 0.1,
                },
            }
        ],
        "lighting": {
            "key_light": {"type": "AREA", "position": [2.0, -1.0, 3.0], "energy": 200},
            "fill_light": {"type": "AREA", "position": [-2.0, 1.0, 2.5], "energy": 80},
            "rim_light": {"type": "AREA", "position": [0.0, 2.0, 2.0], "energy": 120},
        },
    }


def candidate_light(reference_light):
    return {
        "type": reference_light["type"],
        "location": list(reference_light["position"]),
        "energy": reference_light["energy"],
    }


@pytest.mark.parametrize(
    "light_names",
    [
        selected
        for count in range(4)
        for selected in combinations(("key_light", "fill_light", "rim_light"), count)
    ],
)
def test_each_missing_reference_light_loses_only_its_share(reference_spec, light_names):
    lights = [candidate_light(reference_spec["lighting"][name]) for name in light_names]

    assert main._lighting_score(reference_spec, {"lights": lights}) == pytest.approx(
        len(light_names) / 3
    )


def test_missing_lights_field_scores_zero(reference_spec):
    assert main._lighting_score(reference_spec, {}) == 0.0


@pytest.mark.parametrize("candidate_count", [1, 2, 3])
def test_a_candidate_cannot_be_reused_for_multiple_reference_lights(
    reference_spec, candidate_count
):
    reference_light = reference_spec["lighting"]["key_light"]
    reference_spec["lighting"] = {
        name: deepcopy(reference_light) for name in ("key_light", "fill_light", "rim_light")
    }
    lights = [candidate_light(reference_light) for _ in range(candidate_count)]
    lights.extend({"type": "SUN", "location": [0, 0, 0], "energy": 1} for _ in range(3))

    assert main._lighting_score(reference_spec, {"lights": lights}) == pytest.approx(
        candidate_count / 3
    )


def test_matching_finds_global_optimum_independent_of_candidate_order():
    reference_spec = {
        "lighting": {
            "key_light": {"type": "AREA", "position": [0, 0, 0], "energy": 100},
            "fill_light": {"type": "AREA", "position": [0.4, 0, 0], "energy": 100},
            "rim_light": {"type": "AREA", "position": [10, 10, 10], "energy": 100},
        }
    }
    lights = [
        {"type": "AREA", "location": [0.2, 0, 0], "energy": 100},
        {"type": "AREA", "location": [-0.4, 0, 0], "energy": 100},
        candidate_light(reference_spec["lighting"]["rim_light"]),
    ]

    for ordered_lights in permutations(lights):
        assert main._lighting_score(reference_spec, {"lights": list(ordered_lights)}) == 1.0


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("offset, expected", [(0.49, 1 / 3), (0.5, 0.25), (2.0, 0.25)])
def test_position_tolerance_and_per_axis_credit_are_preserved(
    reference_spec, axis, offset, expected
):
    light = candidate_light(reference_spec["lighting"]["key_light"])
    light["location"][axis] += offset

    assert main._lighting_score(reference_spec, {"lights": [light]}) == pytest.approx(expected)


@pytest.mark.parametrize(
    "energy, expected",
    [(140, 1 / 3), (260, 1 / 3), (139, 0.25), (261, 0.25)],
)
def test_energy_tolerance_and_partial_credit_are_preserved(reference_spec, energy, expected):
    light = candidate_light(reference_spec["lighting"]["key_light"])
    light["energy"] = energy

    assert main._lighting_score(reference_spec, {"lights": [light]}) == pytest.approx(expected)


@pytest.mark.parametrize("light_type, expected", [("area", 1 / 3), ("POINT", 0.0), ("", 0.0)])
def test_only_matching_light_types_receive_credit(reference_spec, light_type, expected):
    light = candidate_light(reference_spec["lighting"]["key_light"])
    light["type"] = light_type

    assert main._lighting_score(reference_spec, {"lights": [light]}) == pytest.approx(expected)


@pytest.mark.parametrize("energy, expected", [(200, 1 / 12), (1000, 0.0)])
def test_incorrect_positions_keep_only_existing_energy_credit(reference_spec, energy, expected):
    light = {"type": "AREA", "location": [100, 100, 100], "energy": energy}

    assert main._lighting_score(reference_spec, {"lights": [light]}) == pytest.approx(expected)


@pytest.mark.parametrize("candidate_count", [0, 1, 2, 3])
def test_unrelated_extra_lights_do_not_reduce_or_inflate_credit(reference_spec, candidate_count):
    lights = [candidate_light(light) for light in reference_spec["lighting"].values()]
    metrics = {
        "lights": lights[:candidate_count]
        + [
            {"type": "AREA", "location": [100, 100, 100], "energy": 1000},
            {"type": "SUN", "location": [2.0, -1.0, 3.0], "energy": 200},
        ]
    }
    original_metrics = deepcopy(metrics)
    original_reference = deepcopy(reference_spec)

    assert main._lighting_score(reference_spec, metrics) == pytest.approx(candidate_count / 3)
    assert metrics == original_metrics
    assert reference_spec == original_reference


@pytest.mark.parametrize("candidate_count", [0, 1, 2, 3])
def test_lighting_retains_its_fifteen_percent_weight(reference_spec, candidate_count):
    reference_object = reference_spec["objects"][0]
    exports = ("verification_render", "mustard_bottle", "mug", "potted_meat_can", "full_scene")
    metrics = {
        "objects": {
            "mug": {
                "location": reference_object["position"],
                "rotation_deg": reference_object["rotation_deg"],
                "material": reference_object["material"],
            }
        },
        "exports": dict.fromkeys(exports, True),
        "lights": [
            candidate_light(light)
            for light in list(reference_spec["lighting"].values())[:candidate_count]
        ],
    }

    scores = main._final_score(reference_spec, metrics, 0.6, dict.fromkeys(exports, 10_000))

    assert scores == pytest.approx(
        {
            "spatial_score": 1.0,
            "rotation_score": 1.0,
            "material_score": 1.0,
            "export_score": 1.0,
            "lighting_score": candidate_count / 3,
            "vlm_render_score": 0.6,
            "final_score": 0.15 + 0.10 + 0.10 + 0.05 + 0.05 * candidate_count + 0.45 * 0.6,
            "pass_threshold": 0.70,
        }
    )
