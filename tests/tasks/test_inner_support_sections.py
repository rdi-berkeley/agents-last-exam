import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "tasks/engineering/inner_support_elevation_optimization/scripts/native_member_sections.py"
)
SPEC = importlib.util.spec_from_file_location("support_member_sections_test", SCRIPT)
sections = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sections)


@pytest.fixture
def family():
    return {
        "inner_polygon": [[0, 0], [4, 0], [4, 2], [0, 2]],
        "braces": [
            {
                "system": "inner",
                "segments_xy_m": [[[0, 0], [2, 0]], [[2, 0], [4, 0]], [[0, 0], [4, 2]]],
                "crown_segments_xy_m": [[[0, 0], [4, 0]]],
            }
        ],
        "engineering_choices": {
            "beam_section_m": {"width": 1.0, "height": 1.2},
            "crowns": {"inner": {"axis_z_m": -10.1, "section_m": {"width": 1.0, "height": 1.0}}},
        },
    }


def test_boundary_subsegments_use_crown_but_chord_uses_same_support_section_at_five_heights(family):
    for position in range(5):
        members = sections.member_sections(family, "inner", -10.1 - position)
        assert [member["role"] for member in members] == (
            ["crown", "crown", "support"] if position == 0 else ["support"] * 3
        )
        assert sections.section_properties(members[-1]["section_m"])["CROSS_AREA"] == 1.2
    crowns = sections.member_sections(family, "inner", -10.1, crown_only=True)
    assert len(crowns) == 1 and crowns[0]["role"] == "crown"


def test_cross_pit_member_cannot_be_declared_crown(family):
    family["braces"][0]["crown_segments_xy_m"] = [[[0, 0], [4, 2]]]
    with pytest.raises(ValueError, match="perimeter"):
        sections.member_sections(family, "inner", -10.1, crown_only=True)


def test_equivalent_polygon_subdivision_does_not_change_crown_roles(family):
    before = sections.member_sections(family, "inner", -10.1, crown_only=True)
    family["inner_polygon"].insert(1, [1, 0])
    family["inner_polygon"].insert(2, [3, 0])
    assert sections.member_sections(family, "inner", -10.1, crown_only=True) == before
    perimeter = list(
        zip([[0, 0], [1, 0], [1, 1], [3, 1], [3, 0]], [[1, 0], [1, 1], [3, 1], [3, 0], [4, 0]])
    )
    assert not sections.on_perimeter([0, 0], [4, 0], perimeter)


def test_short_rounded_member_uses_distance_to_declared_edge_not_extrapolated_member_line():
    assert sections.on_perimeter(
        [3.00000004, 4.0], [3.03000004, 4.04], [([0, 0], [300, 400])]
    )


def test_declared_section_choices_remain_free_and_axes_preserved(family):
    family["engineering_choices"]["beam_section_m"] = {"width": 0.8, "height": 1.5}
    member = sections.member_sections(family, "inner", -10.1)[-1]
    properties = sections.section_properties(member["section_m"])
    assert properties["CROSS_AREA"] == pytest.approx(1.2)
    assert properties["I22"] == pytest.approx(0.225)
    assert properties["I33"] == pytest.approx(0.064)


def test_native_audit_rejects_actual_wrong_strut_property_independent_of_role_receipt(family):
    class Properties(dict):
        Id = 22

    elements = {}
    members = sections.member_sections(family, "inner", -10.1)
    for identifier, member in enumerate(members, 1):
        elements[identifier] = SimpleNamespace(
            Properties=Properties(sections.section_properties(member["section_m"])),
            Is=lambda flag: True,
        )
    runner = SimpleNamespace(
        family=family,
        part=SimpleNamespace(GetElement=elements.__getitem__),
        Kratos=SimpleNamespace(ACTIVE="active", KratosGlobals=SimpleNamespace(GetVariable=str)),
        receipt={
            "support_installations": [
                {
                    "system": "inner",
                    "label": "stage_8_inner_support_pos_0m",
                    "elevation_m": -10.1,
                    "crown_only": False,
                    "native_element_ids": [1, 2, 3],
                }
            ]
        },
    )
    assert sections.audit_support_sections(runner)["passed"]
    elements[3].Properties["CROSS_AREA"] = 1.0
    with pytest.raises(ValueError, match=r"support\).*CROSS_AREA"):
        sections.audit_support_sections(runner)
