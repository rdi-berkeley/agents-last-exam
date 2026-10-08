import hashlib

import numpy as np
import pytest

from tasks.engineering.inner_support_elevation_optimization.scripts import (
    native_result_adapter as candidate,
)


MDPA = "Begin ModelPartData\nEnd ModelPartData\nBegin Properties 1\nEnd Properties\nBegin Properties 8\nTHICKNESS 1\nEnd Properties\nBegin Nodes\n1 0 0 -1\n2 2 0 -1\n3 2 3 -1\n4 0 3 -1\n5 0 0 0\n6 2 0 0\n7 2 3 0\n8 0 3 0\n101 0.007 -0.003 -1.008\n102 2.007 -0.003 -1.008\n103 2.007 -0.003 -0.008\n104 0.007 -0.003 -0.008\nEnd Nodes\nBegin Elements UPwSmallStrainElement3D8N\n81 1 1 2 3 4 5 6 7 8\nEnd Elements\nBegin Elements MITCThickShellElement3D4N\n91 8 101 102 103 104\nEnd Elements\nBegin Constraints LinearMasterSlaveConstraint DISPLACEMENT_X DISPLACEMENT_X\n303 -0.007 [1] 101 1\n306 -0.007 [1] 102 2\n309 -0.007 [1] 103 6\n312 -0.007 [1] 104 5\nEnd Constraints\nBegin NodalData DISPLACEMENT_X\n1 0 0.004\n2 0 0.008\n3 0 0.023000000000000003\n4 0 0.013000000000000001\n5 0 0.004\n6 0 0.008\n7 0 0.023000000000000003\n8 0 0.013000000000000001\n101 0 -0.003\n102 0 0.001\n103 0 0.001\n104 0 -0.003\nEnd NodalData\nBegin Constraints LinearMasterSlaveConstraint DISPLACEMENT_Y DISPLACEMENT_Y\n304 0.003 [1] 101 1\n307 0.003 [1] 102 2\n310 0.003 [1] 103 6\n313 0.003 [1] 104 5\nEnd Constraints\nBegin NodalData DISPLACEMENT_Y\n1 0 -0.002\n2 0 -0.004\n3 0 -0.004\n4 0 0.01\n5 0 -0.002\n6 0 -0.004\n7 0 -0.004\n8 0 0.01\n101 0 0.001\n102 0 -0.001\n103 0 -0.001\n104 0 0.001\nEnd NodalData\nBegin Constraints LinearMasterSlaveConstraint DISPLACEMENT_Z DISPLACEMENT_Z\n305 0.008 [1] 101 1\n308 0.008 [1] 102 2\n311 0.008 [1] 103 6\n314 0.008 [1] 104 5\nEnd Constraints\nBegin NodalData DISPLACEMENT_Z\n1 0 -0.007\n2 0 -0.011\n3 0 -0.044000000000000004\n4 0 -0.016\n5 0 -0.007\n6 0 -0.011\n7 0 -0.044000000000000004\n8 0 -0.016\n101 0 0.001\n102 0 -0.002999999999999999\n103 0 -0.002999999999999999\n104 0 0.001\nEnd NodalData\nBegin SubModelPart Active\nBegin SubModelPartElements\n81\n91\nEnd SubModelPartElements\nBegin SubModelPartConstraints\n303\n304\n305\n306\n307\n308\n309\n310\n311\n312\n313\n314\nEnd SubModelPartConstraints\nEnd SubModelPart\nBegin SubModelPart Soil\nBegin SubModelPartNodes\n1\n2\n3\n4\n5\n6\n7\n8\nEnd SubModelPartNodes\nEnd SubModelPart\n"


def analytical_displacement(point):
    east, north = point[:2]
    return np.array(
        [
            0.002 * east + 0.003 * north + 0.001 * east * north,
            -0.001 * east + 0.004 * north - 0.002 * east * north,
            -0.001 - 0.002 * east - 0.003 * north - 0.004 * east * north,
        ]
    )


def load_fixture(tmp_path):
    path = tmp_path / "hex.mdpa"
    path.write_text(MDPA)
    result = candidate.read_native_result(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    baseline = {node: np.array([0.004, -0.002, -0.006]) for node in result.soil_nodes}
    registration = {f"DBC{index}": [index * 0.1, index * 0.15, 0] for index in range(1, 11)}
    registration.update({f"CX{index}": [index * 0.1, 0, 0] for index in range(1, 11)})
    return result, baseline, registration, {"outer_wall_property_ids": [8], "wall_range": [-1, 0]}


def test_full_native_fixture_matches_analytic_bilinear_field(tmp_path):
    result, baseline, registration, kwargs = load_fixture(tmp_path)
    extracted = candidate.extract_monitors(result, baseline, registration, **kwargs)
    assert len(extracted["stations"]) == 10
    assert len(extracted["profiles"]) == 20
    for row in extracted["stations"] + extracted["profiles"]:
        np.testing.assert_allclose(row["u_m"], analytical_displacement(row["xyz_m"]), atol=1e-15)
    assert extracted["max_settlement_mm"] == pytest.approx(13.5, abs=1e-12)
    center = next(row for row in extracted["stations"] if row["monitor"] == "DBC10")
    assert center["element_ids"] == [81]
    diagonal_triangle_value = (
        analytical_displacement((0, 0)) + analytical_displacement((2, 3))
    ) / 2
    assert abs(center["u_m"][2] - diagonal_triangle_value[2]) == pytest.approx(0.006)


def test_spatially_varying_stage1_baseline_is_subtracted_at_native_nodes(tmp_path):
    result, baseline, registration, kwargs = load_fixture(tmp_path)
    before = candidate.extract_monitors(result, baseline, registration, **kwargs)
    for node in result.soil_nodes:
        shift = np.array([0.001 * node, -0.003 * node**2, 0.01 / node])
        baseline[node] = baseline[node] + shift
        for component, value in zip(candidate.COMPONENTS, shift):
            result.fields[component][node] += value
    for node, master in [(101, 1), (102, 2), (103, 6), (104, 5)]:
        for component in candidate.COMPONENTS:
            offset = result.constraints[node, component][1]
            result.fields[component][node] = result.fields[component][master] + offset
    after = candidate.extract_monitors(result, baseline, registration, **kwargs)
    for table in ("stations", "profiles"):
        for original, shifted in zip(before[table], after[table]):
            np.testing.assert_allclose(shifted["u_m"], original["u_m"], atol=1e-15, rtol=0)


@pytest.mark.parametrize(
    "angle,scale,translation", [(0, 1, 0), (0.73, 0.001, 123), (-1.1, 1000, -456)]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_distorted_convex_native_face_reproduces_natural_bilinear_field(
    tmp_path, angle, scale, translation, reverse
):
    result, baseline, _, _ = load_fixture(tmp_path)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    if reverse:
        corners.reverse()

    def physical_point(xi, eta):
        planar = np.array(
            [
                1.4 + 1.2 * xi + 0.2 * eta + 0.3 * xi * eta,
                1.1 + 0.1 * xi + 1.1 * eta - 0.15 * xi * eta,
            ]
        )
        return translation + scale * (rotation @ planar)

    def field_value(xi, eta):
        return np.array(
            [
                0.003 + 0.002 * xi - 0.004 * eta + 0.006 * xi * eta,
                -0.002 + 0.001 * xi * eta,
                -0.02 - 0.007 * eta + 0.009 * xi * eta,
            ]
        )

    for node, (xi, eta) in enumerate(corners, 5):
        result.nodes[node] = np.array([*physical_point(xi, eta), 0])
        for component, value in zip(candidate.COMPONENTS, field_value(xi, eta) + baseline[node]):
            result.fields[component][node] = value
    _, origin, coefficients, face_scale, values = candidate.ground_quadrilaterals(
        result, baseline, 0
    )[0]
    for xi in (-1, -0.7, 0, 0.42, 1):
        for eta in (-1, -0.51, 0, 0.81, 1):
            weights = candidate.quadrilateral_weights(
                origin, coefficients, face_scale, physical_point(xi, eta)
            )
            assert weights is not None
            np.testing.assert_allclose(weights @ values, field_value(xi, eta), atol=2e-11, rtol=0)
    for xi, eta in [(-1.01, 0), (1.01, 0), (0, -1.01), (0, 1.01), (5, 5)]:
        assert (
            candidate.quadrilateral_weights(
                origin, coefficients, face_scale, physical_point(xi, eta)
            )
            is None
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_any_native_cube_face_can_be_the_horizontal_top(tmp_path, axis, sign):
    result, baseline, _, _ = load_fixture(tmp_path)
    remaining = [dimension for dimension in range(3) if dimension != axis]
    original_coordinates = {node: result.nodes[node].copy() for node in range(1, 9)}
    top = max(sign * point[axis] for point in original_coordinates.values())
    for node, point in original_coordinates.items():
        result.nodes[node] = np.array(
            [point[remaining[0]], point[remaining[1]], sign * point[axis] - top]
        )
        for component, value in zip(
            candidate.COMPONENTS, analytical_displacement(result.nodes[node]) + baseline[node]
        ):
            result.fields[component][node] = value
    faces = candidate.ground_quadrilaterals(result, baseline, 0)
    assert len(faces) == 1
    _, origin, coefficients, scale, values = faces[0]
    point = np.array([result.nodes[node][:2] for node in range(1, 9)]).mean(axis=0)
    weights = candidate.quadrilateral_weights(origin, coefficients, scale, point)
    np.testing.assert_allclose(weights @ values, analytical_displacement(point), atol=1e-15)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("inactive", "No active ground interpolation coverage"),
        ("baseline", "Missing Stage1 ground baseline"),
        ("outside", "No active ground interpolation coverage"),
        ("warped", "No active ground interpolation coverage"),
        ("above", "No active ground interpolation coverage"),
        ("folded", "Degenerate or folded"),
        ("collapsed", "Degenerate or folded"),
        ("not_face", "do not form a native hexahedron face"),
        ("wrong_arity", "Malformed native eight-node hexahedron"),
    ],
)
def test_bad_or_uncovered_hex_does_not_gain_interpolation(tmp_path, damage, match):
    result, baseline, registration, kwargs = load_fixture(tmp_path)
    if damage == "inactive":
        result.active.remove(81)
    elif damage == "baseline":
        del baseline[7]
    elif damage == "outside":
        registration["DBC1"] = [2.1, 1, 0]
    elif damage == "warped":
        result.nodes[7][2] = -0.1
    elif damage == "above":
        result.nodes[3][2] = 0.1
    elif damage == "folded":
        result.nodes[7], result.nodes[8] = result.nodes[8], result.nodes[7]
    elif damage == "collapsed":
        result.nodes[7] = result.nodes[8].copy()
    elif damage == "not_face":
        result.nodes[3][2], result.nodes[7][2] = 0, -1
    elif damage == "wrong_arity":
        result.elements[81] = (*result.elements[81][:2], result.elements[81][2][:-1])
    with pytest.raises(ValueError, match=match):
        candidate.extract_monitors(result, baseline, registration, **kwargs)


@pytest.mark.parametrize("damage", [False, True])
def test_mixed_tet_hex_shared_edge_uses_existing_field_agreement(tmp_path, damage):
    result, baseline, registration, kwargs = load_fixture(tmp_path)
    for node, point in zip(range(11, 15), [(2, 0, 0), (4, 0, 0), (2, 3, 0), (2, 0, -1)]):
        result.nodes[node] = np.array(point, dtype=float)
        result.soil_nodes.add(node)
        baseline[node] = baseline[5].copy()
        for component, value in zip(
            candidate.COMPONENTS, analytical_displacement(point) + baseline[node]
        ):
            result.fields[component][node] = value
    result.elements[82] = ("UPwSmallStrainElement3D4N", 1, (11, 12, 13, 14))
    result.active.add(82)
    if damage:
        result.fields["DISPLACEMENT_Z"][13] += 0.001
    for name in registration:
        if name.startswith("DBC"):
            registration[name][0] = 2
    if damage:
        with pytest.raises(ValueError, match="Discontinuous ground interpolation at DBC1"):
            candidate.extract_monitors(result, baseline, registration, **kwargs)
    else:
        extracted = candidate.extract_monitors(result, baseline, registration, **kwargs)
        for row in extracted["stations"]:
            assert set(row["element_ids"]) == {81, 82}
            np.testing.assert_allclose(
                row["u_m"], analytical_displacement(row["xyz_m"]), atol=1e-15
            )


@pytest.mark.parametrize(
    "before,after,match",
    [
        ("7 0 0.023000000000000003", "7 0 nan", "nonfinite native displacement"),
        ("7 0 0.023000000000000003\n", "", "Missing native displacement field/node"),
        ("81 1 1 2 3 4 5 6 7 8", "81 1 1 2 3 4 5 6 7 7", "repeated vertex"),
        ("81 1 1 2 3 4 5 6 7 8", "81 1 1 2 3 4 5 6 7 999", "Unknown element node"),
    ],
)
def test_native_parser_still_rejects_damaged_input(tmp_path, before, after, match):
    path = tmp_path / "damaged.mdpa"
    original = MDPA
    assert before in original
    path.write_text(original.replace(before, after))
    with pytest.raises(ValueError, match=match):
        candidate.read_native_result(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
