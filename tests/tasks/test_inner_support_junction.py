import numpy as np
import pytest

from tasks.engineering.inner_support_elevation_optimization.scripts import (
    native_result_adapter as candidate,
)


def crown_fixture(axis_z=0.5):
    model = candidate.NativeResult()
    model.nodes = {
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([4.0, 0.0, 0.0]),
        3: np.array([0.0, 4.0, 0.0]),
        4: np.array([0.0, 0.0, -1.0]),
        5: np.array([4.0, 0.0, -1.0]),
    }
    model.soil_nodes = set(model.nodes)
    baseline = {node: np.zeros(3) for node in model.soil_nodes}
    for slave, master in ((201, 4), (202, 5), (203, 2), (204, 1)):
        model.nodes[slave] = model.nodes[master].copy()
        for component, variable in enumerate(candidate.COMPONENTS):
            identifier = 3 * slave + component
            model.constraints[slave, variable] = (identifier, 0.0, (1.0,), (master,))
            model.constraint_variables[slave, variable] = (variable,)
            model.active_constraints.add(identifier)
    model.nodes.update({301: np.array([0.0, 0.0, axis_z]), 302: np.array([4.0, 0.0, axis_z])})
    model.fields = {
        variable: {node: 0.0 for node in model.nodes} for variable in candidate.COMPONENTS
    }
    model.rotations = {
        variable: {node: 0.0 for node in model.nodes} for variable in candidate.ROTATIONS
    }
    model.rotations["ROTATION_Z"][204] = 0.01
    model.rotations["ROTATION_Z"][203] = -0.01
    for node, weights in ((301, (1.0, 0.0)), (302, (0.0, 1.0))):
        for component, variable in enumerate(candidate.COMPONENTS + candidate.ROTATIONS):
            masters = (1, 2) if component < 3 else (204, 203)
            variables = (variable, variable)
            coefficients = weights
            if axis_z and component < 2:
                masters += (204, 203)
                variables += (candidate.ROTATIONS[1 - component],) * 2
                sign = 1 if component == 0 else -1
                coefficients += tuple(sign * axis_z * weight for weight in weights)
            target = model.constraints if component < 3 else model.rotation_constraints
            identifier = 10000 + 6 * node + component
            target[node, variable] = (identifier, 0.0, coefficients, masters)
            model.constraint_variables[node, variable] = variables
            model.active_constraints.add(identifier)
            fields = model.fields if component < 3 else model.rotations
            fields[variable][node] = sum(
                coefficient
                * (model.fields if name in candidate.COMPONENTS else model.rotations)[name][master]
                for coefficient, name, master in zip(coefficients, variables, masters)
            )
    model.elements = {
        81: ("UPwSmallStrainElement3D4N", 1, (1, 2, 3, 4)),
        91: ("MITCThickShellElement3D4N", 8, (201, 202, 203, 204)),
        101: ("CrLinearBeamElement3D2N", 9, (301, 302)),
    }
    model.active = set(model.elements)
    model.properties[9] = {"CROSS_AREA": 1.0, "I22": 1 / 12, "I33": 1 / 12}
    definition = {
        "element": 101,
        "reference_axis_m": [[0, 0, axis_z], [4, 0, axis_z]],
        "reference_frame": np.eye(3).tolist(),
        "native_initial_frame": np.eye(3).tolist(),
        "stage1_axis_displacement_m": [[0, 0, 0], [0, 0, 0]],
        "profile_range_z_m": [0.0, axis_z + 0.5],
    }
    registration = {f"DBC{index}": [0.1, index * 0.1, 0.0] for index in range(1, 11)}
    registration.update({f"CX{index}": [(index - 0.5) * 0.4, 0.0, 0.0] for index in range(1, 11)})
    kwargs = {
        "outer_wall_property_ids": {8},
        "wall_range": (-1.0, axis_z + 0.5),
        "crown_geometry": [definition],
    }
    return model, baseline, registration, kwargs


@pytest.mark.parametrize("axis_z", [-0.25, 0.0, 0.5])
def test_endpoint_coupling_allows_distinct_interior_traces_with_signed_eccentricity(axis_z):
    model, baseline, registration, kwargs = crown_fixture(axis_z)
    actual = candidate.extract_monitors(model, baseline, registration, **kwargs)
    assert len(actual["native_owner_boundaries"]) == 10
    assert len(actual["profiles"]) == 40
    assert actual["max_disp_mm"] == pytest.approx(9.9)
    assert any(
        np.linalg.norm(row["attachment_motion_difference_m"]) > 0.001
        for row in actual["native_owner_boundaries"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "inactive",
        "weights",
        "variables",
        "translation",
        "rotation",
        "baseline",
        "range",
        "overlap",
        "installation",
    ],
)
def test_invalid_crown_attachment_never_gains_the_interior_exception(damage):
    model, baseline, registration, kwargs = crown_fixture()
    key = (301, "DISPLACEMENT_X")
    identifier, offset, weights, masters = model.constraints[key]
    if damage == "missing":
        del model.constraints[key]
    elif damage == "inactive":
        model.active_constraints.remove(identifier)
    elif damage == "weights":
        model.constraints[key] = (identifier, offset, (0.5, 0.5, *weights[2:]), masters)
    elif damage == "variables":
        model.constraint_variables[key] = ("DISPLACEMENT_Y", *model.constraint_variables[key][1:])
    elif damage == "translation":
        model.fields["DISPLACEMENT_X"][301] += 0.001
    elif damage == "rotation":
        model.rotations["ROTATION_Z"][301] += 0.01
    elif damage == "baseline":
        kwargs["crown_geometry"][0]["stage1_axis_displacement_m"][0][0] += 0.001
    elif damage in ("range", "overlap"):
        kwargs["crown_geometry"][0]["profile_range_z_m"][0] += 0.1 if damage == "range" else -0.1
    else:
        for node in (301, 302):
            model.nodes[node] += [0, 0, 0.001]
    with pytest.raises(ValueError):
        candidate.extract_monitors(model, baseline, registration, **kwargs)


@pytest.mark.parametrize("owner", ["shell", "crown"])
def test_maximum_retains_both_junction_traces(monkeypatch, owner):
    model, baseline, registration, kwargs = crown_fixture()
    if owner == "crown":
        original = candidate.beam_physical_point

        def larger(*args):
            return original(*args) + [1, 0, 0]

        monkeypatch.setattr(candidate, "beam_physical_point", larger)
    else:
        original = candidate.wall_panels

        def larger(*args):
            return [(*panel[:6], panel[6] + [1, 0, 0]) for panel in original(*args)]

        monkeypatch.setattr(candidate, "wall_panels", larger)
    actual = candidate.extract_monitors(model, baseline, registration, **kwargs)
    assert actual["max_disp_mm"] >= 1000
    maximum = max(actual["profiles"], key=lambda row: np.linalg.norm(row["u_m"][:2]))
    assert (maximum["element_ids"] == [101]) == (owner == "crown")


def test_same_owner_discontinuity_remains_an_error(monkeypatch):
    model, baseline, registration, kwargs = crown_fixture()
    original = candidate.crown_intervals

    def duplicated(*args):
        intervals = original(*args)
        lower, upper, identifier, values = intervals[0]
        intervals.append((lower, upper, identifier, [values[0] + [0.01, 0, 0], values[1]]))
        return intervals

    monkeypatch.setattr(candidate, "crown_intervals", duplicated)
    with pytest.raises(ValueError, match="Discontinuous"):
        candidate.extract_monitors(model, baseline, registration, **kwargs)
