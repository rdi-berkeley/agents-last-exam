import numpy as np
import pytest

from tasks.engineering.inner_support_elevation_optimization.mesh_evidence import describe_mesh


def measured(scale=1, shift=0):
    nodes = {1: [0, 0, 0], 2: [1, 0, 0], 3: [0, 1, 0], 4: [0, 0, -1]}
    nodes = {identifier: np.asarray(point) * scale + shift for identifier, point in nodes.items()}
    geometry = {
        "outer_polygon": (np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]]) * scale + shift).tolist(),
        "layers": [{"id": "soil", "top": shift, "bottom": shift - scale}],
    }
    faces = np.asarray([[[0, 0, 0], [1, 0, 0], [1, 0, -0.5], [0, 0, -0.5]]]) * scale + shift
    return describe_mesh(nodes, {1: ("tetra4", 1, [1, 2, 3, 4])}, {"outer": faces}, geometry)


def test_native_measurements_preserve_units_and_translation_without_awarding_acceptance():
    baseline = measured()
    translated = measured(shift=100)
    assert baseline["tetrahedron_volume_m3"] == translated["tetrahedron_volume_m3"]
    assert baseline["tetrahedron_volume_m3"]["median"] == pytest.approx(1 / 6)
    walls = baseline["wall_panels_and_boundary_clearance"]["outer"]
    assert walls["toe_to_fixed_bottom_m"] == pytest.approx(0.5)
    assert baseline["wall_distance_bins"]["0_to_2_m"]["diameter_m"]["max"] == pytest.approx(2**0.5)
    assert measured(scale=2)["tetrahedron_volume_m3"]["median"] == pytest.approx(8 / 6)
    assert not {"passed", "adequate", "score"} & baseline.keys()
    assert "cannot prove boundary independence" in baseline["limits"]
