from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import stim

from tasks.computing_math.clustered_cyclic_code_circuit_level_simulation.scripts.decoder_contract import (
    full_dem_matrices,
)


def test_same_syndrome_different_logicals_preserves_joint_distribution():
    model = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 L1")
    check, logical, priors = full_dem_matrices(model)
    observed = {}
    for faults in product((0, 1), repeat=2):
        vector = np.asarray(faults, dtype=np.uint8)
        outcome = tuple(np.concatenate((check @ vector % 2, logical @ vector % 2)))
        probability = float(np.prod(np.where(vector, priors, 1 - priors)))
        observed[outcome] = observed.get(outcome, 0) + probability
    assert observed == pytest.approx(
        {(0, 0, 0): 0.72, (1, 1, 0): 0.08, (1, 0, 1): 0.18, (0, 1, 1): 0.02}
    )


def test_identical_joint_effects_remain_separate_bp_variables():
    model = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 L0")
    check, logical, priors = full_dem_matrices(model)
    np.testing.assert_array_equal(check.toarray(), [[1, 1]])
    np.testing.assert_array_equal(logical.toarray(), [[1, 1]])
    np.testing.assert_array_equal(priors, [0.1, 0.2])


def test_flattening_resolves_repeats_and_detector_shifts_in_order():
    model = stim.DetectorErrorModel(
        "repeat 2 {\nerror(0.125) D0 L0\nshift_detectors 1\n}\n"
        "error(0.25) L1\ndetector D2\nlogical_observable L2"
    )
    check, logical, priors = full_dem_matrices(model)
    assert check.shape == (5, 3)
    np.testing.assert_array_equal(check.toarray()[:2], [[1, 0, 0], [0, 1, 0]])
    assert not check.toarray()[2:].any()
    np.testing.assert_array_equal(logical.toarray(), [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(priors, [0.125, 0.125, 0.25])


def test_separators_do_not_turn_correlated_components_into_independent_faults():
    model = stim.DetectorErrorModel("error(0.1) D0 L0 ^ D1 L1")
    check, logical, priors = full_dem_matrices(model)
    np.testing.assert_array_equal(check.toarray(), [[1], [1]])
    np.testing.assert_array_equal(logical.toarray(), [[1], [1]])
    np.testing.assert_array_equal(priors, [0.1])
