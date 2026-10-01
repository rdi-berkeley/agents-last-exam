import numpy as np
import pytest
from scipy.integrate import solve_bvp
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu

from tasks.engineering.aerospace_low_thrust_trajectory.scripts.reference_bvp import (
    SeparatedBandedLU,
    banded_collocation,
)


@pytest.mark.parametrize("nodes", [2, 13, 200])
@pytest.mark.parametrize("state_dimension,left_conditions", [(14, 7), (15, 7)])
def test_banded_factorization_matches_sparse_solution(nodes, state_dimension, left_conditions):
    generator = np.random.default_rng(62109)
    dimension = nodes * state_dimension
    matrix = np.zeros((dimension, dimension))
    for index in range(nodes - 1):
        matrix[
            index * state_dimension : (index + 1) * state_dimension,
            index * state_dimension : (index + 1) * state_dimension,
        ] = (
            -np.eye(state_dimension)
            + generator.normal(size=(state_dimension, state_dimension)) * 0.003
        )
        matrix[
            index * state_dimension : (index + 1) * state_dimension,
            (index + 1) * state_dimension : (index + 2) * state_dimension,
        ] = (
            np.eye(state_dimension)
            + generator.normal(size=(state_dimension, state_dimension)) * 0.003
        )
    right_conditions = state_dimension - left_conditions
    matrix[-state_dimension:-right_conditions, :left_conditions] = np.eye(left_conditions)
    matrix[-right_conditions:, -right_conditions:] = np.eye(right_conditions)
    sparse = csc_array(matrix)
    expected = generator.normal(size=dimension)
    right = sparse @ expected
    banded = SeparatedBandedLU(sparse, state_dimension, left_conditions)
    actual = banded.solve(right)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(actual, splu(sparse).solve(right), atol=1e-11, rtol=1e-11)
    assert banded.statistics["factor_bytes"] == dimension * (
        (2 * banded.lower + banded.upper + 1) * 8 + 4
    )


def test_banded_solver_rejects_nonseparated_boundary():
    matrix = np.eye(140)
    matrix[-14, -1] = 1
    with pytest.raises(ValueError, match="not separated"):
        SeparatedBandedLU(csc_array(matrix))


def test_banded_collocation_solves_canonical_boundary_problem():
    times = np.linspace(0, 1, 31)
    guess = np.zeros((14, len(times)))

    def derivative(times, state):
        return np.vstack((state[7:], -state[:7]))

    def boundary(left, right):
        return np.r_[left[:7], right[7:] - 1]

    statistics = []
    with banded_collocation(statistics.append):
        solution = solve_bvp(derivative, boundary, times, guess, tol=1e-9)
    assert solution.success
    np.testing.assert_allclose(
        solution.sol(times)[:7],
        np.broadcast_to(np.sin(times) / np.cos(1), (7, len(times))),
        atol=1e-9,
    )
    assert statistics
