"""Bounded banded factorization for separated reference BVPs."""

from contextlib import contextmanager
import time

import numpy as np
from scipy.integrate import _bvp
from scipy.linalg.lapack import dgbtrf, dgbtrs


class SeparatedBandedLU:
    """Factor a collocation Jacobian with separated endpoint conditions."""

    def __init__(self, matrix, state_dimension=14, left_conditions=7):
        started = time.monotonic()
        dimension = matrix.shape[0]
        if (
            matrix.shape != (dimension, dimension)
            or dimension % state_dimension
            or not 0 < left_conditions < state_dimension
        ):
            raise ValueError("Expected a square separated collocation Jacobian")
        self.left_conditions = left_conditions
        self.state_dimension = state_dimension
        self.lower = left_conditions + state_dimension - 1
        self.upper = 2 * state_dimension - left_conditions - 1
        self.dimension = dimension
        band = np.zeros((2 * self.lower + self.upper + 1, dimension), order="F")
        for start in range(0, dimension, 4096):
            stop = min(start + 4096, dimension)
            first, last = matrix.indptr[start], matrix.indptr[stop]
            columns = np.repeat(np.arange(start, stop), np.diff(matrix.indptr[start : stop + 1]))
            original_rows = matrix.indices[first:last]
            data = matrix.data[first:last]
            nonzero = data != 0
            columns = columns[nonzero]
            original_rows = original_rows[nonzero]
            rows = np.where(
                original_rows < dimension - state_dimension,
                original_rows + left_conditions,
                np.where(
                    original_rows < dimension - state_dimension + left_conditions,
                    original_rows - dimension + state_dimension,
                    original_rows,
                ),
            )
            offset = rows - columns
            if np.any(offset > self.lower) or np.any(offset < -self.upper):
                raise ValueError("Boundary conditions are not separated as specified")
            band[self.lower + self.upper + offset, columns] = data[nonzero]
        self.lu, self.pivots, info = dgbtrf(band, self.lower, self.upper, overwrite_ab=True)
        if info:
            raise RuntimeError(f"Banded factorization failed with LAPACK info={info}")
        self.statistics = {
            "dimension": dimension,
            "nodes": dimension // state_dimension,
            "input_nnz": int(matrix.nnz),
            "factor_bytes": self.lu.nbytes + self.pivots.nbytes,
            "seconds": time.monotonic() - started,
        }

    @classmethod
    def from_band(cls, band, state_dimension=14, left_conditions=7):
        """Factor an already row-permuted LAPACK band, taking ownership of it."""
        started = time.monotonic()
        result = cls.__new__(cls)
        result.left_conditions = left_conditions
        result.state_dimension = state_dimension
        result.lower = left_conditions + state_dimension - 1
        result.upper = 2 * state_dimension - left_conditions - 1
        result.dimension = band.shape[1]
        if (
            band.shape[0] != 2 * result.lower + result.upper + 1
            or result.dimension % state_dimension
            or not 0 < left_conditions < state_dimension
        ):
            raise ValueError("Invalid separated LAPACK band dimensions")
        result.lu, result.pivots, info = dgbtrf(band, result.lower, result.upper, overwrite_ab=True)
        if info:
            raise RuntimeError(f"Banded factorization failed with LAPACK info={info}")
        result.statistics = {
            "dimension": result.dimension,
            "nodes": result.dimension // state_dimension,
            "factor_bytes": result.lu.nbytes + result.pivots.nbytes,
            "seconds": time.monotonic() - started,
            "assembly": "direct_banded",
        }
        return result

    def solve(self, right):
        boundary_start = self.dimension - self.state_dimension
        boundary_split = boundary_start + self.left_conditions
        ordered = np.concatenate(
            (right[boundary_start:boundary_split], right[:boundary_start], right[boundary_split:])
        )
        solution, info = dgbtrs(
            self.lu, self.lower, self.upper, ordered[:, None], self.pivots, overwrite_b=True
        )
        if info:
            raise RuntimeError(f"Banded solve failed with LAPACK info={info}")
        return solution[:, 0]


@contextmanager
def banded_collocation(callback=None, state_dimension=14, left_conditions=7):
    """Use a banded linear solver only within this private reference process."""
    original = _bvp.splu

    def factor(matrix):
        result = SeparatedBandedLU(matrix, state_dimension, left_conditions)
        if callback is not None:
            callback(result.statistics)
        return result

    _bvp.splu = factor
    try:
        yield
    finally:
        _bvp.splu = original
