"""Construct decoder inputs without discarding joint detector/logical effects."""

from __future__ import annotations

import numpy as np
import stim
from scipy.sparse import csc_matrix


def full_dem_matrices(
    model: stim.DetectorErrorModel,
) -> tuple[csc_matrix, csc_matrix, np.ndarray]:
    detector_rows: list[int] = []
    detector_columns: list[int] = []
    observable_rows: list[int] = []
    observable_columns: list[int] = []
    priors: list[float] = []
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        column = len(priors)
        priors.append(instruction.args_copy()[0])
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update((target.val,))
            elif target.is_logical_observable_id():
                observables.symmetric_difference_update((target.val,))
        detector_rows.extend(sorted(detectors))
        detector_columns.extend([column] * len(detectors))
        observable_rows.extend(sorted(observables))
        observable_columns.extend([column] * len(observables))
    check = csc_matrix(
        (np.ones(len(detector_rows), dtype=np.uint8), (detector_rows, detector_columns)),
        shape=(model.num_detectors, len(priors)),
    )
    logical = csc_matrix(
        (np.ones(len(observable_rows), dtype=np.uint8), (observable_rows, observable_columns)),
        shape=(model.num_observables, len(priors)),
    )
    return check, logical, np.asarray(priors, dtype=np.float64)
