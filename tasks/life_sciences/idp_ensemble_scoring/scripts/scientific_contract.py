"""Full-pool IDP raw-score aggregation, independent of reference values."""

import math

METHODS = tuple("Model{}".format(index) for index in range(1, 6))
OBSERVABLES = ("CS", "JC", "NOE", "PRE")
COLUMNS = ("Method", "Total", "CS", "JC", "NOE/PRE")
DEGENERATE_VALUE = 0.5


def normalize_columns(matrix):
    """Min-max each protein; a zero-span column contributes 0.5 to every method."""
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix) or not matrix[0]:
        raise ValueError("Expected a nonempty rectangular matrix")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("Raw scores must be finite")
    normalized = [[0.0] * len(matrix[0]) for row in matrix]
    for column in range(len(matrix[0])):
        minimum = min(row[column] for row in matrix)
        maximum = max(row[column] for row in matrix)
        span = maximum - minimum
        for index, row in enumerate(matrix):
            normalized[index][column] = (row[column] - minimum) / span if span else DEGENERATE_VALUE
    return [math.fsum(row) / len(row) for row in normalized]


def aggregate_scores(records, protein_sets):
    """Require every declared model/protein/observable and aggregate raw log scores."""
    sets = {key: set(protein_sets[key]) for key in OBSERVABLES}
    proteins = sorted(set().union(*sets.values()))
    if not proteins or any(not values for values in sets.values()):
        raise ValueError("Each observable must have a nonempty protein set")
    expected = {
        (method, protein, observable)
        for method in METHODS
        for observable in OBSERVABLES
        for protein in sets[observable]
    }
    raw = {}
    for record in records:
        key = (record["Method"], record["protein"], record["observable"])
        if key not in expected or key in raw:
            raise ValueError("Unexpected or duplicate raw score: {}".format(key))
        value = float(record["score"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite raw score: {}".format(key))
        raw[key] = value
    if set(raw) != expected:
        raise ValueError("Missing raw scores: {}".format(sorted(expected - set(raw))))
    groups = {
        "CS": (sorted(sets["CS"]), ("CS",)),
        "JC": (sorted(sets["JC"]), ("JC",)),
        "NOE/PRE": (sorted(sets["NOE"] | sets["PRE"]), ("NOE", "PRE")),
        "Total": (proteins, OBSERVABLES),
    }
    values = {}
    for label, (selected, observables) in groups.items():
        matrix = [
            [
                math.fsum(
                    raw[(method, protein, observable)]
                    for observable in observables
                    if protein in sets[observable]
                )
                for protein in selected
            ]
            for method in METHODS
        ]
        values[label] = normalize_columns(matrix)
    result = [
        dict(Method=method, **{label: values[label][index] for label in COLUMNS[1:]})
        for index, method in enumerate(METHODS)
    ]
    return sorted(result, key=lambda row: (-row["Total"], row["Method"]))
