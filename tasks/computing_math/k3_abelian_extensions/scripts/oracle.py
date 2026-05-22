#!/usr/bin/env python
"""Deterministic oracle for computing_math/k3_abelian_extensions."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from math import gcd, lcm
from typing import Any


def _factorint(value: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    candidate = 2
    remaining = value
    while candidate * candidate <= remaining:
        while remaining % candidate == 0:
            factors[candidate] = factors.get(candidate, 0) + 1
            remaining //= candidate
        candidate += 1 if candidate == 2 else 2
    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors


def _product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _divisors(value: int) -> list[int]:
    divisors = [1]
    for prime, exponent in _factorint(value).items():
        divisors = [
            divisor * (prime**power)
            for divisor in divisors
            for power in range(exponent + 1)
        ]
    return sorted(set(divisors))


def _partitions(total: int, min_part: int = 1):
    if total == 0:
        yield []
        return
    for first in range(min_part, total + 1):
        for rest in _partitions(total - first, first):
            yield [first] + rest


@lru_cache(maxsize=None)
def _enumerate_invariant_factors(order: int) -> tuple[tuple[int, ...], ...]:
    prime_factors = list(_factorint(order).items())
    per_prime_partitions = [list(_partitions(exponent)) for _, exponent in prime_factors]
    seen: set[tuple[int, ...]] = set()
    outputs: list[tuple[int, ...]] = []

    for choice in product(*per_prime_partitions):
        rank = max(len(partition) for partition in choice) if choice else 0
        factors = [1] * rank
        for (prime, _), partition in zip(prime_factors, choice):
            padded = [0] * (rank - len(partition)) + partition
            for index, exponent in enumerate(padded):
                factors[index] *= prime**exponent
        canonical = tuple(factors)
        if canonical not in seen:
            seen.add(canonical)
            outputs.append(canonical)

    return tuple(sorted(outputs))


def _element_order(element: tuple[int, ...], invariant_factors: tuple[int, ...]) -> int:
    order = 1
    for coordinate, modulus in zip(element, invariant_factors):
        if coordinate:
            order = lcm(order, modulus // gcd(modulus, coordinate))
    return order


def _generated_subgroup(
    element: tuple[int, ...],
    invariant_factors: tuple[int, ...],
) -> frozenset[tuple[int, ...]]:
    order = _element_order(element, invariant_factors)
    return frozenset(
        tuple((multiplier * coordinate) % modulus for coordinate, modulus in zip(element, invariant_factors))
        for multiplier in range(order)
    )


def _det_bareiss(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if size == 0:
        return 1

    work = [row[:] for row in matrix]
    sign = 1
    prev_pivot = 1

    for pivot_index in range(size - 1):
        if work[pivot_index][pivot_index] == 0:
            for row_index in range(pivot_index + 1, size):
                if work[row_index][pivot_index] != 0:
                    work[pivot_index], work[row_index] = work[row_index], work[pivot_index]
                    sign *= -1
                    break
            else:
                return 0

        pivot = work[pivot_index][pivot_index]
        for row_index in range(pivot_index + 1, size):
            for col_index in range(pivot_index + 1, size):
                numerator = (
                    work[row_index][col_index] * pivot
                    - work[row_index][pivot_index] * work[pivot_index][col_index]
                )
                work[row_index][col_index] = numerator // prev_pivot

        prev_pivot = pivot
        for row_index in range(pivot_index + 1, size):
            work[row_index][pivot_index] = 0
        for col_index in range(pivot_index + 1, size):
            work[pivot_index][col_index] = 0

    return sign * work[-1][-1]


def _quotient_invariant_factors(
    invariant_factors: tuple[int, ...],
    generator: tuple[int, ...],
) -> tuple[int, ...]:
    rank = len(invariant_factors)
    relations = [[0] * rank for _ in range(rank + 1)]
    for index, modulus in enumerate(invariant_factors):
        relations[index][index] = modulus
    relations[-1] = list(generator)

    minors_gcd = [1]
    for minor_size in range(1, rank + 1):
        current_gcd = 0
        for row_indices in combinations(range(rank + 1), minor_size):
            for col_indices in combinations(range(rank), minor_size):
                minor = [[relations[row][col] for col in col_indices] for row in row_indices]
                current_gcd = gcd(current_gcd, abs(_det_bareiss(minor)))
        minors_gcd.append(current_gcd)

    diagonal: list[int] = []
    previous = 1
    for minor_size in range(1, rank + 1):
        diagonal_value = minors_gcd[minor_size] // previous
        previous = minors_gcd[minor_size]
        if diagonal_value > 1:
            diagonal.append(diagonal_value)
    return tuple(diagonal)


def _p_adic_valuation(value: int, prime: int) -> int:
    exponent = 0
    remaining = value
    while remaining and remaining % prime == 0:
        exponent += 1
        remaining //= prime
    return exponent


def _to_primary_exponents(invariant_factors: list[int]) -> dict[int, list[int]]:
    primes = sorted({prime for value in invariant_factors for prime in _factorint(value)})
    primary: dict[int, list[int]] = {}
    for prime in primes:
        exponents = [_p_adic_valuation(value, prime) for value in invariant_factors]
        primary[prime] = [exponent for exponent in exponents if exponent]
    return primary


def _direct_product_invariant_factors(left: list[int], right: list[int]) -> list[int]:
    primary: dict[int, list[int]] = {}
    for factors in (left, right):
        for prime, exponents in _to_primary_exponents(factors).items():
            primary.setdefault(prime, []).extend(exponents)

    rank = max((len(exponents) for exponents in primary.values()), default=0)
    invariant_factors = [1] * rank
    for prime, exponents in primary.items():
        padded = [0] * (rank - len(exponents)) + sorted(exponents)
        for index, exponent in enumerate(padded):
            invariant_factors[index] *= prime**exponent
    return [value for value in invariant_factors if value > 1]


@lru_cache(maxsize=None)
def _allowed_rank_leq_two_groups(max_order: int) -> tuple[tuple[int, ...], ...]:
    candidates: set[tuple[int, ...]] = {tuple()}
    divisor_list = _divisors(max_order)
    for value in divisor_list:
        candidates.add((value,))
    for left in divisor_list:
        for right in divisor_list:
            if left <= right and right % left == 0 and left * right <= max_order:
                candidates.add((left, right))
    return tuple(sorted(candidates))


def is_product_type(invariant_factors: list[int]) -> bool:
    order = _product(invariant_factors)

    g1_candidates = [[value] for value in _divisors(order)]
    g1_candidates.append([2, 2])

    g2_candidates = [list(candidate) for candidate in _allowed_rank_leq_two_groups(order)]
    n = 1
    while 2 * n <= order:
        g2_candidates.append([2, 2, 2 * n])
        n += 1
    for fixed in ([2, 4, 4], [3, 3, 3], [2, 2, 2, 2]):
        if _product(list(fixed)) <= order:
            g2_candidates.append(list(fixed))

    target = list(invariant_factors)
    for g1 in g1_candidates:
        for g2 in g2_candidates:
            if _direct_product_invariant_factors(g1, g2) == target:
                return True
    return False


@lru_cache(maxsize=None)
def _solve_cached(
    h_invariant_factors: tuple[int, ...],
    m_start: int,
    m_end: int,
) -> dict[str, Any]:
    h_order = _product(list(h_invariant_factors))
    extensions: list[dict[str, Any]] = []

    for m_value in range(m_start, m_end + 1):
        target_order = h_order * m_value
        for candidate_group in _enumerate_invariant_factors(target_order):
            seen_subgroups: set[frozenset[tuple[int, ...]]] = set()
            has_valid_extension = False

            for element in product(*[range(modulus) for modulus in candidate_group]):
                if _element_order(element, candidate_group) != m_value:
                    continue
                subgroup = _generated_subgroup(element, candidate_group)
                if subgroup in seen_subgroups:
                    continue
                seen_subgroups.add(subgroup)
                if _quotient_invariant_factors(candidate_group, element) == h_invariant_factors:
                    has_valid_extension = True
                    break

            if not has_valid_extension:
                continue

            group_factors = list(candidate_group)
            extensions.append(
                {
                    "m": m_value,
                    "G_invariant_factors": group_factors,
                    "G_order": target_order,
                    "product_type": is_product_type(group_factors),
                }
            )

    extensions = sorted(
        extensions,
        key=lambda entry: (int(entry["m"]), tuple(int(value) for value in entry["G_invariant_factors"])),
    )
    non_product = [entry for entry in extensions if not entry["product_type"]]
    return {
        "total_extensions": len(extensions),
        "extensions": extensions,
        "non_product_type_count": len(non_product),
        "non_product_type": non_product,
    }


def build_expected_output(
    *,
    h_invariant_factors: list[int],
    m_search_range: list[int],
) -> dict[str, Any]:
    return _solve_cached(
        tuple(int(value) for value in h_invariant_factors),
        int(m_search_range[0]),
        int(m_search_range[1]),
    )


def build_negative_fixture(expected_output: dict[str, Any]) -> dict[str, Any]:
    negative = {
        "total_extensions": int(expected_output["total_extensions"]),
        "extensions": [dict(entry) for entry in expected_output["extensions"]],
        "non_product_type_count": int(expected_output["non_product_type_count"]),
        "non_product_type": [dict(entry) for entry in expected_output["non_product_type"]],
    }
    if negative["extensions"]:
        negative["extensions"][0]["product_type"] = not bool(negative["extensions"][0]["product_type"])
    if negative["non_product_type"]:
        negative["non_product_type"] = negative["non_product_type"][:-1]
    else:
        negative["non_product_type"] = [dict(negative["extensions"][0])]
    negative["non_product_type_count"] = len(negative["non_product_type"])
    return negative
