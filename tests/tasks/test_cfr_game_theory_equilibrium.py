from __future__ import annotations

import numpy as np
import pytest

from tasks.computing_math.cfr_game_theory_equilibrium.scripts.score_outputs import (
    LEDUC4_DECK,
    LeducBestResponse,
    LeducState,
)


def _expected_value(state: LeducState, strategy: dict[str, np.ndarray]) -> float:
    if state.is_terminal():
        return float(state.terminal_utility_p0())
    actions = state.get_actions()
    probabilities = strategy.get(state.info_key())
    if probabilities is None or len(probabilities) != len(actions):
        probabilities = np.ones(len(actions)) / len(actions)
    return sum(
        float(probabilities[index]) * _expected_value(state.apply(action), strategy)
        for index, action in enumerate(actions)
    )


def _game_value(strategy: dict[str, np.ndarray]) -> float:
    total = 0.0
    deal_probability = 1.0 / (8 * 7 * 6)
    for index0, card0 in enumerate(LEDUC4_DECK):
        for index1, card1 in enumerate(LEDUC4_DECK):
            if index0 == index1:
                continue
            for community in [card for card in LEDUC4_DECK if card != card0 and card != card1]:
                total += deal_probability * _expected_value(
                    LeducState([card0, card1], community), strategy
                )
    return total


def _ten_iteration_cfr_strategy() -> dict[str, np.ndarray]:
    regrets: dict[str, np.ndarray] = {}
    strategy_sums: dict[str, np.ndarray] = {}
    deal_probability = 1.0 / (8 * 7 * 6)

    def current_strategy(key: str, action_count: int) -> np.ndarray:
        values = regrets.setdefault(key, np.zeros(action_count))
        positive = np.maximum(values, 0.0)
        return positive / positive.sum() if positive.sum() > 0 else np.ones(action_count) / action_count

    def traverse(state: LeducState, reach0: float, reach1: float) -> float:
        if state.is_terminal():
            return float(state.terminal_utility_p0())
        actions = state.get_actions()
        key = state.info_key()
        strategy = current_strategy(key, len(actions))
        player = state.current_player
        strategy_sums.setdefault(key, np.zeros(len(actions)))[:] += (
            deal_probability * (reach0 if player == 0 else reach1) * strategy
        )
        action_values = np.empty(len(actions))
        for index, action in enumerate(actions):
            if player == 0:
                action_values[index] = traverse(state.apply(action), reach0 * strategy[index], reach1)
            else:
                action_values[index] = traverse(state.apply(action), reach0, reach1 * strategy[index])
        node_value = float(np.dot(strategy, action_values))
        if player == 0:
            regrets[key] += deal_probability * reach1 * (action_values - node_value)
        else:
            regrets[key] += deal_probability * reach0 * (node_value - action_values)
        return node_value

    for _iteration in range(10):
        for index0, card0 in enumerate(LEDUC4_DECK):
            for index1, card1 in enumerate(LEDUC4_DECK):
                if index0 == index1:
                    continue
                for community in [
                    card for card in LEDUC4_DECK if card != card0 and card != card1
                ]:
                    traverse(LeducState([card0, card1], community), 1.0, 1.0)

    return {
        key: totals / totals.sum()
        if totals.sum() > 0
        else np.ones(len(totals)) / len(totals)
        for key, totals in strategy_sums.items()
    }


def test_best_response_bounds_strategy_value_for_uniform_strategy():
    strategy: dict[str, np.ndarray] = {}
    value = _game_value(strategy)
    best_response = LeducBestResponse(strategy)
    br0 = best_response._compute_br(0)
    br1_p0 = best_response._compute_br(1)

    assert br0 == pytest.approx(2.95535714285714)
    assert br1_p0 == pytest.approx(-3.82862103174603)
    assert br0 >= value >= br1_p0


def test_cfr_regression_uses_information_set_respecting_best_response():
    strategy = _ten_iteration_cfr_strategy()

    assert len(strategy) == 504
    assert LeducBestResponse(strategy).compute_exploitability() == pytest.approx(
        0.29601689341506665
    )
