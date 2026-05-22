"""Local evaluator for computing_math/cfr_game_theory_equilibrium."""

import json
import math
from dataclasses import dataclass

import numpy as np


PAYOFF_MATRIX = np.array(
    [
        [3.0, -2.0, 5.0, 1.0, -4.0],
        [-1.0, 4.0, -3.0, 2.0, 6.0],
        [7.0, -5.0, 0.0, -1.0, 3.0],
        [-6.0, 8.0, 2.0, 4.0, -2.0],
        [1.0, 3.0, -7.0, 5.0, 0.0],
    ],
    dtype=float,
)

KUHN_CARDS = ["J", "Q", "K"]
KUHN_RANK = {"J": 0, "Q": 1, "K": 2}
KUHN_TERMINALS = frozenset(["pp", "pbp", "pbb", "bp", "bb"])

LEDUC4_RANKS = [1, 2, 3, 4]
LEDUC4_SUITS = [0, 1]
LEDUC4_DECK = [(r, s) for r in LEDUC4_RANKS for s in LEDUC4_SUITS]
LEDUC4_BET = {1: 3, 2: 6}
LEDUC4_MAX_RAISES = 2

TIER1_TOL = 1e-6
TIER2_EXPLOITABILITY_MAX = 0.01
TIER3_EXPLOITABILITY_MAX = 0.05


@dataclass
class ScoreResult:
    score: float
    tier1_passed: bool
    tier2_passed: bool
    tier3_passed: bool
    details: dict


def _is_prob_vector(values, expected_len):
    if not isinstance(values, list) or len(values) != expected_len:
        return False
    try:
        arr = np.array(values, dtype=float)
    except Exception:
        return False
    return bool(np.all(arr >= -1e-9) and abs(float(arr.sum()) - 1.0) <= 1e-6)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _is_index_list(values, *, size):
    if not isinstance(values, list):
        return False
    ints = []
    for value in values:
        try:
            intval = int(value)
        except Exception:
            return False
        if value != intval:
            return False
        if intval < 0 or intval >= size:
            return False
        ints.append(intval)
    return len(set(ints)) == len(ints)


def _support_matches_probs(probs, support, *, tol=1e-8):
    expected = {idx for idx, prob in enumerate(probs) if float(prob) > tol}
    return expected == set(support)


def score_submission_texts(agent_text: str, reference_text: str | None = None) -> ScoreResult:
    try:
        agent = json.loads(agent_text)
    except Exception as exc:
        return ScoreResult(0.0, False, False, False, {"parse_error": str(exc)})

    tier1 = _score_tier1(agent.get("tier1"))
    tier2 = _score_tier2(agent.get("tier2"))
    tier3 = _score_tier3(agent.get("tier3"))

    if tier1["passed"] and tier2["passed"] and tier3["passed"]:
        score = 1.0
    elif tier1["passed"] and tier2["passed"]:
        score = 0.67
    elif tier1["passed"]:
        score = 0.33
    else:
        score = 0.0

    return ScoreResult(
        score=score,
        tier1_passed=tier1["passed"],
        tier2_passed=tier2["passed"],
        tier3_passed=tier3["passed"],
        details={"tier1": tier1, "tier2": tier2, "tier3": tier3},
    )


def _score_tier1(agent_tier):
    result = {"passed": False}
    if not isinstance(agent_tier, dict):
        result["error"] = "missing tier1 object"
        return result

    row = agent_tier.get("row_strategy")
    col = agent_tier.get("col_strategy")
    row_support = agent_tier.get("row_support")
    col_support = agent_tier.get("col_support")
    value = _safe_float(agent_tier.get("game_value"))
    if (
        not _is_prob_vector(row, 5)
        or not _is_prob_vector(col, 5)
        or not _is_index_list(row_support, size=5)
        or not _is_index_list(col_support, size=5)
        or value is None
    ):
        result["error"] = "invalid tier1 schema"
        return result

    row_arr = np.array(row, dtype=float)
    col_arr = np.array(col, dtype=float)
    realized = float(row_arr @ PAYOFF_MATRIX @ col_arr)
    max_dev = float(np.max(PAYOFF_MATRIX @ col_arr))
    min_dev = float(np.min(row_arr @ PAYOFF_MATRIX))

    conditions = {
        "value_matches_realized": abs(realized - value) <= TIER1_TOL,
        "no_row_deviation": max_dev <= value + TIER1_TOL,
        "no_col_deviation": min_dev >= value - TIER1_TOL,
        "row_support_matches_strategy": _support_matches_probs(row_arr, row_support),
        "col_support_matches_strategy": _support_matches_probs(col_arr, col_support),
    }
    result.update(
        {
            "realized_value": realized,
            "max_row_payoff": max_dev,
            "min_col_payoff": min_dev,
            "conditions": conditions,
            "passed": all(conditions.values()),
        }
    )
    return result


def kuhn_terminal_util_p0(cards, history):
    p0_wins = KUHN_RANK[cards[0]] > KUHN_RANK[cards[1]]
    if history == "pp":
        return 1 if p0_wins else -1
    if history == "bp":
        return 1
    if history == "pbp":
        return -1
    if history in ("bb", "pbb"):
        return 2 if p0_wins else -2
    raise ValueError(history)


def _kuhn_subtree_value(cards, history, avg_strat, br_player, br_policy):
    if history in KUHN_TERMINALS:
        return kuhn_terminal_util_p0(cards, history)

    player = len(history) % 2
    info_set = cards[player] + "|" + history
    if player == br_player:
        action = br_policy.get(info_set)
        if action is not None:
            return _kuhn_subtree_value(cards, history + "pb"[action], avg_strat, br_player, br_policy)
        vals = [_kuhn_subtree_value(cards, history + "pb"[a], avg_strat, br_player, br_policy) for a in range(2)]
        return max(vals) if br_player == 0 else min(vals)

    strat = avg_strat.get(info_set, np.array([0.5, 0.5]))
    return sum(
        float(strat[a]) * _kuhn_subtree_value(cards, history + "pb"[a], avg_strat, br_player, br_policy)
        for a in range(2)
    )


def _kuhn_reach_prob(cards, target_history, avg_strat, br_player, br_policy):
    prob = 1.0
    for i, ch in enumerate(target_history):
        player = i % 2
        info_set = cards[player] + "|" + target_history[:i]
        action = 0 if ch == "p" else 1
        if player == br_player:
            if info_set in br_policy and br_policy[info_set] != action:
                return 0.0
        else:
            strat = avg_strat.get(info_set, np.array([0.5, 0.5]))
            prob *= float(strat[action])
    return prob


def _kuhn_br(avg_strat, br_player):
    all_info_sets = set()
    for card in KUHN_CARDS:
        for history in ["", "p", "b", "pb", "pp", "bp", "bb", "pbp", "pbb"]:
            if history in KUHN_TERMINALS:
                continue
            player = len(history) % 2
            if player == br_player:
                all_info_sets.add(card + "|" + history)

    br_policy = {}
    for info_set in sorted(all_info_sets, key=lambda item: len(item.split("|")[1]), reverse=True):
        action_values = np.zeros(2)
        card_in_set, history = info_set.split("|", 1)
        for c0 in KUHN_CARDS:
            for c1 in KUHN_CARDS:
                if c0 == c1:
                    continue
                cards = [c0, c1]
                if cards[br_player] != card_in_set:
                    continue
                deal_prob = 1.0 / 6.0
                reach = _kuhn_reach_prob(cards, history, avg_strat, br_player, br_policy)
                if reach < 1e-15:
                    continue
                for action in range(2):
                    next_history = history + "pb"[action]
                    value = _kuhn_subtree_value(cards, next_history, avg_strat, br_player, br_policy)
                    action_values[action] += deal_prob * reach * value
        br_policy[info_set] = int(np.argmax(action_values) if br_player == 0 else np.argmin(action_values))

    total = 0.0
    for c0 in KUHN_CARDS:
        for c1 in KUHN_CARDS:
            if c0 != c1:
                total += _kuhn_subtree_value([c0, c1], "", avg_strat, br_player, br_policy)
    return total / 6.0


def kuhn_exploitability(avg_strat):
    br0 = _kuhn_br(avg_strat, 0)
    br1_p0 = _kuhn_br(avg_strat, 1)
    return (br0 - br1_p0) / 2.0


def _kuhn_ev_rec(cards, history, strategy):
    if history in KUHN_TERMINALS:
        return kuhn_terminal_util_p0(cards, history)
    player = len(history) % 2
    info_set = cards[player] + "|" + history
    strat = strategy.get(info_set, np.array([0.5, 0.5]))
    return sum(float(strat[a]) * _kuhn_ev_rec(cards, history + "pb"[a], strategy) for a in range(2))


def kuhn_ev(strategy):
    total = 0.0
    for c0 in KUHN_CARDS:
        for c1 in KUHN_CARDS:
            if c0 != c1:
                total += _kuhn_ev_rec([c0, c1], "", strategy)
    return total / 6.0


def _score_tier2(agent_tier):
    result = {"passed": False}
    if not isinstance(agent_tier, dict):
        result["error"] = "missing tier2 object"
        return result

    avg = agent_tier.get("average_strategy")
    n_info = agent_tier.get("n_information_sets")
    n_iterations = agent_tier.get("n_iterations")
    reported_exploitability = _safe_float(agent_tier.get("final_exploitability"))
    reported_game_value = _safe_float(agent_tier.get("game_value_estimate"))
    if (
        not isinstance(avg, dict)
        or n_info != 12
        or not isinstance(n_iterations, int)
        or n_iterations < 1
        or reported_exploitability is None
        or reported_game_value is None
    ):
        result["error"] = "invalid tier2 schema"
        return result

    strategy = {}
    for key, value in avg.items():
        if not _is_prob_vector(value, 2):
            result["error"] = f"invalid tier2 strategy at {key}"
            return result
        strategy[key] = np.array(value, dtype=float)

    required = {
        "J|p": np.array([2.0 / 3.0, 1.0 / 3.0]),
        "Q|b": np.array([2.0 / 3.0, 1.0 / 3.0]),
        "K|p": np.array([0.0, 1.0]),
        "K|b": np.array([0.0, 1.0]),
    }
    fixed_family_ok = True
    family_errors = {}
    for key, expected in required.items():
        if key not in strategy:
            fixed_family_ok = False
            family_errors[key] = None
            continue
        err = float(np.max(np.abs(strategy[key] - expected)))
        family_errors[key] = err
        if err > 0.08:
            fixed_family_ok = False

    exploitability = float(kuhn_exploitability(strategy))
    ev = float(kuhn_ev(strategy))
    conditions = {
        "n_information_sets": n_info == 12,
        "n_iterations_reported": n_iterations >= 1,
        "exploitability": exploitability < TIER2_EXPLOITABILITY_MAX,
        "reported_exploitability_matches": abs(reported_exploitability - exploitability) <= 0.01,
        "reported_exploitability_threshold": reported_exploitability < TIER2_EXPLOITABILITY_MAX,
        "game_value_estimate": abs(ev - (-1.0 / 18.0)) <= 0.01,
        "reported_game_value_matches": abs(reported_game_value - ev) <= 0.01,
        "fixed_family_behaviors": fixed_family_ok,
    }
    result.update(
        {
            "n_iterations": n_iterations,
            "exploitability": exploitability,
            "reported_exploitability": reported_exploitability,
            "game_value_estimate": ev,
            "reported_game_value_estimate": reported_game_value,
            "family_errors": family_errors,
            "conditions": conditions,
            "passed": all(conditions.values()),
        }
    )
    return result


def _round_done(round_state):
    if len(round_state) < 2:
        return False
    if round_state == "cc":
        return True
    return round_state[-1] == "c" and "r" in round_state


class LeducState:
    __slots__ = ["cards", "community", "history", "pot", "round_num", "round_raises", "current_player"]

    def __init__(self, cards, community=None, history="", pot=None, round_num=1, round_raises=0, current_player=0):
        self.cards = cards
        self.community = community
        self.history = history
        self.pot = pot if pot is not None else [1, 1]
        self.round_num = round_num
        self.round_raises = round_raises
        self.current_player = current_player

    def is_terminal(self):
        if not self.history:
            return False
        if self.history.endswith("f"):
            return True
        rounds = self.history.split("/")
        return len(rounds) >= 2 and _round_done(rounds[-1])

    def get_actions(self):
        if self.is_terminal():
            return []
        current_round = self.history.split("/")[-1]
        if not current_round or (len(current_round) == 1 and current_round[-1] == "c"):
            return ["c", "r"]
        if current_round[-1] == "r":
            actions = ["f", "c"]
            if self.round_raises < LEDUC4_MAX_RAISES:
                actions.append("r")
            return actions
        return ["c", "r"]

    def apply(self, action):
        new_history = self.history + action
        new_pot = list(self.pot)
        new_raises = self.round_raises
        new_round = self.round_num
        next_player = 1 - self.current_player
        bet_size = LEDUC4_BET[min(self.round_num, 2)]
        me = self.current_player
        opp = 1 - me

        if action == "c":
            new_pot[me] += max(0, new_pot[opp] - new_pot[me])
        elif action == "r":
            new_pot[me] += max(0, new_pot[opp] - new_pot[me]) + bet_size
            new_raises += 1

        if action != "f":
            rounds = new_history.split("/")
            if _round_done(rounds[-1]) and new_round == 1:
                new_history += "/"
                new_round = 2
                new_raises = 0
                next_player = 0

        return LeducState(
            self.cards,
            self.community,
            new_history,
            new_pot,
            new_round,
            new_raises,
            next_player,
        )

    def terminal_utility_p0(self):
        if self.history.endswith("f"):
            folder = _who_acted_last(self.history)
            return -self.pot[0] if folder == 0 else self.pot[1]
        r0, r1, rc = self.cards[0][0], self.cards[1][0], self.community[0]
        p0_pair, p1_pair = (r0 == rc), (r1 == rc)
        if p0_pair and not p1_pair:
            return self.pot[1]
        if p1_pair and not p0_pair:
            return -self.pot[0]
        if r0 > r1:
            return self.pot[1]
        if r1 > r0:
            return -self.pot[0]
        return 0

    def info_key(self):
        my_rank = self.cards[self.current_player][0]
        if self.community is not None and "/" in self.history:
            first, second = self.history.split("/", 1)
            return f"{my_rank}:{first}/{self.community[0]}:{second}"
        return f"{my_rank}:{self.history}"


def _who_acted_last(history):
    player = 0
    last = 0
    for ch in history:
        if ch == "/":
            player = 0
            continue
        last = player
        player = 1 - player
    return last


def count_leduc_info_sets():
    info_sets = set()

    def traverse(state):
        if state.is_terminal():
            return
        if state.round_num == 2 and state.community is None:
            remaining = [card for card in LEDUC4_DECK if card != state.cards[0] and card != state.cards[1]]
            for community in remaining:
                traverse(
                    LeducState(
                        state.cards,
                        community,
                        state.history,
                        list(state.pot),
                        state.round_num,
                        state.round_raises,
                        state.current_player,
                    )
                )
            return
        actions = state.get_actions()
        if not actions:
            return
        info_sets.add(state.info_key())
        for action in actions:
            traverse(state.apply(action))

    for i, c0 in enumerate(LEDUC4_DECK):
        for j, c1 in enumerate(LEDUC4_DECK):
            if i != j:
                traverse(LeducState([c0, c1]))
    return info_sets


class LeducBestResponse:
    def __init__(self, avg_strategy):
        self.avg_strategy = avg_strategy

    def compute_exploitability(self):
        br0 = self._compute_br(0)
        br1_p0 = self._compute_br(1)
        return (br0 - br1_p0) / 2.0

    def _compute_br(self, br_player):
        cf_action_vals = {}
        for i, c0 in enumerate(LEDUC4_DECK):
            for j, c1 in enumerate(LEDUC4_DECK):
                if i == j:
                    continue
                remaining = [c for c in LEDUC4_DECK if c != c0 and c != c1]
                for community in remaining:
                    deal_prob = 1.0 / (8 * 7 * 6)
                    self._cf_br_traverse(
                        LeducState([c0, c1], community),
                        deal_prob,
                        1.0,
                        br_player,
                        cf_action_vals,
                    )
        br_policy = {}
        for key, values in cf_action_vals.items():
            br_policy[key] = int(np.argmax(values) if br_player == 0 else np.argmin(values))

        total = 0.0
        for i, c0 in enumerate(LEDUC4_DECK):
            for j, c1 in enumerate(LEDUC4_DECK):
                if i == j:
                    continue
                remaining = [c for c in LEDUC4_DECK if c != c0 and c != c1]
                for community in remaining:
                    deal_prob = 1.0 / (8 * 7 * 6)
                    total += self._eval_br_policy(LeducState([c0, c1], community), deal_prob, br_player, br_policy)
        return total

    def _cf_br_traverse(self, state, deal_prob, opp_reach, br_player, cf_vals):
        if state.is_terminal():
            return deal_prob * opp_reach * state.terminal_utility_p0()

        actions = state.get_actions()
        if not actions:
            return 0.0

        player = state.current_player
        key = state.info_key()
        n_actions = len(actions)

        if player == br_player:
            if key not in cf_vals:
                cf_vals[key] = np.zeros(n_actions)
            action_values = np.zeros(n_actions)
            for idx, action in enumerate(actions):
                value = self._cf_br_traverse(state.apply(action), deal_prob, opp_reach, br_player, cf_vals)
                action_values[idx] = value
                cf_vals[key][idx] += value
            return float(np.max(action_values) if br_player == 0 else np.min(action_values))

        strat = self.avg_strategy.get(key)
        if strat is None or len(strat) != n_actions:
            strat = np.ones(n_actions) / n_actions
        total = 0.0
        for idx, action in enumerate(actions):
            total += self._cf_br_traverse(state.apply(action), deal_prob, opp_reach * float(strat[idx]), br_player, cf_vals)
        return total

    def _eval_br_policy(self, state, prob, br_player, policy):
        if state.is_terminal():
            return prob * state.terminal_utility_p0()

        actions = state.get_actions()
        if not actions:
            return 0.0

        player = state.current_player
        key = state.info_key()
        if player == br_player:
            chosen = policy.get(key, 0)
            if chosen >= len(actions):
                chosen = 0
            return self._eval_br_policy(state.apply(actions[chosen]), prob, br_player, policy)

        strat = self.avg_strategy.get(key)
        if strat is None or len(strat) != len(actions):
            strat = np.ones(len(actions)) / len(actions)
        return sum(
            self._eval_br_policy(state.apply(actions[idx]), prob * float(strat[idx]), br_player, policy)
            for idx in range(len(actions))
        )


def _score_tier3(agent_tier):
    result = {"passed": False}
    if not isinstance(agent_tier, dict):
        result["error"] = "missing tier3 object"
        return result

    avg = agent_tier.get("average_strategy")
    game_params = agent_tier.get("game_params")
    n_info = agent_tier.get("n_information_sets")
    n_iterations = agent_tier.get("n_iterations")
    reported_exploitability = _safe_float(agent_tier.get("final_exploitability"))
    if (
        not isinstance(avg, dict)
        or not isinstance(game_params, dict)
        or not isinstance(n_info, int)
        or not isinstance(n_iterations, int)
        or n_iterations < 1
        or reported_exploitability is None
    ):
        result["error"] = "invalid tier3 schema"
        return result

    expected_params = {
        "ranks": [1, 2, 3, 4],
        "n_suits": 2,
        "deck_size": 8,
        "bet_sizes": {"1": 3, "2": 6},
        "max_raises": 2,
        "ante": 1,
    }
    if game_params != expected_params:
        result["error"] = "wrong tier3 game parameters"
        result["game_params"] = game_params
        return result

    if n_info != 504 or len(avg) != 504:
        result["error"] = "wrong tier3 information set count"
        result["n_information_sets"] = n_info
        result["len_average_strategy"] = len(avg)
        return result

    strategy = {}
    for key, value in avg.items():
        if not isinstance(value, list) or len(value) not in (2, 3):
            result["error"] = f"invalid tier3 strategy length at {key}"
            return result
        try:
            arr = np.array(value, dtype=float)
        except Exception:
            result["error"] = f"non-numeric tier3 strategy at {key}"
            return result
        if np.any(arr < -1e-9) or abs(float(arr.sum()) - 1.0) > 1e-6:
            result["error"] = f"invalid tier3 probabilities at {key}"
            return result
        strategy[key] = arr

    exploitability = float(LeducBestResponse(strategy).compute_exploitability())
    conditions = {
        "game_params": True,
        "n_information_sets": True,
        "valid_probabilities": True,
        "n_iterations_reported": n_iterations >= 1,
        "exploitability": exploitability < TIER3_EXPLOITABILITY_MAX,
        "reported_exploitability_matches": abs(reported_exploitability - exploitability) <= 0.02,
        "reported_exploitability_threshold": reported_exploitability < TIER3_EXPLOITABILITY_MAX,
        "theoretical_info_sets": len(count_leduc_info_sets()) == 504,
    }
    result.update(
        {
            "n_iterations": n_iterations,
            "exploitability": exploitability,
            "reported_exploitability": reported_exploitability,
            "conditions": conditions,
            "passed": all(conditions.values()),
        }
    )
    return result
