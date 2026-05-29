#!/usr/bin/env python3
"""Score a candidate SGF against the hidden reference checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgfmill import boards, sgf

CHECKPOINTS = [10, 25, 50, 75, 100, 125, 150, 168]


def parse_moves(path: Path) -> tuple[int, list[tuple[str, tuple[int, int] | None]]]:
    with path.open("rb") as handle:
        game = sgf.Sgf_game.from_bytes(handle.read())
    size = game.get_size()
    moves: list[tuple[str, tuple[int, int] | None]] = []
    for node in game.get_main_sequence():
        color, point = node.get_move()
        if color is not None:
            moves.append((color, point))
    return size, moves


def replay_state(size: int, moves: list[tuple[str, tuple[int, int] | None]], checkpoint: int) -> set[tuple[int, int, str]]:
    board = boards.Board(size)
    for color, point in moves[:checkpoint]:
        if point is not None:
            row, col = point
            board.play(row, col, color)
    state: set[tuple[int, int, str]] = set()
    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            if stone is not None:
                state.add((row, col, stone))
    return state


def choose_candidate(candidate_dir: Path, preferred_name: str) -> Path:
    preferred = candidate_dir / preferred_name
    if preferred.exists():
        return preferred
    sgfs = sorted(candidate_dir.glob("*.sgf"))
    if len(sgfs) == 1:
        return sgfs[0]
    for fallback in ("ground-truth.sgf", "blank_19x19.sgf"):
        path = candidate_dir / fallback
        if path.exists():
            return path
    raise FileNotFoundError(f"could not identify candidate SGF in {candidate_dir}")


def score(candidate: Path, reference: Path) -> dict:
    candidate_size, candidate_moves = parse_moves(candidate)
    reference_size, reference_moves = parse_moves(reference)
    if candidate_size != 19 or reference_size != 19:
        raise ValueError("both games must be 19x19")
    checkpoints = [cp for cp in CHECKPOINTS if cp <= max(len(candidate_moves), len(reference_moves))]

    passed = 0
    details = []
    for checkpoint in checkpoints:
        candidate_state = replay_state(candidate_size, candidate_moves, checkpoint)
        reference_state = replay_state(reference_size, reference_moves, checkpoint)
        match = candidate_state == reference_state
        if match:
            passed += 1
        details.append(
            {
                "checkpoint": checkpoint,
                "passed": match,
                "candidate_stones": len(candidate_state),
                "reference_stones": len(reference_state),
            }
        )

    total = len(checkpoints)
    return {
        "candidate_path": str(candidate),
        "reference_path": str(reference),
        "checkpoints_passed": passed,
        "checkpoints_total": total,
        "score": (passed / total) if total else 0.0,
        "details": details,
    }


def score_candidate_dir(candidate_dir: Path, reference: Path, preferred_name: str) -> dict:
    candidate = choose_candidate(candidate_dir, preferred_name)
    return score(candidate, reference)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--preferred-name", default="reconstructed_game.sgf")
    args = parser.parse_args()

    try:
        data = score_candidate_dir(
            Path(args.candidate_dir),
            Path(args.reference),
            args.preferred_name,
        )
    except Exception as exc:
        data = {
            "score": 0.0,
            "error": str(exc),
            "candidate_dir": args.candidate_dir,
            "reference": args.reference,
        }

    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
