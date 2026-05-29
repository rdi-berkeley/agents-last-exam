"""Verify symbolic amplitude expressions with local SymPy normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

REQUIRED_KEYS = ("V_cc", "F_cc", "V_sc", "F_sc")
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

eps, mu2, s12, s23, s34, s45, s15 = sp.symbols("eps mu2 s12 s23 s34 s45 s15")
spa = sp.Function("spa")
sand = sp.Function("sand")
PolyLog = sp.polylog
Gamma = sp.gamma


def L0(value):
    return sp.log(value)


def L1(value):
    return sp.log(value) / (1 - value)


def LS_m1(r1, r2):
    return sp.polylog(2, 1 - r1) + sp.polylog(2, 1 - r2) + sp.log(r1) * sp.log(r2) - sp.pi**2 / 6


def _read_submission(path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"submission must be a JSON object: {path}")

    normalized: dict[str, str] = {}
    for key in REQUIRED_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing_or_invalid_key:{key}")
        normalized[key] = value.strip()
    return normalized


def _normalize_source(expr: str) -> str:
    return (
        expr.replace("[", "(")
        .replace("]", ")")
        .replace("\\\n", " ")
        .replace("\n", " ")
    )


def _local_dict() -> dict[str, object]:
    return {
        "eps": eps,
        "mu2": mu2,
        "s12": s12,
        "s23": s23,
        "s34": s34,
        "s45": s45,
        "s15": s15,
        "spa": spa,
        "sand": sand,
        "L0": L0,
        "L1": L1,
        "LS_m1": LS_m1,
        "Log": sp.log,
        "PolyLog": PolyLog,
        "Gamma": Gamma,
        "Pi": sp.pi,
    }


def _parse_expr(text: str) -> sp.Expr:
    return parse_expr(
        _normalize_source(text),
        local_dict=_local_dict(),
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def _equivalent(left_text: str, right_text: str) -> bool:
    left = _parse_expr(left_text)
    right = _parse_expr(right_text)
    diff = sp.expand(left - right)
    if diff == 0:
        return True
    compact = sp.cancel(sp.together(diff))
    if compact == 0:
        return True

    function_atoms = sorted(
        compact.atoms(sp.Function),
        key=sp.default_sort_key,
    )
    opaque_subs = {
        func_atom: sp.Symbol(f"_f_{index}")
        for index, func_atom in enumerate(function_atoms)
    }
    numeric_trials = [
        {eps: sp.Rational(1, 7), mu2: 2, s12: 3, s23: 5, s34: 11, s45: 7, s15: 13},
        {eps: sp.Rational(1, 9), mu2: 5, s12: 8, s23: 3, s34: 10, s45: 6, s15: 14},
        {eps: sp.Rational(1, 11), mu2: 7, s12: 4, s23: 9, s34: 15, s45: 10, s15: 12},
    ]
    opaque_numeric = [
        {sp.Symbol(f"_f_{index}"): sp.Rational(index + 2, index + 3) for index in range(len(function_atoms))},
        {sp.Symbol(f"_f_{index}"): sp.Rational(index + 5, index + 7) for index in range(len(function_atoms))},
        {sp.Symbol(f"_f_{index}"): sp.Rational(index + 11, index + 13) for index in range(len(function_atoms))},
    ]

    reduced = compact.xreplace(opaque_subs)
    for trial_idx, symbol_values in enumerate(numeric_trials):
        substituted = reduced.subs(symbol_values).subs(opaque_numeric[trial_idx])
        try:
            magnitude = abs(complex(sp.N(substituted, 50)))
        except Exception:
            return False
        if magnitude > 1e-10:
            return False
    return True


def score_submission_payload(agent: dict[str, str], reference: dict[str, str]) -> dict[str, object]:
    normalized = {key: _equivalent(agent[key], reference[key]) for key in REQUIRED_KEYS}
    score = sum(1.0 for matched in normalized.values() if matched) / 4.0
    return {
        "score": score,
        "passed": score == 1.0,
        "reason": "ok" if score == 1.0 else "symbolic_mismatch",
        "component_scores": normalized,
    }


def score_submission_texts(agent_text: str, reference_text: str) -> dict[str, object]:
    return score_submission_payload(json.loads(agent_text), json.loads(reference_text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    payload = {
        "score": 0.0,
        "passed": False,
        "reason": "",
        "component_scores": {},
    }

    try:
        reference = _read_submission(args.ref)
        agent = _read_submission(args.agent)
        payload = score_submission_payload(agent, reference)
        print(json.dumps(payload))
        return 0
    except FileNotFoundError as exc:
        payload["reason"] = f"missing_file:{exc}"
    except json.JSONDecodeError as exc:
        payload["reason"] = f"json_error:{exc}"
    except ValueError as exc:
        payload["reason"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive path
        payload["reason"] = f"unexpected_error:{type(exc).__name__}:{exc}"

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
