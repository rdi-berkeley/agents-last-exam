"""Validate a SAR extraction CSV against the reference CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

EXPECTED_COLUMNS = ["Compound_ID", "SMILES", "IC50_uM", "Solubility_mM"]
COMPARATOR_RE = re.compile(r"^(<=|>=|<|>)\s*(.+)$")


def _normalized_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _parse_csv(path: str) -> dict[str, dict[str, str]]:
    file_path = Path(path)
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Input must end with .csv: {path}")
    if not file_path.exists():
        raise FileNotFoundError(path)

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        normalized_headers = [_normalized_text(name) for name in fieldnames]
        if normalized_headers != EXPECTED_COLUMNS:
            raise ValueError(f"Expected headers {EXPECTED_COLUMNS}, got {normalized_headers}")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            normalized_row = {
                key: _normalized_text(row.get(key, ""))
                for key in EXPECTED_COLUMNS
            }
            compound_id = normalized_row["Compound_ID"]
            if not compound_id:
                raise ValueError("Blank Compound_ID encountered")
            if compound_id in rows:
                raise ValueError(f"Duplicate Compound_ID: {compound_id}")
            rows[compound_id] = normalized_row

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _normalize_measurement(value: str) -> tuple[str, str]:
    text = _normalized_text(value).replace("μ", "u")
    match = COMPARATOR_RE.match(text)
    if match:
        return ("cmp", f"{match.group(1)}{match.group(2).strip()}")
    try:
        return ("num", str(Decimal(text)))
    except InvalidOperation:
        return ("txt", text)


def _same_measurement(agent_value: str, ref_value: str) -> bool:
    agent_kind, agent_norm = _normalize_measurement(agent_value)
    ref_kind, ref_norm = _normalize_measurement(ref_value)
    if agent_kind != ref_kind:
        return False
    if agent_kind == "num":
        return Decimal(agent_norm) == Decimal(ref_norm)
    return agent_norm == ref_norm


def _inchi_key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "score": 0.0,
        "matched_rows": 0,
        "total_rows": 0,
        "passed": False,
        "reason": "",
        "row_results": {},
    }

    try:
        agent_rows = _parse_csv(args.agent)
        ref_rows = _parse_csv(args.ref)
        if set(agent_rows) != set(ref_rows):
            raise ValueError(
                f"Compound_ID set mismatch: agent={sorted(agent_rows)} ref={sorted(ref_rows)}"
            )

        total_rows = len(ref_rows)
        matched_rows = 0
        row_results: dict[str, dict[str, object]] = {}

        for compound_id, ref_row in ref_rows.items():
            agent_row = agent_rows[compound_id]
            agent_key = _inchi_key(agent_row["SMILES"])
            ref_key = _inchi_key(ref_row["SMILES"])
            smiles_match = agent_key is not None and ref_key is not None and agent_key == ref_key
            ic50_match = _same_measurement(agent_row["IC50_uM"], ref_row["IC50_uM"])
            solubility_match = _same_measurement(agent_row["Solubility_mM"], ref_row["Solubility_mM"])
            row_match = smiles_match and ic50_match and solubility_match
            if row_match:
                matched_rows += 1
            row_results[compound_id] = {
                "matched": row_match,
                "smiles_match": smiles_match,
                "ic50_match": ic50_match,
                "solubility_match": solubility_match,
            }

        payload["score"] = matched_rows / total_rows
        payload["matched_rows"] = matched_rows
        payload["total_rows"] = total_rows
        payload["passed"] = matched_rows == total_rows
        payload["reason"] = "ok" if matched_rows == total_rows else "row_mismatch"
        payload["row_results"] = row_results
    except FileNotFoundError as exc:
        payload["reason"] = f"missing_file:{exc}"
    except ValueError as exc:
        payload["reason"] = str(exc)
    except Exception as exc:
        payload["reason"] = f"unexpected_error:{type(exc).__name__}:{exc}"

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
