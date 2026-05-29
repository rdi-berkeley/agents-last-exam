"""Validate a lenacapavir SAR extraction CSV against the reference CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

EXPECTED_COLUMNS = ["Ligand_ID", "SMILES", "EC50_MT4"]


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
            ligand_id = normalized_row["Ligand_ID"]
            if not ligand_id:
                raise ValueError("Blank Ligand_ID encountered")
            if ligand_id in rows:
                raise ValueError(f"Duplicate Ligand_ID: {ligand_id}")
            rows[ligand_id] = normalized_row

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


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
                f"Ligand_ID set mismatch: agent={sorted(agent_rows)} ref={sorted(ref_rows)}"
            )

        total_rows = len(ref_rows)
        matched_rows = 0
        row_results: dict[str, dict[str, object]] = {}

        for ligand_id, ref_row in ref_rows.items():
            agent_row = agent_rows[ligand_id]
            agent_key = _inchi_key(agent_row["SMILES"])
            ref_key = _inchi_key(ref_row["SMILES"])
            smiles_match = agent_key is not None and ref_key is not None and agent_key == ref_key
            ec50_match = agent_row["EC50_MT4"] == ref_row["EC50_MT4"]
            row_match = smiles_match and ec50_match
            if row_match:
                matched_rows += 1
            row_results[ligand_id] = {
                "matched": row_match,
                "smiles_match": smiles_match,
                "ec50_match": ec50_match,
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
