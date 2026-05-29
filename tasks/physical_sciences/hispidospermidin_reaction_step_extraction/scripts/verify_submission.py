"""Validate a reaction-step extraction JSON against the reference JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")

COMPOUND_IDS = {"2", "3"}
TEXT_SPACE_RE = re.compile(r"\s+")
NUMERIC_TOLERANCE = 1e-3


def _normalize_text(value: object) -> str:
    return TEXT_SPACE_RE.sub(" ", str(value).strip()).casefold()


def _normalize_formula(value: object) -> str:
    return str(value).replace(" ", "").upper()


def _load_json(path: str) -> dict:
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Input must end with .json: {path}")
    if not file_path.exists():
        raise FileNotFoundError(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc


def _inchi_key(smiles: object) -> str | None:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _same_float(agent_value: object, ref_value: object, tol: float = NUMERIC_TOLERANCE) -> bool:
    try:
        return math.isclose(float(agent_value), float(ref_value), abs_tol=tol, rel_tol=0.0)
    except Exception:
        return False


def _require_schema(data: dict) -> tuple[dict[str, dict], dict]:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")
    if "compounds" not in data or "step" not in data:
        raise ValueError("Top-level keys 'compounds' and 'step' are required")
    compounds = data["compounds"]
    step = data["step"]
    if not isinstance(compounds, list) or len(compounds) != 2:
        raise ValueError("compounds must be a 2-entry list")
    if not isinstance(step, dict):
        raise ValueError("step must be an object")

    indexed: dict[str, dict] = {}
    for entry in compounds:
        if not isinstance(entry, dict):
            raise ValueError("Each compound entry must be an object")
        compound_id = str(entry.get("compound_id", "")).strip()
        if not compound_id:
            raise ValueError("Each compound must have compound_id")
        if compound_id in indexed:
            raise ValueError(f"Duplicate compound_id: {compound_id}")
        indexed[compound_id] = entry

    if set(indexed) != COMPOUND_IDS:
        raise ValueError(f"Compound_ID set mismatch: agent={sorted(indexed)} ref={sorted(COMPOUND_IDS)}")
    return indexed, step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "score": 0.0,
        "matched_fields": 0,
        "total_fields": 0,
        "passed": False,
        "reason": "",
        "field_results": {},
    }

    try:
        agent = _load_json(args.agent)
        ref = _load_json(args.ref)

        agent_compounds, agent_step = _require_schema(agent)
        ref_compounds, ref_step = _require_schema(ref)

        field_results: dict[str, object] = {}
        matched_fields = 0
        total_fields = 0

        compound_text_fields = ["iupac", "hrms_ion"]
        compound_formula_fields = ["molecular_formula"]
        compound_numeric_fields = ["hrms_calculated", "hrms_found"]

        for compound_id in sorted(COMPOUND_IDS):
            agent_entry = agent_compounds[compound_id]
            ref_entry = ref_compounds[compound_id]
            compound_result: dict[str, bool] = {}

            for field in compound_text_fields:
                total_fields += 1
                ok = _normalize_text(agent_entry.get(field, "")) == _normalize_text(ref_entry.get(field, ""))
                matched_fields += int(ok)
                compound_result[field] = ok

            total_fields += 1
            smiles_ok = _inchi_key(agent_entry.get("smiles")) == _inchi_key(ref_entry.get("smiles"))
            matched_fields += int(smiles_ok)
            compound_result["smiles"] = smiles_ok

            for field in compound_formula_fields:
                total_fields += 1
                ok = _normalize_formula(agent_entry.get(field, "")) == _normalize_formula(ref_entry.get(field, ""))
                matched_fields += int(ok)
                compound_result[field] = ok

            for field in compound_numeric_fields:
                total_fields += 1
                ok = _same_float(agent_entry.get(field), ref_entry.get(field))
                matched_fields += int(ok)
                compound_result[field] = ok

            field_results[f"compound_{compound_id}"] = compound_result

        step_text_fields = [
            "reaction_name",
            "reagents",
            "solvent",
            "temperature_c",
            "time",
            "workup",
            "notes",
        ]
        step_list_fields = ["reactant_ids", "product_ids"]

        step_result: dict[str, bool] = {}

        total_fields += 1
        ok = str(agent_step.get("step_number")) == str(ref_step.get("step_number"))
        matched_fields += int(ok)
        step_result["step_number"] = ok

        for field in step_text_fields:
            total_fields += 1
            ok = _normalize_text(agent_step.get(field, "")) == _normalize_text(ref_step.get(field, ""))
            matched_fields += int(ok)
            step_result[field] = ok

        for field in step_list_fields:
            total_fields += 1
            agent_list = [str(x).strip() for x in agent_step.get(field, [])] if isinstance(agent_step.get(field, []), list) else []
            ref_list = [str(x).strip() for x in ref_step.get(field, [])] if isinstance(ref_step.get(field, []), list) else []
            ok = agent_list == ref_list
            matched_fields += int(ok)
            step_result[field] = ok

        total_fields += 1
        ok = _same_float(agent_step.get("yield_pct"), ref_step.get("yield_pct"), tol=0.5)
        matched_fields += int(ok)
        step_result["yield_pct"] = ok

        payload["score"] = matched_fields / total_fields if total_fields else 0.0
        payload["matched_fields"] = matched_fields
        payload["total_fields"] = total_fields
        payload["passed"] = matched_fields == total_fields
        payload["reason"] = "ok" if matched_fields == total_fields else "field_mismatch"
        field_results["step"] = step_result
        payload["field_results"] = field_results
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
