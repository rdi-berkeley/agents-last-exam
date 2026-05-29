"""Validate a single-entry SMILES submission against a reference SMILES file."""

import argparse
import json
import sys
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.warning")


def read_single_smiles(path: str) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() != ".smi":
        raise ValueError(f"Input must end with .smi: {path}")
    if not file_path.exists():
        raise FileNotFoundError(path)

    smiles = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        smiles.append(line.split()[0])

    if len(smiles) != 1:
        raise ValueError(f"Expected exactly 1 SMILES in {path}, found {len(smiles)}")
    return smiles[0]


def compute_similarity(agent_smiles: str, ref_smiles: str) -> float:
    agent_mol = Chem.MolFromSmiles(agent_smiles)
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if agent_mol is None or ref_mol is None:
        raise ValueError("Invalid SMILES in agent or reference file")

    agent_fp = AllChem.GetMorganFingerprintAsBitVect(agent_mol, radius=2, nBits=2048)
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(agent_fp, ref_fp))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    payload = {
        "score": 0.0,
        "similarity": 0.0,
        "threshold": args.threshold,
        "passed": False,
        "reason": "",
    }

    try:
        agent_smiles = read_single_smiles(args.agent)
        ref_smiles = read_single_smiles(args.ref)
        similarity = compute_similarity(agent_smiles, ref_smiles)
        payload["similarity"] = similarity
        payload["passed"] = similarity >= args.threshold
        payload["score"] = 1.0 if payload["passed"] else 0.0
        payload["reason"] = "ok" if payload["passed"] else "below_threshold"
    except FileNotFoundError as exc:
        payload["reason"] = f"missing_file:{exc}"
    except ValueError as exc:
        payload["reason"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive path
        payload["reason"] = f"unexpected_error:{type(exc).__name__}:{exc}"

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
