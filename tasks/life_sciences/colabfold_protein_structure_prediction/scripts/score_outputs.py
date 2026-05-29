#!/usr/bin/env python3
"""Local scorer for colabfold_protein_structure_prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pdb_tools import complete_residues, extract_sequence, kabsch_rmsd, parse_pdb_bytes


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    expected_length: int
    complete_backbone_residues: int
    min_complete_residues: int
    rmsd: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _paired_ca_coordinates(agent_residues, reference_residues) -> tuple[np.ndarray, np.ndarray]:
    if len(agent_residues) != len(reference_residues):
        raise ValueError("agent/reference residue lengths differ")
    agent_coords = []
    reference_coords = []
    for agent_residue, reference_residue in zip(agent_residues, reference_residues):
        if agent_residue.one_letter != reference_residue.one_letter:
            raise ValueError("agent/reference residue identities differ")
        if "CA" not in agent_residue.atoms or "CA" not in reference_residue.atoms:
            continue
        agent_coords.append(agent_residue.atoms["CA"])
        reference_coords.append(reference_residue.atoms["CA"])
    if not agent_coords:
        raise ValueError("no paired CA coordinates available")
    return np.vstack(agent_coords).astype(float), np.vstack(reference_coords).astype(float)


def _normalize_sequence(raw: bytes) -> str:
    lines = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    sequence = "".join(lines).strip().upper()
    if not sequence:
        raise ValueError("empty FASTA sequence")
    return sequence


def score_submission_bytes(
    agent_pdb: bytes,
    reference_pdb: bytes,
    expected_sequence: str,
    min_complete_residues: int,
) -> ScoreResult:
    try:
        agent_residues = parse_pdb_bytes(agent_pdb)
    except Exception as exc:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"invalid_pdb: {exc}",
            expected_length=len(expected_sequence),
            complete_backbone_residues=0,
            min_complete_residues=min_complete_residues,
            rmsd=None,
        )
    reference_residues = parse_pdb_bytes(reference_pdb)
    agent_complete = complete_residues(agent_residues)
    ref_complete = complete_residues(reference_residues)
    agent_sequence = extract_sequence(agent_residues)
    reference_sequence = extract_sequence(reference_residues)
    if reference_sequence != expected_sequence:
        raise ValueError("reference sequence does not match expected FASTA")
    if agent_sequence != expected_sequence:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="sequence_mismatch",
            expected_length=len(expected_sequence),
            complete_backbone_residues=len(agent_complete),
            min_complete_residues=min_complete_residues,
            rmsd=None,
        )
    if len(agent_complete) < min_complete_residues:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="insufficient_complete_backbone_residues",
            expected_length=len(expected_sequence),
            complete_backbone_residues=len(agent_complete),
            min_complete_residues=min_complete_residues,
            rmsd=None,
        )
    if len(ref_complete) < len(expected_sequence):
        raise ValueError("reference PDB is unexpectedly incomplete")
    agent_ca, reference_ca = _paired_ca_coordinates(agent_residues, reference_residues)
    if len(agent_ca) < min_complete_residues:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="insufficient_ca_pairs_for_rmsd",
            expected_length=len(expected_sequence),
            complete_backbone_residues=len(agent_complete),
            min_complete_residues=min_complete_residues,
            rmsd=None,
        )
    rmsd = kabsch_rmsd(agent_ca, reference_ca)
    if rmsd < 3.0:
        score = 1.0
        reason = "rmsd_below_3A"
    elif rmsd < 5.0:
        score = 0.7
        reason = "rmsd_between_3A_and_5A"
    else:
        score = 0.2
        reason = "rmsd_at_or_above_5A"
    return ScoreResult(
        score=score,
        passed=score >= 1.0,
        reason=reason,
        expected_length=len(expected_sequence),
        complete_backbone_residues=len(agent_complete),
        min_complete_residues=min_complete_residues,
        rmsd=rmsd,
    )


def score_submission_paths(
    submission_pdb: Path,
    reference_pdb: Path,
    fasta_path: Path,
    min_complete_residues: int,
) -> ScoreResult:
    if not submission_pdb.exists() or submission_pdb.stat().st_size == 0:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="missing_output_pdb",
            expected_length=len(_normalize_sequence(fasta_path.read_bytes())),
            complete_backbone_residues=0,
            min_complete_residues=min_complete_residues,
            rmsd=None,
        )
    return score_submission_bytes(
        agent_pdb=submission_pdb.read_bytes(),
        reference_pdb=reference_pdb.read_bytes(),
        expected_sequence=_normalize_sequence(fasta_path.read_bytes()),
        min_complete_residues=min_complete_residues,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a predicted protein structure PDB")
    parser.add_argument("--submission-pdb", required=True, type=Path)
    parser.add_argument("--reference-pdb", required=True, type=Path)
    parser.add_argument("--input-fasta", required=True, type=Path)
    parser.add_argument("--min-complete-residues", required=True, type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = score_submission_paths(
        submission_pdb=args.submission_pdb,
        reference_pdb=args.reference_pdb,
        fasta_path=args.input_fasta,
        min_complete_residues=args.min_complete_residues,
    )
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if result.score >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
