"""PDB parsing and scoring helpers for colabfold_protein_structure_prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class ParsedResidue:
    key: tuple[str, str, str]
    resname: str
    atoms: dict[str, np.ndarray]

    @property
    def one_letter(self) -> str | None:
        return THREE_TO_ONE.get(self.resname)

    @property
    def is_complete(self) -> bool:
        return all(atom in self.atoms for atom in BACKBONE_ATOMS)


def parse_pdb_bytes(payload: bytes) -> list[ParsedResidue]:
    text = payload.decode("utf-8", errors="replace")
    residues: list[ParsedResidue] = []
    current_key: tuple[str, str, str] | None = None
    current_resname: str | None = None
    current_atoms: dict[str, np.ndarray] = {}
    in_first_model = False
    saw_model = False

    def flush_current() -> None:
        nonlocal current_key, current_resname, current_atoms
        if current_key is None or current_resname is None:
            return
        residues.append(ParsedResidue(current_key, current_resname, dict(current_atoms)))
        current_key = None
        current_resname = None
        current_atoms = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("MODEL"):
            if saw_model:
                flush_current()
                break
            saw_model = True
            in_first_model = True
            continue
        if saw_model and line.startswith("ENDMDL"):
            flush_current()
            break
        if not line.startswith("ATOM"):
            continue
        if saw_model and not in_first_model:
            continue
        atom_name = line[12:16].strip()
        alt_loc = line[16].strip()
        if alt_loc not in ("", "A"):
            continue
        resname = line[17:20].strip().upper()
        chain_id = line[21].strip() or "_"
        resseq = line[22:26].strip()
        icode = line[26].strip() or "_"
        key = (chain_id, resseq, icode)
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError as exc:
            raise ValueError(f"invalid ATOM coordinates in line: {line!r}") from exc
        if current_key != key:
            flush_current()
            current_key = key
            current_resname = resname
        if current_resname is None:
            current_resname = resname
        if atom_name not in current_atoms:
            current_atoms[atom_name] = np.array([x, y, z], dtype=float)
    flush_current()
    if not residues:
        raise ValueError("no ATOM residues found")
    return residues


def extract_sequence(residues: list[ParsedResidue]) -> str:
    letters: list[str] = []
    for residue in residues:
        letter = residue.one_letter
        if letter is None:
            raise ValueError(f"unsupported residue name {residue.resname!r}")
        letters.append(letter)
    return "".join(letters)


def complete_residues(residues: list[ParsedResidue]) -> list[ParsedResidue]:
    return [residue for residue in residues if residue.one_letter and residue.is_complete]


def ca_coordinates(residues: list[ParsedResidue]) -> np.ndarray:
    return np.vstack([residue.atoms["CA"] for residue in residues]).astype(float)


def kabsch_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 3:
        raise ValueError(f"invalid coordinate shapes {left.shape} vs {right.shape}")
    if left.shape[0] == 0:
        raise ValueError("need at least one coordinate pair")
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    covariance = left_centered.T @ right_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    aligned = left_centered @ rotation
    diff = aligned - right_centered
    return float(math.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def format_fasta(header: str, sequence: str, line_width: int = 80) -> str:
    lines = [f">{header}"]
    for idx in range(0, len(sequence), line_width):
        lines.append(sequence[idx : idx + line_width])
    return "\n".join(lines) + "\n"


def truncate_pdb_bytes(payload: bytes, keep_residue_count: int) -> bytes:
    if keep_residue_count <= 0:
        raise ValueError("keep_residue_count must be positive")
    text = payload.decode("utf-8", errors="replace")
    kept_lines: list[str] = []
    seen_keys: list[tuple[str, str, str]] = []
    seen_lookup: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        if line.startswith("ATOM"):
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip() or "_"
            key = (chain_id, resseq, icode)
            if key not in seen_lookup:
                seen_lookup.add(key)
                seen_keys.append(key)
            if len(seen_keys) > keep_residue_count:
                continue
        kept_lines.append(line)
    return ("\n".join(kept_lines) + "\n").encode("utf-8")
