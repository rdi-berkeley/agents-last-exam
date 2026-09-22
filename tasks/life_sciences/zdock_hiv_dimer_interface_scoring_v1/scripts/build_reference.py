#!/usr/bin/env python3
"""Regenerate reference/zdock_v4_reference_output.csv from the staged inputs.

Implements the task prompt literally:

* protein ``ATOM`` records only (``HETATM`` excluded), heavy atoms only;
* interface residue: any residue with at least one heavy atom within 5 A of any heavy atom
  on the opposite chain;
* Overlap Score = |predicted interface residues ∩ native interface residues| / |native|;
* Fnat = fraction of native heavy-atom contact pairs (< = 5 A across chains) recovered;
* IRMSD = C-alpha RMSD over the native interface residues after Kabsch superposition of the
  predicted structure onto the native structure on those same C-alpha atoms;
* Final Score = 0.5 * Fnat + 0.3 * Overlap Score - 0.2 * (IRMSD / 10);
* Final Rank orders poses by Final Score, higher is better (ties broken by ZDOCK rank).

Usage::

    python build_reference.py --input-dir <task>/input --output <csv> [--decimals 6]
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path

import numpy as np

CONTACT_CUTOFF = 5.0
HYDROGEN_ELEMENTS = {"H", "D"}


def _element(atom_name: str, element_field: str) -> str:
    element = element_field.strip().upper()
    if element:
        return element
    name = atom_name.strip()
    # PDB atom names right-justify the element in columns 13-14; a digit in column 13 means a
    # hydrogen (e.g. "1HB1"). Otherwise the first alphabetic character is the element.
    if name and name[0].isdigit():
        return "H"
    for char in name:
        if char.isalpha():
            return char.upper()
    return ""


def parse_heavy_atoms(text: str) -> list[tuple[str, tuple[str, str, str], str, np.ndarray]]:
    """Return (chain, residue_key, atom_name, xyz) for every protein heavy atom."""
    atoms = []
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16]
        alt_loc = line[16]
        res_name = line[17:20].strip()
        chain = line[21]
        res_seq = line[22:26].strip()
        i_code = line[26].strip()
        element = _element(atom_name, line[76:78] if len(line) >= 78 else "")
        if element in HYDROGEN_ELEMENTS:
            continue
        if alt_loc not in (" ", "", "A", "1"):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        atoms.append((chain, (chain, res_seq, i_code), atom_name.strip(), xyz))
    return atoms


def split_chains(atoms, chain_a: str = "A", chain_b: str = "B"):
    a = [atom for atom in atoms if atom[0] == chain_a]
    b = [atom for atom in atoms if atom[0] == chain_b]
    if not a or not b:
        raise ValueError(f"missing chain {chain_a!r} or {chain_b!r}")
    return a, b


def cross_chain_contacts(a, b, cutoff: float = CONTACT_CUTOFF):
    """Return (set of residue keys on both chains, set of (atom_id_a, atom_id_b) contact pairs)."""
    xa = np.stack([atom[3] for atom in a])
    xb = np.stack([atom[3] for atom in b])
    d2 = ((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1)
    ia, ib = np.nonzero(d2 <= cutoff * cutoff)
    residues = set()
    contacts = set()
    for i, j in zip(ia.tolist(), ib.tolist()):
        residues.add(a[i][1])
        residues.add(b[j][1])
        contacts.add(((a[i][1], a[i][2]), (b[j][1], b[j][2])))
    return residues, contacts


def ca_coordinates(atoms, residue_keys):
    coords = {}
    for _, key, name, xyz in atoms:
        if name == "CA" and key in residue_keys and key not in coords:
            coords[key] = xyz
    return coords


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    h = mobile_c.T @ target_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    aligned = mobile_c @ rotation.T
    return float(np.sqrt(((aligned - target_c) ** 2).sum(axis=1).mean()))


def score_pose(pose_text: str, native_residues, native_contacts, native_ca):
    atoms = parse_heavy_atoms(pose_text)
    a, b = split_chains(atoms)
    residues, contacts = cross_chain_contacts(a, b)
    overlap = len(residues & native_residues) / len(native_residues)
    fnat = len(contacts & native_contacts) / len(native_contacts)
    pose_ca = ca_coordinates(atoms, native_residues)
    keys = [key for key in sorted(native_residues) if key in pose_ca and key in native_ca]
    if len(keys) < 3:
        raise ValueError("fewer than 3 interface C-alpha atoms shared with the native structure")
    irmsd = kabsch_rmsd(np.stack([pose_ca[k] for k in keys]), np.stack([native_ca[k] for k in keys]))
    final = 0.5 * fnat + 0.3 * overlap - 0.2 * (irmsd / 10.0)
    return overlap, fnat, irmsd, final


def build(input_dir: Path, decimals: int) -> list[dict[str, object]]:
    native_atoms = parse_heavy_atoms((input_dir / "1HVR.pdb").read_text(errors="replace"))
    native_a, native_b = split_chains(native_atoms)
    native_residues, native_contacts = cross_chain_contacts(native_a, native_b)
    native_ca = ca_coordinates(native_atoms, native_residues)

    poses = {}
    with tarfile.open(input_dir / "top_preds.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if member.isfile() and name.startswith("complex.") and name.endswith(".pdb"):
                rank = int(name.split(".")[1])
                poses[rank] = archive.extractfile(member).read().decode("utf-8", errors="replace")
    if sorted(poses) != list(range(1, 11)):
        raise ValueError(f"expected complex.1.pdb .. complex.10.pdb, found ranks {sorted(poses)}")

    rows = []
    for rank in range(1, 11):
        overlap, fnat, irmsd, final = score_pose(poses[rank], native_residues, native_contacts, native_ca)
        rows.append({"Pose Rank (ZDOCK)": rank, "Overlap Score": overlap, "Fnat": fnat, "IRMSD": irmsd, "Final Score": final})
    order = sorted(rows, key=lambda r: (-r["Final Score"], r["Pose Rank (ZDOCK)"]))
    for final_rank, row in enumerate(order, start=1):
        row["Final Rank"] = final_rank
    for row in rows:
        for key in ("Overlap Score", "Fnat", "IRMSD", "Final Score"):
            row[key] = round(row[key], decimals)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="directory with 1HVR.pdb and top_preds.tar.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args()
    rows = build(args.input_dir, args.decimals)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["Pose Rank (ZDOCK)", "Overlap Score", "Fnat", "IRMSD", "Final Score", "Final Rank"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    args.output.write_text(buffer.getvalue(), encoding="utf-8")
    print(buffer.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
