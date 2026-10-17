"""Audit the required IDP conformer pools, optionally applying a data overlay."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy


def structure_signature(path):
    import numpy as np
    from Bio.PDB import PDBParser

    structure = PDBParser(PERMISSIVE=False, QUIET=True).get_structure("audit", str(path))
    models = list(structure)
    if len(models) != 1 or [chain.id for chain in models[0]] != ["A"]:
        raise ValueError("expected one nonempty model with chain A")
    residues = list(models[0]["A"])
    if not residues:
        raise ValueError("empty structure")
    signature = []
    for residue in residues:
        missing = {"N", "CA", "C"} - set(residue.child_dict)
        if missing:
            raise ValueError("residue {} missing backbone {}".format(residue.id, sorted(missing)))
        atoms = list(residue)
        if not np.isfinite([atom.coord for atom in atoms]).all():
            raise ValueError("nonfinite atom coordinates")
        signature.append((residue.id, residue.resname, tuple(sorted(atom.id for atom in atoms))))
    return tuple(signature)


def audit(input_dir, overlay=None):
    protein_sets = runpy.run_path(str(input_dir / "info.py"))["PROTEINS_TO_TEST"]
    proteins = sorted(set().union(*(set(protein_sets[key]) for key in ("CS", "JC", "NOE", "PRE"))))
    report = {
        "proteins": proteins,
        "ensembles": 0,
        "conformers": 0,
        "issues": [],
        "atom_inventory_variations": {},
    }
    digest = hashlib.sha256()
    for model in range(1, 6):
        for protein in proteins:
            relative = Path("Ensembles") / ("Model{}".format(model)) / protein
            paths = sorted((input_dir / relative).glob("*.pdb"))
            report["ensembles"] += 1
            if len(paths) != 200:
                report["issues"].append(
                    [str(relative), "expected 200 conformers, found {}".format(len(paths))]
                )
            expected = None
            for original in paths:
                key = original.relative_to(input_dir)
                path = overlay / key if overlay and (overlay / key).is_file() else original
                report["conformers"] += 1
                digest.update(
                    str(key).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
                )
                try:
                    signature = structure_signature(path)
                    if expected is None:
                        expected = signature
                    elif [residue[:2] for residue in signature] != [
                        residue[:2] for residue in expected
                    ]:
                        raise ValueError("residue identity/numbering differs within ensemble")
                    elif signature != expected:
                        for observed, baseline in zip(signature, expected):
                            if observed != baseline:
                                difference = sorted(set(observed[2]) ^ set(baseline[2]))
                                label = observed[1] + ":" + ",".join(difference)
                                variations = report["atom_inventory_variations"]
                                variations[label] = variations.get(label, 0) + 1
                except Exception as error:
                    report["issues"].append([str(key), str(error)])
    report["manifest_sha256"] = digest.hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.input_dir, args.overlay)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "issues": report["issues"][:10]}, indent=2))
    return bool(report["issues"])


if __name__ == "__main__":
    raise SystemExit(main())
