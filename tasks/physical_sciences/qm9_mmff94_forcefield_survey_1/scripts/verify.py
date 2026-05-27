"""VM-side verifier for physical_sciences/qm9_mmff94_forcefield_survey_1.

Uploaded to the eval temp dir at runtime by ``main.py::evaluate`` and invoked
with ``--agent <agent_output_dir> --reference <reference_dir>``. Prints a
single JSON object to stdout describing hard-gate status and weighted score;
diagnostic messages go to stderr.

Why VM-side: RDKit (``Chem.CanonSmiles``, ``Chem.MolFromSmiles``) is required
for SMILES canonicalization per the verification method, and RDKit is already
installed on the task VM but is not pinned into the agenthle evaluator env.
The outputs themselves are small (~0.6 MB), so this is a dep-driven exception,
not a payload-size one.
"""

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

try:
    from rdkit import Chem
except Exception as exc:  # pragma: no cover
    print(json.dumps({"score": 0.0, "error": f"rdkit import failed: {exc}"}))
    sys.exit(0)


REQUIRED_OUTPUT_FILES = [
    "force_field_failures.csv",
    "phase2_classified.csv",
    "phase3_scaffold_analysis.json",
    "phase4_pes_results.json",
    "phase5_final_report.json",
    "pes_scan_rank1.png",
    "pes_scan_rank2.png",
    "pes_scan_rank3.png",
    "pes_scan_rank4.png",
    "pes_scan_rank5.png",
]
RANK_KEYS = ["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]
PHASE5_TOP_KEYS = [
    "survey_statistics",
    "scaffold_analysis",
    "top5_characterization",
    "top5_pes_summary",
]

PHASE_WEIGHTS = {
    "phase1": 0.10,
    "phase2": 0.25,
    "phase3": 0.25,
    "phase4": 0.30,
    "phase5": 0.10,
}


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canon(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _num(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _eq_case_insensitive(a, b) -> bool:
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()


def _within(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _phase4_data(block):
    if not isinstance(block, dict):
        return None
    inner = block.get("data")
    if isinstance(inner, dict):
        return inner
    return block


def hard_gate(agent_dir: Path) -> str | None:
    for name in REQUIRED_OUTPUT_FILES:
        if not (agent_dir / name).exists():
            return f"missing required output file: {name}"

    try:
        p1 = _load_csv(agent_dir / "force_field_failures.csv")
    except Exception as exc:
        return f"force_field_failures.csv unparseable: {exc}"
    _ = p1

    try:
        p2 = _load_csv(agent_dir / "phase2_classified.csv")
    except Exception as exc:
        return f"phase2_classified.csv unparseable: {exc}"
    if p2 and "Classification" not in p2[0]:
        return "phase2_classified.csv missing Classification column"

    for name in (
        "phase3_scaffold_analysis.json",
        "phase4_pes_results.json",
        "phase5_final_report.json",
    ):
        try:
            _load_json(agent_dir / name)
        except Exception as exc:
            return f"{name} unparseable: {exc}"

    p4 = _load_json(agent_dir / "phase4_pes_results.json")
    for key in RANK_KEYS:
        if key not in p4:
            return f"phase4_pes_results.json missing {key}"
        data = _phase4_data(p4.get(key))
        if data is None:
            return f"phase4_pes_results.json {key} has no data block"
        smi = data.get("canonical_isomeric_smiles")
        if smi is None or Chem.MolFromSmiles(str(smi)) is None:
            return f"phase4_pes_results.json {key} canonical_isomeric_smiles fails RDKit parse"

    p5 = _load_json(agent_dir / "phase5_final_report.json")
    for key in PHASE5_TOP_KEYS:
        if key not in p5:
            return f"phase5_final_report.json missing {key}"

    pngs = list(agent_dir.glob("pes_scan_rank*.png"))
    if len(pngs) < 5:
        return f"fewer than 5 PNG files in output (found {len(pngs)})"

    return None


def _score_phase1(agent_csv, ref_csv):
    checks: list[bool] = []
    details: dict = {}

    row_diff = abs(len(agent_csv) - len(ref_csv))
    checks.append(row_diff <= 10)
    details["row_count_within_10"] = checks[-1]
    details["row_count_diff"] = row_diff

    all_geq_1 = True
    for row in agent_csv:
        disc = _num(row.get("Discrepancy_A"))
        if disc is None or disc < 1.0 - 1e-9:
            all_geq_1 = False
            break
    checks.append(all_geq_1)
    details["all_discrepancies_geq_1.0"] = all_geq_1

    by_id = {str(row.get("Molecule_ID")): row for row in agent_csv}
    spot_ids = [str(r.get("Molecule_ID")) for r in ref_csv[:10]]
    qm9_ok = 0
    disc_ok = 0
    spot_total = 0
    for mol_id in spot_ids:
        ref_row = next(
            (r for r in ref_csv if str(r.get("Molecule_ID")) == mol_id), None
        )
        agent_row = by_id.get(mol_id)
        if ref_row is None or agent_row is None:
            continue
        spot_total += 1
        if _within(
            _num(agent_row.get("QM9_Dist_A")),
            _num(ref_row.get("QM9_Dist_A")),
            0.005,
        ):
            qm9_ok += 1
        if _within(
            _num(agent_row.get("Discrepancy_A")),
            _num(ref_row.get("Discrepancy_A")),
            0.005,
        ):
            disc_ok += 1
    checks.append(spot_total > 0 and qm9_ok == spot_total)
    details["spot_qm9_dist_within_tol"] = {
        "matches": qm9_ok,
        "total": spot_total,
    }
    checks.append(spot_total > 0 and disc_ok == spot_total)
    details["spot_discrepancy_within_tol"] = {
        "matches": disc_ok,
        "total": spot_total,
    }
    return checks, details


def _top5_by_residual(rows):
    scored = []
    for row in rows:
        val = _num(row.get("Residual_Discrepancy_A"))
        if val is None:
            continue
        scored.append((val, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:5]]


def _score_phase2(agent_csv, ref_csv):
    checks: list[bool] = []
    details: dict = {}

    def _counts(rows):
        gen = sum(
            1 for r in rows if _eq_case_insensitive(r.get("Classification"), "genuine_ff_failure")
        )
        art = sum(
            1 for r in rows if _eq_case_insensitive(r.get("Classification"), "sampling_artifact")
        )
        return gen, art

    agent_gen, agent_art = _counts(agent_csv)
    ref_gen, ref_art = _counts(ref_csv)

    checks.append(abs(agent_gen - ref_gen) <= 5)
    details["genuine_count_within_5"] = {"agent": agent_gen, "ref": ref_gen}
    checks.append(abs(agent_art - ref_art) <= 5)
    details["artifact_count_within_5"] = {"agent": agent_art, "ref": ref_art}

    ref_frac = ref_art / (ref_gen + ref_art) if (ref_gen + ref_art) else 0.0
    agent_total = agent_gen + agent_art
    agent_frac = agent_art / agent_total if agent_total else 0.0
    checks.append(abs(agent_frac - ref_frac) <= 0.01)
    details["artifact_fraction_within_0.01"] = {
        "agent": round(agent_frac, 4),
        "ref": round(ref_frac, 4),
    }

    ref_top5 = _top5_by_residual(ref_csv)
    agent_by_id = {str(row.get("Molecule_ID")): row for row in agent_csv}

    gm_dist_ok = gm_energy_ok = resid_ok = cls_ok = 0
    for ref_row in ref_top5:
        agent_row = agent_by_id.get(str(ref_row.get("Molecule_ID")))
        if agent_row is None:
            continue
        if _within(
            _num(agent_row.get("Global_Min_Dist_A")),
            _num(ref_row.get("Global_Min_Dist_A")),
            0.010,
        ):
            gm_dist_ok += 1
        if _within(
            _num(agent_row.get("Global_Min_Energy_kcal")),
            _num(ref_row.get("Global_Min_Energy_kcal")),
            0.10,
        ):
            gm_energy_ok += 1
        if _within(
            _num(agent_row.get("Residual_Discrepancy_A")),
            _num(ref_row.get("Residual_Discrepancy_A")),
            0.010,
        ):
            resid_ok += 1
        if _eq_case_insensitive(
            agent_row.get("Classification"), ref_row.get("Classification")
        ):
            cls_ok += 1

    for _ in range(5):
        checks.append(False)
    # overwrite with per-row results: spec says "for each of the top 5 …" — per-row 1 check each
    # We model it as 4*5 = 20 per-row checks.
    # Replace the five placeholders above with the four-field aggregates expanded below.
    checks = checks[:-5]
    details["top5_global_min_dist_matches"] = gm_dist_ok
    details["top5_global_min_energy_matches"] = gm_energy_ok
    details["top5_residual_discrepancy_matches"] = resid_ok
    details["top5_classification_matches"] = cls_ok

    for idx in range(5):
        checks.append(idx < gm_dist_ok)
    for idx in range(5):
        checks.append(idx < gm_energy_ok)
    for idx in range(5):
        checks.append(idx < resid_ok)
    for idx in range(5):
        checks.append(idx < cls_ok)

    return checks, details


def _scaffold_canon(smi):
    if smi is None:
        return None
    s = str(smi).strip()
    if s.lower() == "acyclic":
        return "acyclic"
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return s
        return Chem.MolToSmiles(mol)
    except Exception:
        return s


def _top_functional_groups(fg_stats: dict) -> list[str]:
    if not isinstance(fg_stats, dict):
        return []
    entries = []
    for name, body in fg_stats.items():
        if isinstance(body, dict):
            entries.append((_num(body.get("mean_discrepancy")) or -1e9, name))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    return [name for _, name in entries[:5]]


def _score_phase3(agent_json, ref_json):
    checks: list[bool] = []
    details: dict = {}

    agent_stats = agent_json.get("survey_statistics", {}) if isinstance(agent_json, dict) else {}
    ref_stats = ref_json.get("survey_statistics", {})

    checks.append(
        abs(
            int(_num(agent_stats.get("genuine_ff_failures")) or -1e9)
            - int(_num(ref_stats.get("genuine_ff_failures")) or 0)
        )
        <= 5
    )
    details["genuine_ff_failures_within_5"] = {
        "agent": agent_stats.get("genuine_ff_failures"),
        "ref": ref_stats.get("genuine_ff_failures"),
    }

    checks.append(
        abs(
            int(_num(agent_stats.get("sampling_artifacts")) or -1e9)
            - int(_num(ref_stats.get("sampling_artifacts")) or 0)
        )
        <= 5
    )
    details["sampling_artifacts_within_5"] = {
        "agent": agent_stats.get("sampling_artifacts"),
        "ref": ref_stats.get("sampling_artifacts"),
    }

    checks.append(
        _within(
            _num(agent_stats.get("artifact_fraction")),
            _num(ref_stats.get("artifact_fraction")),
            0.01,
        )
    )
    details["artifact_fraction_within_0.01"] = {
        "agent": agent_stats.get("artifact_fraction"),
        "ref": ref_stats.get("artifact_fraction"),
    }

    agent_scaff = (agent_json.get("scaffold_analysis") or {}) if isinstance(agent_json, dict) else {}
    ref_scaff = ref_json.get("scaffold_analysis", {})

    a_most = _scaffold_canon(agent_scaff.get("most_affected_scaffold"))
    r_most = _scaffold_canon(ref_scaff.get("most_affected_scaffold"))
    checks.append(a_most is not None and a_most == r_most)
    details["most_affected_scaffold_match"] = {
        "agent": agent_scaff.get("most_affected_scaffold"),
        "ref": ref_scaff.get("most_affected_scaffold"),
    }

    a_top_fg = _top_functional_groups(agent_scaff.get("functional_group_stats") or {})
    r_top_fg = _top_functional_groups(ref_scaff.get("functional_group_stats") or {})
    checks.append(a_top_fg == r_top_fg and len(r_top_fg) > 0)
    details["top5_functional_groups_ordered_match"] = {
        "agent": a_top_fg,
        "ref": r_top_fg,
    }

    agent_top5 = agent_json.get("top5_worst_molecules") if isinstance(agent_json, dict) else None
    ref_top5 = ref_json.get("top5_worst_molecules") or []

    def _smiles_set(entries):
        out = set()
        if not isinstance(entries, list):
            return out
        for e in entries:
            if isinstance(e, dict):
                c = _canon(e.get("smiles"))
                if c:
                    out.add(c)
        return out

    checks.append(_smiles_set(agent_top5) == _smiles_set(ref_top5) and _smiles_set(ref_top5))
    details["top5_smiles_set_match"] = {
        "agent": sorted(_smiles_set(agent_top5)),
        "ref": sorted(_smiles_set(ref_top5)),
    }

    ref_resid_sorted = sorted(
        [v for v in
         [_num(e.get("residual_discrepancy")) for e in ref_top5 if isinstance(e, dict)]
         if v is not None],
        reverse=True,
    )
    agent_resid_sorted = sorted(
        [v for v in
         [_num(e.get("residual_discrepancy"))
          for e in (agent_top5 or [])
          if isinstance(e, dict)]
         if v is not None],
        reverse=True,
    )
    matches = 0
    for ref_val, agent_val in zip(ref_resid_sorted, agent_resid_sorted):
        if _within(agent_val, ref_val, 0.010):
            matches += 1
    checks.append(
        len(ref_resid_sorted) > 0 and matches == len(ref_resid_sorted)
    )
    details["top5_residual_discrepancy_within_tol"] = {
        "agent": agent_resid_sorted,
        "ref": ref_resid_sorted,
    }

    return checks, details


PHASE4_TOLS = {
    "qm9_heteroatom_distance": 0.001,
    "mmff94_global_min_distance": 0.010,
    "mmff94_global_min_energy": 0.10,
    "quantum_forced_energy": 0.10,
    "conformational_snap_distance": 0.10,
    "snap_energy_drop": 0.20,
    "delta_e": 0.15,
}


def _score_phase4(agent_json, ref_json, agent_dir: Path):
    checks: list[bool] = []
    details: dict = {}

    for rank in RANK_KEYS:
        agent_blk = _phase4_data(agent_json.get(rank) if isinstance(agent_json, dict) else None)
        ref_blk = _phase4_data(ref_json.get(rank))
        if ref_blk is None:
            for _ in range(10):
                checks.append(False)
            continue

        # canonical_isomeric_smiles via Chem.CanonSmiles
        a_smi = _canon(agent_blk.get("canonical_isomeric_smiles") if agent_blk else None)
        r_smi = _canon(ref_blk.get("canonical_isomeric_smiles"))
        checks.append(a_smi is not None and a_smi == r_smi)

        # heteroatom_pair exact (allow "O-N" vs "N-O")
        a_pair = str(agent_blk.get("heteroatom_pair", "")).strip() if agent_blk else ""
        r_pair = str(ref_blk.get("heteroatom_pair", "")).strip()
        checks.append(
            bool(r_pair)
            and (a_pair == r_pair or set(a_pair.split("-")) == set(r_pair.split("-")))
        )

        for field, tol in PHASE4_TOLS.items():
            agent_val = _num(agent_blk.get(field)) if agent_blk else None
            ref_val = _num(ref_blk.get(field))
            checks.append(_within(agent_val, ref_val, tol))

        # PNG existence — file pes_scan_rank{n}.png
        n = rank.split("_", 1)[1]
        checks.append((agent_dir / f"pes_scan_rank{n}.png").exists())

        details[rank] = {"agent_smiles": a_smi, "ref_smiles": r_smi}

    return checks, details


def _score_phase5(agent_json, ref_json, phase4_ref_json):
    checks: list[bool] = []
    details: dict = {}

    a_stats = agent_json.get("survey_statistics", {}) if isinstance(agent_json, dict) else {}
    r_stats = ref_json.get("survey_statistics", {})

    checks.append(
        abs(
            int(_num(a_stats.get("total_molecules_scanned")) or -1e9)
            - int(_num(r_stats.get("total_molecules_scanned")) or 0)
        )
        <= 100
    )
    details["total_molecules_scanned_within_100"] = {
        "agent": a_stats.get("total_molecules_scanned"),
        "ref": r_stats.get("total_molecules_scanned"),
    }

    checks.append(
        abs(
            int(_num(a_stats.get("phase1_candidates")) or -1e9)
            - int(_num(r_stats.get("phase1_candidates")) or 0)
        )
        <= 10
    )
    details["phase1_candidates_within_10"] = {
        "agent": a_stats.get("phase1_candidates"),
        "ref": r_stats.get("phase1_candidates"),
    }

    checks.append(
        abs(
            int(_num(a_stats.get("genuine_ff_failures")) or -1e9)
            - int(_num(r_stats.get("genuine_ff_failures")) or 0)
        )
        <= 5
    )
    checks.append(
        abs(
            int(_num(a_stats.get("sampling_artifacts")) or -1e9)
            - int(_num(r_stats.get("sampling_artifacts")) or 0)
        )
        <= 5
    )
    checks.append(
        _within(
            _num(a_stats.get("artifact_fraction")),
            _num(r_stats.get("artifact_fraction")),
            0.01,
        )
    )

    a_scaff = (agent_json.get("scaffold_analysis") or {}) if isinstance(agent_json, dict) else {}
    r_scaff = ref_json.get("scaffold_analysis", {})

    checks.append(
        _scaffold_canon(a_scaff.get("most_affected_scaffold"))
        == _scaffold_canon(r_scaff.get("most_affected_scaffold"))
    )
    checks.append(
        isinstance(a_scaff.get("top_functional_groups"), list)
        and a_scaff.get("top_functional_groups") == r_scaff.get("top_functional_groups")
    )

    a_char = (agent_json.get("top5_characterization") or {}) if isinstance(agent_json, dict) else {}
    r_char = ref_json.get("top5_characterization", {})

    def _str_set(xs):
        return set(str(x) for x in xs) if isinstance(xs, list) else set()

    checks.append(
        _str_set(a_char.get("short_qm9_distance_molecules"))
        == _str_set(r_char.get("short_qm9_distance_molecules"))
    )
    checks.append(
        _str_set(a_char.get("long_qm9_distance_molecules"))
        == _str_set(r_char.get("long_qm9_distance_molecules"))
    )

    # top5_pes_summary numeric match — condense to one binary check: all entries within Phase 4 tolerances.
    a_sum = agent_json.get("top5_pes_summary") if isinstance(agent_json, dict) else None
    r_sum = ref_json.get("top5_pes_summary") or []
    all_match = isinstance(a_sum, list) and len(a_sum) == len(r_sum) and len(r_sum) > 0
    if all_match:
        r_by_rank = {int(_num(e.get("rank")) or -1): e for e in r_sum if isinstance(e, dict)}
        for entry in a_sum:
            if not isinstance(entry, dict):
                all_match = False
                break
            rank = int(_num(entry.get("rank")) or -1)
            ref_entry = r_by_rank.get(rank)
            if ref_entry is None:
                all_match = False
                break
            a_pair = str(entry.get("heteroatom_pair", "")).strip()
            r_pair = str(ref_entry.get("heteroatom_pair", "")).strip()
            if not (a_pair == r_pair or set(a_pair.split("-")) == set(r_pair.split("-"))):
                all_match = False
                break
            for field, tol in PHASE4_TOLS.items():
                if not _within(_num(entry.get(field)), _num(ref_entry.get(field)), tol):
                    all_match = False
                    break
            if not all_match:
                break
    checks.append(all_match)
    details["top5_pes_summary_numeric_match"] = all_match

    return checks, details


def score(agent_dir: Path, ref_dir: Path) -> dict:
    gate = hard_gate(agent_dir)
    if gate is not None:
        return {"score": 0.0, "hard_gate": gate}

    agent_p1 = _load_csv(agent_dir / "force_field_failures.csv")
    agent_p2 = _load_csv(agent_dir / "phase2_classified.csv")
    agent_p3 = _load_json(agent_dir / "phase3_scaffold_analysis.json")
    agent_p4 = _load_json(agent_dir / "phase4_pes_results.json")
    agent_p5 = _load_json(agent_dir / "phase5_final_report.json")

    ref_p1 = _load_csv(ref_dir / "force_field_failures.csv")
    ref_p2 = _load_csv(ref_dir / "phase2_classified.csv")
    ref_p3 = _load_json(ref_dir / "phase3_scaffold_analysis.json")
    ref_p4 = _load_json(ref_dir / "phase4_pes_results.json")
    ref_p5 = _load_json(ref_dir / "phase5_final_report.json")

    p1_checks, p1_detail = _score_phase1(agent_p1, ref_p1)
    p2_checks, p2_detail = _score_phase2(agent_p2, ref_p2)
    p3_checks, p3_detail = _score_phase3(agent_p3, ref_p3)
    p4_checks, p4_detail = _score_phase4(agent_p4, ref_p4, agent_dir)
    p5_checks, p5_detail = _score_phase5(agent_p5, ref_p5, ref_p4)

    def frac(checks):
        return sum(1 for c in checks if c) / max(1, len(checks))

    phase_fracs = {
        "phase1": frac(p1_checks),
        "phase2": frac(p2_checks),
        "phase3": frac(p3_checks),
        "phase4": frac(p4_checks),
        "phase5": frac(p5_checks),
    }

    weighted = sum(PHASE_WEIGHTS[ph] * phase_fracs[ph] for ph in phase_fracs)

    return {
        "score": round(max(0.0, min(1.0, weighted)), 6),
        "hard_gate": None,
        "phase_fracs": phase_fracs,
        "phase_check_counts": {
            "phase1": {"passed": sum(1 for c in p1_checks if c), "total": len(p1_checks)},
            "phase2": {"passed": sum(1 for c in p2_checks if c), "total": len(p2_checks)},
            "phase3": {"passed": sum(1 for c in p3_checks if c), "total": len(p3_checks)},
            "phase4": {"passed": sum(1 for c in p4_checks if c), "total": len(p4_checks)},
            "phase5": {"passed": sum(1 for c in p5_checks if c), "total": len(p5_checks)},
        },
        "phase_details": {
            "phase1": p1_detail,
            "phase2": p2_detail,
            "phase3": p3_detail,
            "phase4": p4_detail,
            "phase5": p5_detail,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="Path to agent output directory")
    ap.add_argument("--reference", required=True, help="Path to reference directory")
    args = ap.parse_args()

    try:
        result = score(Path(args.agent), Path(args.reference))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result = {"score": 0.0, "error": f"verifier crashed: {exc}"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
