#!/usr/bin/env python3
"""Local scorer for the HG002 chr22 germline benchmark.

Scores an agent's `submission/` directory against a 100-point rubric with
local recomputation of SNP/INDEL TP/FP/FN, Ti/Tv, and PASS counts from the
submitted filtered VCF against a hidden copy of the GIAB HG002 chr22 truth.

Unlike the raw submitter evaluator, this benchmark-facing scorer does not
require the agent to have visible truth data or to emit hap.py benchmark
artifacts. It only requires solve-time outputs the agent can produce from the
visible starter project plus standard workflow tooling.

Usage:
  python score_outputs.py \\
      --submission <dir> \\
      --hidden-truth-dir <dir> \\
      [--out <json>]

Expected submission layout:
  submission/
    pipeline/
      samplesheet.csv                          # corrected
      nextflow.config                          # REQUIRED (hard gate)
      known_sites_chr22/
        dbsnp_138.chr22.vcf.gz(+.tbi)          # REQUIRED both files
        Mills_and_1000G_gold_standard.indels.chr22.vcf.gz(+.tbi)
    results/
      variants/HG002.filtered.vcf.gz[.tbi]     # hard-filtered joint VCF
      annotation/HG002.filtered.annotated.vcf.gz[.tbi]
      reports/multiqc_report.html
      qc/qc_summary.json
    DECISIONS.md

Hidden truth bundle (--hidden-truth-dir):
  HG002_GRCh38_v4.2.1_chr22.vcf.gz(+.tbi)
  HG002_GRCh38_v4.2.1_chr22_confident.bed
  GRCh38_chr22.fa(+.fai, .dict)                # optional, unused by evaluator

Exit codes:
  0 - pass (>= 85 points and all hard gates)
  1 - soft fail (ran but didn't meet threshold)
  2 - hard fail (structural error, missing files, invalid JSON)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("evaluate")

# ----------------------------------------------------------------------------
# Gate thresholds, calibrated from the gold reference run on the synthetic
# HG002 chr22 data (see reference_outputs/benchmarks/happy.summary.csv and
# reference_outputs/qc/qc_summary.json). The gold run produced (hap.py PASS):
#   SNP F1    = 0.9942     INDEL F1       = 0.9977
#   SNP TiTv  = 2.411      mean coverage  = 29.99x
#   dup_rate  = 0.04%      PASS variants  = 42,295 + 7,453 = 49,748
# Numerical gates score against values recomputed by the evaluator, not
# self-reported variant metrics.
# ----------------------------------------------------------------------------
GATES = {
    # Hard gates (binary, fail the whole submission)
    "qc_json_valid": {"hard": True},
    "nextflow_config_present": {"hard": True},
    "vcf_min_variants": {"hard": True, "min_pass_variants": 10_000},

    # Numerical gates (partial credit available) — scored on recomputed values
    "snp_f1": {"weight": 25, "target": 0.990, "tol": 0.003,
               "partial_weight": 15, "partial_target": 0.985},
    "indel_f1": {"weight": 20, "target": 0.990, "tol": 0.005,
                 "partial_weight": 12, "partial_target": 0.980},
    "snp_ti_tv": {"weight": 10, "lo": 2.30, "hi": 2.55},
    "mean_coverage": {"weight": 10, "lo": 28.0, "hi": 32.0},
    "dup_rate": {"weight": 5, "max": 0.12},

    # Structural gates
    "samplesheet_valid": {"weight": 5},
    "known_sites_chr22_only": {"weight": 5},
    # Split into ClinVar (rare coverage) and gnomAD (broad coverage).
    # Provenance: annotated PASS positions must overlap the submitted
    # filtered VCF >= min_filtered_overlap (defends against synthetic
    # CSQ attacks on made-up positions); CLNSIG-bearing annotations
    # must overlap real ClinVar chr22 records >= clinvar_provenance_min.
    "vep_annotation_coverage": {"weight": 10,
                                "clin_min_fraction": 0.10,
                                "gnomad_min_fraction": 0.80,
                                "clin_weight": 4,
                                "gnomad_weight": 6,
                                "min_filtered_overlap": 0.95,
                                "clinvar_provenance_min": 0.50},
    "multiqc_sections": {"weight": 5,
                         "required": ["fastqc", "picard", "samtools", "mosdepth"]},
    "decisions_md_cites_metric": {"weight": 5},
}

REQUIRED_QC_KEYS = [
    "alignment_rate", "dup_rate", "mean_coverage_chr22",
]

PASS_THRESHOLD = 85
CHR22_ALIASES = ("chr22", "22")


@dataclass
class GateResult:
    name: str
    passed: bool
    points: float
    max_points: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "points": self.points,
            "max_points": self.max_points,
            "detail": self.detail,
        }


@dataclass
class Report:
    hard_gates: list[GateResult] = field(default_factory=list)
    scored_gates: list[GateResult] = field(default_factory=list)
    recomputed: dict = field(default_factory=dict)
    total_points: float = 0.0
    max_points: float = 0.0
    hard_pass: bool = False
    pass_: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "hard_pass": self.hard_pass,
            "pass": self.pass_,
            "total_points": self.total_points,
            "max_points": self.max_points,
            "recomputed": self.recomputed,
            "hard_gates": {g.name: g.to_dict() for g in self.hard_gates},
            "scored_gates": {g.name: g.to_dict() for g in self.scored_gates},
            "summary": self.summary,
        }


# ----------------------------------------------------------------------------
# VCF parsing + recomputation against hidden truth
# ----------------------------------------------------------------------------
def _trim_alleles(ref: str, alt: str) -> tuple[str, str, int]:
    """Canonicalize (ref, alt) by trimming common suffix then common prefix.

    Returns the trimmed (ref, alt) plus the positional offset applied from
    prefix-trim. This yields a canonical form that matches across left-
    aligned VCFs (GATK, GIAB v4.2.1, bcftools norm) for the overwhelming
    majority of variants. It is not a full vt-norm but is adequate for the
    chr22 truth-vs-query comparisons used by this scorer.
    """
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]
    offset = 0
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref = ref[1:]
        alt = alt[1:]
        offset += 1
    return ref, alt, offset


def _classify(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1 and ref != alt:
        return "SNP"
    return "INDEL"


def parse_vcf_variants(
    vcf_path: Path,
    *,
    pass_only: bool,
    chrom_filter: tuple[str, ...] = CHR22_ALIASES,
) -> dict:
    """Parse a gzipped VCF into per-type variant sets + Ti/Tv counts.

    Canonical key is (chrom, pos, ref, alt) after prefix/suffix trim.
    Multi-allelic ALT columns are expanded to one variant per ALT.
    """
    snps: set[tuple[str, int, str, str]] = set()
    indels: set[tuple[str, int, str, str]] = set()
    ti = tv = 0
    try:
        fh = gzip.open(vcf_path, "rt")
    except OSError as exc:
        raise RuntimeError(f"cannot open {vcf_path}: {exc}") from exc
    with fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos_s, _id, ref_raw, alt_field, _qual, filt, _info = fields[:8]
            if chrom_filter and chrom not in chrom_filter:
                continue
            if pass_only and filt not in ("PASS", "."):
                continue
            try:
                pos = int(pos_s)
            except ValueError:
                continue
            ref = ref_raw.upper()
            for alt_raw in alt_field.split(","):
                if not alt_raw or alt_raw in (".", "*"):
                    continue
                alt = alt_raw.upper()
                r, a, offset = _trim_alleles(ref, alt)
                key = (chrom, pos + offset, r, a)
                if _classify(r, a) == "SNP":
                    snps.add(key)
                    # Transitions: A<->G (purine), C<->T (pyrimidine); else Tv.
                    if {r, a} in ({"A", "G"}, {"C", "T"}):
                        ti += 1
                    else:
                        tv += 1
                else:
                    indels.add(key)
    return {"snps": snps, "indels": indels, "ti": ti, "tv": tv}


def load_confident_regions(bed_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse a BED into {chrom: sorted [(start, end)]}. Half-open, 0-based."""
    regions: dict[str, list[tuple[int, int]]] = {}
    with bed_path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 3:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            regions.setdefault(parts[0], []).append((start, end))
    for chrom in regions:
        regions[chrom].sort()
    return regions


def _in_region(regions: list[tuple[int, int]], pos: int) -> bool:
    """Binary search for VCF 1-based pos in a sorted BED interval list."""
    if not regions:
        return False
    p0 = pos - 1  # convert to 0-based for BED half-open comparison
    lo, hi = 0, len(regions)
    while lo < hi:
        mid = (lo + hi) // 2
        s, e = regions[mid]
        if p0 < s:
            hi = mid
        elif p0 >= e:
            lo = mid + 1
        else:
            return True
    return False


def _restrict(variants: set, regions: dict[str, list[tuple[int, int]]]) -> set:
    out = set()
    for v in variants:
        chrom = v[0]
        # Accept either chr22 / 22 naming collision
        intervals = regions.get(chrom) or regions.get("chr22" if chrom == "22" else "22")
        if intervals and _in_region(intervals, v[1]):
            out.add(v)
    return out


def recompute_metrics(sub_vcf: Path, hidden_truth_dir: Path) -> dict:
    """Recompute SNP/INDEL precision/recall/F1 + Ti/Tv from submitted VCF."""
    truth_vcf = hidden_truth_dir / "HG002_GRCh38_v4.2.1_chr22.vcf.gz"
    conf_bed = hidden_truth_dir / "HG002_GRCh38_v4.2.1_chr22_confident.bed"
    if not truth_vcf.exists() or not conf_bed.exists():
        raise RuntimeError(
            f"hidden truth bundle incomplete under {hidden_truth_dir}")
    regions = load_confident_regions(conf_bed)

    truth = parse_vcf_variants(truth_vcf, pass_only=False)
    query = parse_vcf_variants(sub_vcf, pass_only=True)

    # PASS counts + Ti/Tv measured on the full chr22 PASS set (unrestricted).
    n_snp_pass = len(query["snps"])
    n_indel_pass = len(query["indels"])
    titv = (query["ti"] / query["tv"]) if query["tv"] > 0 else 0.0

    # F1 computed inside the confident BED on both sides.
    truth_snps = _restrict(truth["snps"], regions)
    truth_indels = _restrict(truth["indels"], regions)
    query_snps = _restrict(query["snps"], regions)
    query_indels = _restrict(query["indels"], regions)

    def prf(truth_set: set, query_set: set) -> tuple[float, float, float]:
        tp = len(truth_set & query_set)
        fn = len(truth_set) - tp
        fp = len(query_set) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    snp_p, snp_r, snp_f1 = prf(truth_snps, query_snps)
    ind_p, ind_r, ind_f1 = prf(truth_indels, query_indels)

    return {
        "snp": {"precision": snp_p, "recall": snp_r, "f1": snp_f1,
                "titv": titv,
                "truth_total": len(truth_snps), "query_total": len(query_snps)},
        "indel": {"precision": ind_p, "recall": ind_r, "f1": ind_f1,
                  "truth_total": len(truth_indels), "query_total": len(query_indels)},
        "n_snp_pass_all_chr22": n_snp_pass,
        "n_indel_pass_all_chr22": n_indel_pass,
    }


# ----------------------------------------------------------------------------
# Hard gates
# ----------------------------------------------------------------------------
def gate_qc_json_valid(sub: Path) -> GateResult:
    qc_path = sub / "results" / "qc" / "qc_summary.json"
    if not qc_path.exists():
        return GateResult("qc_json_valid", False, 0, 0,
                          f"missing {qc_path.relative_to(sub)}")
    try:
        data = json.loads(qc_path.read_text())
    except json.JSONDecodeError as exc:
        return GateResult("qc_json_valid", False, 0, 0, f"invalid JSON: {exc}")
    missing = [k for k in REQUIRED_QC_KEYS if k not in data]
    if missing:
        return GateResult("qc_json_valid", False, 0, 0,
                          f"missing keys: {missing}")
    return GateResult("qc_json_valid", True, 0, 0,
                      f"all {len(REQUIRED_QC_KEYS)} keys present")


def gate_nextflow_config_present(sub: Path) -> GateResult:
    cfg = sub / "pipeline" / "nextflow.config"
    if cfg.exists() and cfg.stat().st_size > 0:
        return GateResult("nextflow_config_present", True, 0, 0,
                          f"{cfg.stat().st_size} B")
    return GateResult("nextflow_config_present", False, 0, 0,
                      f"missing or empty: pipeline/nextflow.config")


def gate_vcf_min_variants(sub: Path) -> GateResult:
    """Count PASS records on chr22 specifically (not any contig)."""
    vcf = sub / "results" / "variants" / "HG002.filtered.vcf.gz"
    if not vcf.exists():
        return GateResult("vcf_min_variants", False, 0, 0, f"missing {vcf.name}")
    chr22_pass = other_pass = 0
    try:
        with gzip.open(vcf, "rt") as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                fields = line.split("\t", 7)
                if len(fields) < 7 or fields[6] != "PASS":
                    continue
                if fields[0] in CHR22_ALIASES:
                    chr22_pass += 1
                else:
                    other_pass += 1
    except (OSError, EOFError) as exc:
        return GateResult("vcf_min_variants", False, 0, 0, f"read error: {exc}")
    min_req = GATES["vcf_min_variants"]["min_pass_variants"]
    ok = chr22_pass >= min_req
    detail = f"{chr22_pass} chr22 PASS (min {min_req}); {other_pass} off-contig PASS"
    return GateResult("vcf_min_variants", ok, 0, 0, detail)


# ----------------------------------------------------------------------------
# Numerical gates — scored on recomputed values
# ----------------------------------------------------------------------------
def _score_f1(name: str, f1: float, cfg: dict) -> GateResult:
    target = cfg["target"] - cfg["tol"]
    if f1 >= target:
        return GateResult(name, True, cfg["weight"], cfg["weight"],
                          f"F1={f1:.4f} >= {target:.4f} (recomputed)")
    if f1 >= cfg["partial_target"]:
        return GateResult(name, False, cfg["partial_weight"], cfg["weight"],
                          f"F1={f1:.4f} partial (>= {cfg['partial_target']}, recomputed)")
    return GateResult(name, False, 0, cfg["weight"],
                      f"F1={f1:.4f} < {cfg['partial_target']} (recomputed)")


def gate_snp_f1(recomputed: dict) -> GateResult:
    return _score_f1("snp_f1", recomputed["snp"]["f1"], GATES["snp_f1"])


def gate_indel_f1(recomputed: dict) -> GateResult:
    return _score_f1("indel_f1", recomputed["indel"]["f1"], GATES["indel_f1"])


def gate_snp_ti_tv(recomputed: dict) -> GateResult:
    cfg = GATES["snp_ti_tv"]
    titv = recomputed["snp"]["titv"]
    ok = cfg["lo"] <= titv <= cfg["hi"]
    detail = (f"Ti/Tv={titv:.3f} in [{cfg['lo']}, {cfg['hi']}] (recomputed)"
              if ok else
              f"Ti/Tv={titv:.3f} out of [{cfg['lo']}, {cfg['hi']}] (recomputed)")
    return GateResult("snp_ti_tv", ok, cfg["weight"] if ok else 0,
                      cfg["weight"], detail)


def gate_mean_coverage(qc: dict) -> GateResult:
    cfg = GATES["mean_coverage"]
    cov = float(qc.get("mean_coverage_chr22", 0))
    ok = cfg["lo"] <= cov <= cfg["hi"]
    return GateResult("mean_coverage", ok, cfg["weight"] if ok else 0,
                      cfg["weight"], f"{cov:.1f}x in [{cfg['lo']}, {cfg['hi']}]")


def gate_dup_rate(qc: dict) -> GateResult:
    cfg = GATES["dup_rate"]
    dup = float(qc.get("dup_rate", 1.0))
    ok = dup <= cfg["max"]
    return GateResult("dup_rate", ok, cfg["weight"] if ok else 0,
                      cfg["weight"], f"{dup*100:.2f}% <= {cfg['max']*100:.1f}%")


# ----------------------------------------------------------------------------
# Structural gates
# ----------------------------------------------------------------------------
def gate_samplesheet_valid(sub: Path) -> GateResult:
    cfg = GATES["samplesheet_valid"]
    ss = sub / "pipeline" / "samplesheet.csv"
    if not ss.exists():
        return GateResult("samplesheet_valid", False, 0, cfg["weight"], "missing")
    try:
        rows = list(csv.DictReader(ss.open()))
    except (csv.Error, OSError) as exc:
        return GateResult("samplesheet_valid", False, 0, cfg["weight"],
                          f"parse error: {exc}")
    if not rows:
        return GateResult("samplesheet_valid", False, 0, cfg["weight"], "empty")
    # nf-core/sarek samplesheet orientation: for HG002 germline on chr22
    # we expect a single sample row. We verify sex/lane and that fastq
    # paths are RELATIVE (so the submission bundle is portable). We also
    # detect R1/R2 swap by filename convention as a quality signal, but
    # only the sex/lane/relative-path checks are blocking.
    problems = []
    for i, row in enumerate(rows):
        if row.get("sex", "").upper() not in ("XY", "MALE", "M"):
            problems.append(f"row {i}: sex={row.get('sex')!r} (expected XY)")
        if not row.get("lane"):
            problems.append(f"row {i}: lane missing")
        paths = {}
        for col in ("fastq_1", "fastq_2"):
            val = (row.get(col) or "").strip()
            if not val:
                problems.append(f"row {i}: {col} empty")
                continue
            if val.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", val):
                problems.append(f"row {i}: {col} is absolute path ({val})")
                continue
            paths[col] = val
        if set(paths) == {"fastq_1", "fastq_2"}:
            f1 = paths["fastq_1"].lower()
            f2 = paths["fastq_2"].lower()
            if "_r2" in f1 and "_r1" in f2:
                problems.append(
                    f"row {i}: fastq_1/fastq_2 appear swapped "
                    f"({paths['fastq_1']} / {paths['fastq_2']})")
    ok = not problems
    return GateResult("samplesheet_valid", ok, cfg["weight"] if ok else 0,
                      cfg["weight"], "valid" if ok else "; ".join(problems))


def gate_known_sites_chr22_only(sub: Path) -> GateResult:
    """Require both chr22-subsetted known-sites VCFs + their .tbi files.

    Checks: (1) chr22 in the ##contig header, (2) all sampled records
    are chr22 records, (3) file size under the 50 MB full-genome ceiling.
    """
    cfg = GATES["known_sites_chr22_only"]
    ks_dir = sub / "pipeline" / "known_sites_chr22"
    if not ks_dir.exists():
        return GateResult("known_sites_chr22_only", False, 0, cfg["weight"],
                          f"missing directory {ks_dir}")
    required = {
        "dbsnp_vcf": list(ks_dir.glob("*dbsnp*.vcf.gz")),
        "dbsnp_tbi": list(ks_dir.glob("*dbsnp*.vcf.gz.tbi")),
        "mills_vcf": list(ks_dir.glob("*[Mm]ills*.vcf.gz")),
        "mills_tbi": list(ks_dir.glob("*[Mm]ills*.vcf.gz.tbi")),
    }
    for key, files in required.items():
        # Strip `.tbi` matches accidentally landing in the `.vcf.gz` glob
        if key.endswith("_vcf"):
            files[:] = [f for f in files if not f.name.endswith(".tbi")]
            required[key] = files
    missing = [k for k, v in required.items() if not v]
    if missing:
        return GateResult("known_sites_chr22_only", False, 0, cfg["weight"],
                          f"missing: {missing}")

    problems = []
    for label, files in required.items():
        if not label.endswith("_vcf"):
            continue
        vcf = files[0]
        try:
            header_has_chr22 = False
            non_chr22_records = 0
            chr22_records = 0
            with gzip.open(vcf, "rt") as fh:
                for line in fh:
                    if line.startswith("##contig="):
                        m = re.search(r"ID=([^,>]+)", line)
                        if m and m.group(1) in CHR22_ALIASES:
                            header_has_chr22 = True
                        continue
                    if line.startswith("#"):
                        continue
                    contig = line.split("\t", 1)[0]
                    if contig in CHR22_ALIASES:
                        chr22_records += 1
                    else:
                        non_chr22_records += 1
                    if chr22_records + non_chr22_records >= 5_000:
                        break
        except (OSError, EOFError) as exc:
            problems.append(f"{vcf.name} read error: {exc}")
            continue
        if not header_has_chr22:
            problems.append(f"{vcf.name} missing chr22 contig in header")
        if non_chr22_records > 0:
            problems.append(f"{vcf.name} has {non_chr22_records} non-chr22 records")
        if vcf.stat().st_size > 50_000_000:
            problems.append(f"{vcf.name} too large ({vcf.stat().st_size} B)")
    ok = not problems
    return GateResult("known_sites_chr22_only", ok,
                      cfg["weight"] if ok else 0, cfg["weight"],
                      "chr22-only + indexed" if ok else "; ".join(problems))


def _chrom_pos_set(vcf_path: Path) -> set[tuple[str, int]]:
    """Return the set of (chrom, pos) records in a gzipped VCF (all filters)."""
    out: set[tuple[str, int]] = set()
    try:
        with gzip.open(vcf_path, "rt") as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                parts = line.split("\t", 3)
                if len(parts) < 2:
                    continue
                try:
                    out.add((parts[0], int(parts[1])))
                except ValueError:
                    continue
    except (OSError, EOFError):
        return out
    return out


def gate_vep_annotation_coverage(sub: Path,
                                 hidden_truth_dir: Path) -> GateResult:
    """Score VEP annotation coverage AND provenance.

    Provenance checks (beyond keyword presence in CSQ):
    - PASS positions in the annotated VCF must be a subset of PASS
      positions in the submitted filtered VCF (within a small slack).
      A fabricator who invents positions fails immediately.
    - Annotated records with non-empty ClinVar_CLNSIG must overlap real
      ClinVar chr22 records at a high rate; synthetic CLNSIG strings on
      made-up positions cannot satisfy this.
    """
    cfg = GATES["vep_annotation_coverage"]
    vcf = sub / "results" / "annotation" / "HG002.filtered.annotated.vcf.gz"
    filtered_vcf = sub / "results" / "variants" / "HG002.filtered.vcf.gz"
    if not vcf.exists():
        return GateResult("vep_annotation_coverage", False, 0, cfg["weight"],
                          f"missing {vcf.name}")

    # Provenance: annotated VCF PASS positions must derive from the
    # filtered VCF. Compute both sets.
    filtered_positions: set[tuple[str, int]] = set()
    if filtered_vcf.exists():
        try:
            with gzip.open(filtered_vcf, "rt") as fh:
                for line in fh:
                    if not line or line[0] == "#":
                        continue
                    f = line.split("\t", 8)
                    if len(f) >= 7 and f[6] == "PASS":
                        try:
                            filtered_positions.add((f[0], int(f[1])))
                        except ValueError:
                            continue
        except (OSError, EOFError):
            pass

    annotated_positions: set[tuple[str, int]] = set()
    clin_positions: set[tuple[str, int]] = set()
    total = with_clin = with_gnomad = 0
    try:
        with gzip.open(vcf, "rt") as fh:
            csq_header_idx: dict[str, int] | None = None
            for line in fh:
                if line.startswith("#"):
                    if "ID=CSQ" in line and "Format:" in line:
                        m = re.search(r"Format:\s*([^\"]+)", line)
                        if m:
                            fields = m.group(1).strip("'").split("|")
                            csq_header_idx = {f.strip(): i for i, f in enumerate(fields)}
                    continue
                parts = line.split("\t", 7)
                if len(parts) < 8 or parts[6] != "PASS" or csq_header_idx is None:
                    continue
                try:
                    ann_key = (parts[0], int(parts[1]))
                except ValueError:
                    continue
                annotated_positions.add(ann_key)
                info = parts[7]
                m = re.search(r"CSQ=([^;]+)", info)
                if not m:
                    continue
                annotations = m.group(1).split(",")
                clin_idx = (csq_header_idx.get("CLIN_SIG")
                            or csq_header_idx.get("ClinVar_CLNSIG")
                            or csq_header_idx.get("CLNSIG"))
                gnomad_idx = (csq_header_idx.get("gnomAD_AF")
                              or csq_header_idx.get("gnomADe_AF")
                              or csq_header_idx.get("gnomADg_AF"))
                for ann in annotations:
                    cols = ann.split("|")
                    impact_idx = csq_header_idx.get("IMPACT")
                    if impact_idx is None or impact_idx >= len(cols):
                        continue
                    if cols[impact_idx] not in ("HIGH", "MODERATE"):
                        continue
                    total += 1
                    if clin_idx is not None and clin_idx < len(cols) and cols[clin_idx]:
                        with_clin += 1
                        clin_positions.add(ann_key)
                    if gnomad_idx is not None and gnomad_idx < len(cols) and cols[gnomad_idx]:
                        with_gnomad += 1
                    break
    except (OSError, EOFError) as exc:
        return GateResult("vep_annotation_coverage", False, 0, cfg["weight"],
                          f"read error: {exc}")
    if total == 0:
        return GateResult("vep_annotation_coverage", False, 0, cfg["weight"],
                          "no HIGH/MODERATE coding variants found")

    provenance_problems: list[str] = []

    if filtered_positions and annotated_positions:
        overlap = annotated_positions & filtered_positions
        overlap_frac = len(overlap) / max(len(annotated_positions), 1)
        if overlap_frac < cfg.get("min_filtered_overlap", 0.95):
            provenance_problems.append(
                f"only {overlap_frac*100:.1f}% of annotated PASS positions "
                f"trace back to the filtered VCF "
                f"({len(overlap)}/{len(annotated_positions)})")
    elif not filtered_positions:
        provenance_problems.append(
            "filtered VCF unavailable for provenance cross-check")

    # ClinVar provenance: sample the first N CLNSIG-bearing positions and
    # verify they overlap a real ClinVar chr22 record. An honest VEP run
    # with --custom ClinVar yields CLNSIG only where ClinVar has data.
    clinvar_ref = hidden_truth_dir / "clinvar.chr22.vcf.gz"
    if clinvar_ref.exists() and clin_positions:
        real_clin = _chrom_pos_set(clinvar_ref)
        # Normalize naming differences: real ClinVar is contig-naming dependent.
        chrom_keys = {k[0] for k in real_clin}
        if chrom_keys and "chr22" not in chrom_keys and "22" in chrom_keys:
            clin_positions = {("22", p) for (_c, p) in clin_positions}
        clin_match = len(clin_positions & real_clin)
        clin_match_frac = clin_match / max(len(clin_positions), 1)
        if clin_match_frac < cfg.get("clinvar_provenance_min", 0.50):
            provenance_problems.append(
                f"ClinVar_CLNSIG provenance failure: only "
                f"{clin_match*100/max(len(clin_positions),1):.1f}% of "
                f"CLNSIG-bearing positions exist in the hidden ClinVar chr22 "
                f"({clin_match}/{len(clin_positions)})")

    if provenance_problems:
        return GateResult(
            "vep_annotation_coverage", False, 0, cfg["weight"],
            "provenance failed: " + "; ".join(provenance_problems))

    clin_frac = with_clin / total
    gnomad_frac = with_gnomad / total
    clin_ok = clin_frac >= cfg["clin_min_fraction"]
    gnomad_ok = gnomad_frac >= cfg["gnomad_min_fraction"]
    points = ((cfg["clin_weight"] if clin_ok else 0)
              + (cfg["gnomad_weight"] if gnomad_ok else 0))
    ok = clin_ok and gnomad_ok
    return GateResult("vep_annotation_coverage", ok, points, cfg["weight"],
                      f"ClinVar={clin_frac*100:.1f}% ({with_clin}/{total}), "
                      f"gnomAD={gnomad_frac*100:.1f}% ({with_gnomad}/{total})")


def gate_multiqc_sections(sub: Path) -> GateResult:
    """Require real MultiQC data files, not just keyword-stuffed HTML.

    MultiQC always emits `multiqc_data/multiqc_general_stats.txt` (a TSV
    with one header row of `<module>-<metric>` columns and one data row
    per sample) and `multiqc_data/multiqc_software_versions.txt` (a
    module × tool × version matrix). Both are load-bearing evidence
    that MultiQC actually ran; neither can be trivially fabricated with
    plausible numbers without knowing the module-specific schema.
    """
    cfg = GATES["multiqc_sections"]
    reports = sub / "results" / "reports"
    html = reports / "multiqc_report.html"
    data_dir = reports / "multiqc_data"
    stats = data_dir / "multiqc_general_stats.txt"
    versions = data_dir / "multiqc_software_versions.txt"

    problems: list[str] = []
    if not html.exists():
        problems.append("missing multiqc_report.html")
    if not data_dir.is_dir():
        problems.append("missing multiqc_data/ directory")
    if not stats.exists():
        problems.append("missing multiqc_data/multiqc_general_stats.txt")
    if not versions.exists():
        problems.append("missing multiqc_data/multiqc_software_versions.txt")

    if problems:
        return GateResult("multiqc_sections", False, 0, cfg["weight"],
                          "; ".join(problems))

    required = [m.lower() for m in cfg["required"]]
    try:
        header = stats.read_text(errors="replace").splitlines()[:1]
        rows = stats.read_text(errors="replace").splitlines()[1:]
    except OSError as exc:
        return GateResult("multiqc_sections", False, 0, cfg["weight"],
                          f"read error on general_stats: {exc}")
    if not header or not rows:
        return GateResult("multiqc_sections", False, 0, cfg["weight"],
                          "multiqc_general_stats.txt empty")

    header_lc = header[0].lower()
    # Each required module must appear as a column-name prefix in the
    # general stats header. These prefixes are stable across MultiQC 1.x.
    prefix_map = {
        "fastqc": ("fastqc_raw", "fastqc-status-check", "fastqc"),
        "picard": ("picard", "gatk4_markduplicates"),
        "samtools": ("samtools_flagstat", "samtools_stats", "samtools"),
        "mosdepth": ("mosdepth",),
    }
    missing = []
    for mod in required:
        prefixes = prefix_map.get(mod, (mod,))
        if not any(p in header_lc for p in prefixes):
            missing.append(mod)

    # At least one non-empty data row with a real sample name
    real_rows = [r for r in rows if r.strip() and not r.startswith("#")]
    if not real_rows:
        missing.append("no sample rows in general_stats")

    # Versions file must declare at least the tool names we scored
    try:
        versions_lc = versions.read_text(errors="replace").lower()
    except OSError as exc:
        return GateResult("multiqc_sections", False, 0, cfg["weight"],
                          f"read error on software_versions: {exc}")
    for tool in ("gatk4", "samtools", "fastqc", "mosdepth"):
        if tool not in versions_lc:
            missing.append(f"software_versions missing '{tool}'")

    ok = not missing
    return GateResult("multiqc_sections", ok, cfg["weight"] if ok else 0,
                      cfg["weight"],
                      f"{len(real_rows)} sample rows; modules verified"
                      if ok else f"missing: {missing}")


def gate_decisions_md(sub: Path) -> GateResult:
    """Require a number adjacent to a filter name, not just co-occurrence.

    Boilerplate DECISIONS.md can trivially mention QD/FS/MQ as keywords
    and have unrelated numbers elsewhere. We require at least one
    (filter, number) citation within 50 characters of each other — the
    shape real engineers use ("QD < 2.0", "FS > 60.0", etc.).
    """
    cfg = GATES["decisions_md_cites_metric"]
    md = sub / "DECISIONS.md"
    if not md.exists():
        return GateResult("decisions_md_cites_metric", False, 0, cfg["weight"],
                          "missing DECISIONS.md")
    text = md.read_text(errors="replace")
    filter_fields = ["QD", "FS", "MQ", "MQRankSum", "ReadPosRankSum", "SOR"]
    # Adjacency: <filter> <= 50 chars <number>
    pattern = (
        r"(?P<filter>" + "|".join(re.escape(f) for f in filter_fields) + r")"
        r"[^A-Za-z0-9]{0,50}?"
        r"(?P<value>\d+(?:\.\d+)?)"
    )
    matches = list(re.finditer(pattern, text))
    filters_cited = {m.group("filter") for m in matches}
    ok = len(matches) >= 2 and len(filters_cited) >= 2
    return GateResult(
        "decisions_md_cites_metric", ok, cfg["weight"] if ok else 0,
        cfg["weight"],
        f"{len(filters_cited)} distinct filters with adjacent numbers "
        f"({len(matches)} citations total)")


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def evaluate(sub: Path, hidden_truth_dir: Path) -> Report:
    rep = Report()

    # Hard gates (order matters: qc_json_valid must pass before we load qc)
    hard_gates = [
        gate_qc_json_valid(sub),
        gate_nextflow_config_present(sub),
        gate_vcf_min_variants(sub),
    ]
    rep.hard_gates = hard_gates
    rep.hard_pass = all(g.passed for g in hard_gates)

    if not rep.hard_pass:
        rep.summary = "Hard gate failure: " + "; ".join(
            g.detail for g in hard_gates if not g.passed
        )
        rep.max_points = sum(
            GATES[k].get("weight", 0) for k in GATES
            if not GATES[k].get("hard") and "weight" in GATES[k]
        )
        return rep

    # Recompute metrics from the submitted VCF against the hidden truth.
    sub_vcf = sub / "results" / "variants" / "HG002.filtered.vcf.gz"
    try:
        recomputed = recompute_metrics(sub_vcf, hidden_truth_dir)
    except RuntimeError as exc:
        rep.hard_pass = False
        rep.hard_gates.append(GateResult(
            "hidden_truth_recompute", False, 0, 0, str(exc)))
        rep.summary = f"Hidden-truth recomputation failed: {exc}"
        return rep
    rep.recomputed = recomputed

    qc = json.loads((sub / "results" / "qc" / "qc_summary.json").read_text())

    scored = [
        gate_snp_f1(recomputed),
        gate_indel_f1(recomputed),
        gate_snp_ti_tv(recomputed),
        gate_mean_coverage(qc),
        gate_dup_rate(qc),
        gate_samplesheet_valid(sub),
        gate_known_sites_chr22_only(sub),
        gate_vep_annotation_coverage(sub, hidden_truth_dir),
        gate_multiqc_sections(sub),
        gate_decisions_md(sub),
    ]

    rep.scored_gates = scored
    rep.total_points = sum(g.points for g in scored)
    rep.max_points = sum(g.max_points for g in scored)
    rep.pass_ = rep.hard_pass and rep.total_points >= PASS_THRESHOLD
    rep.summary = (
        f"{rep.total_points:.1f}/{rep.max_points:.0f} points; "
        f"{'PASS' if rep.pass_ else 'FAIL'}"
    )
    return rep


def print_human_report(rep: Report) -> None:
    print("=" * 72)
    print("HARD GATES")
    print("=" * 72)
    for g in rep.hard_gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"  [{status}] {g.name:<34} {g.detail}")
    if not rep.hard_pass:
        print("\nHard gate failure — scoring aborted.\n")
        return

    if rep.recomputed:
        print("\n" + "=" * 72)
        print("RECOMPUTED METRICS (from submitted VCF vs hidden truth)")
        print("=" * 72)
        snp = rep.recomputed["snp"]
        ind = rep.recomputed["indel"]
        print(f"  SNP    F1={snp['f1']:.4f}  P={snp['precision']:.4f}  "
              f"R={snp['recall']:.4f}  Ti/Tv={snp['titv']:.3f}  "
              f"truth={snp['truth_total']} query={snp['query_total']}")
        print(f"  INDEL  F1={ind['f1']:.4f}  P={ind['precision']:.4f}  "
              f"R={ind['recall']:.4f}  "
              f"truth={ind['truth_total']} query={ind['query_total']}")
        print(f"  chr22 PASS: SNP={rep.recomputed['n_snp_pass_all_chr22']}  "
              f"INDEL={rep.recomputed['n_indel_pass_all_chr22']}")

    print("\n" + "=" * 72)
    print("SCORED GATES")
    print("=" * 72)
    for g in rep.scored_gates:
        status = "PASS" if g.passed else "----"
        print(f"  [{status}] {g.name:<32} "
              f"{g.points:5.1f} / {g.max_points:<3.0f}  {g.detail}")

    print("\n" + "=" * 72)
    verdict = "PASS" if rep.pass_ else "FAIL"
    print(f"VERDICT: {verdict}  ({rep.total_points:.1f} / {rep.max_points:.0f}"
          f"; pass threshold = {PASS_THRESHOLD})")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, required=True,
                    help="Agent submission directory root")
    ap.add_argument("--hidden-truth-dir", type=Path, required=True,
                    help="Evaluator-only directory with chr22 truth VCF + "
                         "confident BED (+ optional reference FASTA)")
    ap.add_argument("--out", type=Path,
                    help="Write machine-readable JSON report to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    sub = args.submission.resolve()
    hidden = args.hidden_truth_dir.resolve()
    if not sub.exists():
        print(f"ERROR: submission directory not found: {sub}", file=sys.stderr)
        return 2
    if not hidden.exists():
        print(f"ERROR: hidden truth directory not found: {hidden}", file=sys.stderr)
        return 2

    rep = evaluate(sub, hidden)

    if not args.quiet:
        print_human_report(rep)

    if args.out:
        args.out.write_text(json.dumps(rep.to_dict(), indent=2))

    if not rep.hard_pass:
        return 2
    return 0 if rep.pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
