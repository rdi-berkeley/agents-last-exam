"""Evaluator-side VCF scoring for WGS_Variant_Calling."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

MAX_COMPRESSED_VCF_BYTES = 10_000_000
MAX_UNCOMPRESSED_VCF_BYTES = 100_000_000


@dataclass(frozen=True)
class TypeMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class ScoreReport:
    snp: TypeMetrics
    indel: TypeMetrics
    query_variant_count: int
    ignored_query_variant_count: int


def validate_tabix_index(payload: bytes) -> bool:
    """Return whether payload is a readable tabix index."""
    if not payload or len(payload) > 1_000_000:
        return False
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
            return handle.read(4) == b"TBI\x01"
    except (EOFError, OSError):
        return False


def validate_summary_csv(payload: bytes) -> bool:
    """Validate the required informational summary without trusting its values."""
    try:
        lines = [line.strip() for line in payload.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        return False
    if not lines or lines[0] != "Type,Precision,Sensitivity,F_measure":
        return False
    row_types = {line.split(",", 1)[0].strip().upper() for line in lines[1:]}
    return {"SNP", "INDEL"}.issubset(row_types)


def _read_fasta(path: Path) -> tuple[str, str]:
    contig = ""
    sequence_parts: list[str] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if contig:
                    raise ValueError("evaluator reference must contain exactly one contig")
                contig = line[1:].split()[0]
                continue
            sequence_parts.append(line.upper())
    if not contig or not sequence_parts:
        raise ValueError(f"invalid evaluator reference FASTA: {path}")
    return contig, "".join(sequence_parts)


def _read_bed(path: Path, expected_contig: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3 or fields[0] != expected_contig:
                continue
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError(f"invalid confident interval: {line.rstrip()}")
            intervals.append((start, end))
    intervals.sort()
    if not intervals:
        raise ValueError(f"no confident intervals found in {path}")
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("confident intervals must not overlap")
    return intervals


def _decode_vcf(payload: bytes) -> str:
    if len(payload) > MAX_COMPRESSED_VCF_BYTES:
        raise ValueError("compressed VCF exceeds evaluator size limit")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
            decoded = handle.read(MAX_UNCOMPRESSED_VCF_BYTES + 1)
    except (EOFError, OSError, UnicodeDecodeError) as exc:
        raise ValueError("submitted VCF is not readable gzip-compressed UTF-8") from exc
    if len(decoded) > MAX_UNCOMPRESSED_VCF_BYTES:
        raise ValueError("uncompressed VCF exceeds evaluator size limit")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("submitted VCF is not valid UTF-8") from exc


def _interval_index(
    intervals: list[tuple[int, int]], starts: list[int], position_1based: int
) -> int | None:
    position_0based = position_1based - 1
    index = bisect.bisect_right(starts, position_0based) - 1
    if index < 0:
        return None
    start, end = intervals[index]
    return index if start <= position_0based < end else None


def _parse_variant_sets(
    vcf_text: str,
    *,
    contig: str,
    reference: str,
    intervals: list[tuple[int, int]],
    pass_only: bool,
) -> tuple[dict[str, set[tuple]], int, int]:
    """Parse alleles and canonicalize each by its confident-interval haplotype."""
    variants: dict[str, set[tuple]] = {"snp": set(), "indel": set()}
    starts = [start for start, _end in intervals]
    parsed_count = 0
    ignored_count = 0
    saw_header = False

    for line in vcf_text.splitlines():
        if line.startswith("#CHROM"):
            saw_header = True
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 8:
            raise ValueError("submitted VCF contains a record with fewer than 8 fields")
        chrom, position_text, _identifier, ref, alt_field, _qual, filter_value = fields[:7]
        if chrom != contig:
            ignored_count += 1
            continue
        if pass_only and filter_value not in {"PASS", "."}:
            ignored_count += 1
            continue
        try:
            position = int(position_text)
        except ValueError as exc:
            raise ValueError(f"invalid VCF position: {position_text!r}") from exc
        interval_index = _interval_index(intervals, starts, position)
        if interval_index is None:
            ignored_count += 1
            continue
        interval_start, interval_end = intervals[interval_index]
        ref = ref.upper()
        ref_start = position - 1
        ref_end = ref_start + len(ref)
        if ref_start < interval_start or ref_end > interval_end:
            ignored_count += 1
            continue
        if reference[ref_start:ref_end] != ref:
            raise ValueError(f"VCF REF does not match evaluator reference at {chrom}:{position}")

        for alt in alt_field.upper().split(","):
            if not alt or alt in {".", "*"} or alt.startswith("<") or "[" in alt or "]" in alt:
                ignored_count += 1
                continue
            if any(base not in "ACGTN" for base in ref + alt):
                ignored_count += 1
                continue
            alternate_haplotype = (
                reference[interval_start:ref_start] + alt + reference[ref_end:interval_end]
            )
            digest = hashlib.sha256(alternate_haplotype.encode("ascii")).digest()
            variant_type = "snp" if len(ref) == len(alt) == 1 and ref != alt else "indel"
            variants[variant_type].add((interval_start, interval_end, digest))
            parsed_count += 1

    if not saw_header:
        raise ValueError("submitted VCF is missing the #CHROM header")
    return variants, parsed_count, ignored_count


def _metrics(truth: set[tuple], query: set[tuple]) -> TypeMetrics:
    true_positive = len(truth & query)
    false_positive = len(query - truth)
    false_negative = len(truth - query)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TypeMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def score_vcf_bytes(query_payload: bytes, evaluator_data_dir: Path) -> ScoreReport:
    """Score a submitted VCF against evaluator-local truth."""
    reference_path = evaluator_data_dir / "chr17.fa"
    truth_path = evaluator_data_dir / "truth_chr17.vcf.gz"
    confident_bed_path = evaluator_data_dir / "eval_region_confident.bed"
    missing = [
        str(path) for path in (reference_path, truth_path, confident_bed_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"evaluator data is incomplete: {missing}")

    contig, reference = _read_fasta(reference_path)
    intervals = _read_bed(confident_bed_path, contig)
    truth_text = _decode_vcf(truth_path.read_bytes())
    query_text = _decode_vcf(query_payload)
    truth, _truth_count, _truth_ignored = _parse_variant_sets(
        truth_text,
        contig=contig,
        reference=reference,
        intervals=intervals,
        pass_only=False,
    )
    query, query_count, query_ignored = _parse_variant_sets(
        query_text,
        contig=contig,
        reference=reference,
        intervals=intervals,
        pass_only=True,
    )
    return ScoreReport(
        snp=_metrics(truth["snp"], query["snp"]),
        indel=_metrics(truth["indel"], query["indel"]),
        query_variant_count=query_count,
        ignored_query_variant_count=query_ignored,
    )
