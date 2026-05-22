"""Local scorer for pseudotime_de: gene-set overlap against gold standard."""

import csv
import io
from dataclasses import dataclass

REQUIRED_FILES = ["de_genes.csv"]

OVERLAP_THRESHOLD = 0.80


@dataclass
class ScoreReport:
    score: float
    overlap: float
    agent_gene_count: int
    reference_gene_count: int
    intersection_count: int
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "overlap": self.overlap,
            "agent_gene_count": self.agent_gene_count,
            "reference_gene_count": self.reference_gene_count,
            "intersection_count": self.intersection_count,
            "error": self.error,
        }


def _parse_gene_set(raw: bytes, expected_column: str) -> set[str]:
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    col = None
    for name in reader.fieldnames:
        if name.strip().lower() == expected_column.lower():
            col = name
            break
    if col is None:
        if len(reader.fieldnames) == 1:
            col = reader.fieldnames[0]
        else:
            raise ValueError(
                f"No '{expected_column}' column found; columns: {reader.fieldnames}"
            )
    genes: set[str] = set()
    for row in reader:
        val = row[col]
        if val is not None:
            val = val.strip()
            if val:
                genes.add(val)
    return genes


def score_submission(
    output_payloads: dict[str, bytes],
    reference_csv: bytes,
) -> ScoreReport:
    de_genes_raw = output_payloads.get("de_genes.csv")
    if de_genes_raw is None:
        return ScoreReport(
            score=0.0, overlap=0.0, agent_gene_count=0,
            reference_gene_count=0, intersection_count=0,
            error="de_genes.csv missing from output payloads",
        )

    try:
        agent_genes = _parse_gene_set(de_genes_raw, "gene")
    except Exception as exc:
        return ScoreReport(
            score=0.0, overlap=0.0, agent_gene_count=0,
            reference_gene_count=0, intersection_count=0,
            error=f"Failed to parse agent output: {exc}",
        )

    if not agent_genes:
        return ScoreReport(
            score=0.0, overlap=0.0, agent_gene_count=0,
            reference_gene_count=0, intersection_count=0,
            error="Agent output contains zero genes",
        )

    try:
        ref_genes = _parse_gene_set(reference_csv, "x")
    except Exception as exc:
        return ScoreReport(
            score=0.0, overlap=0.0, agent_gene_count=0,
            reference_gene_count=0, intersection_count=0,
            error=f"Failed to parse reference: {exc}",
        )

    intersection = agent_genes & ref_genes
    overlap = len(intersection) / len(ref_genes) if ref_genes else 0.0
    score = 1.0 if overlap >= OVERLAP_THRESHOLD else 0.0

    return ScoreReport(
        score=score,
        overlap=round(overlap, 6),
        agent_gene_count=len(agent_genes),
        reference_gene_count=len(ref_genes),
        intersection_count=len(intersection),
    )
