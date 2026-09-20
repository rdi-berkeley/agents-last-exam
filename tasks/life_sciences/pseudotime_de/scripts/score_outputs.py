"""Score case-sensitive gene sets with both precision and recall >= 0.80."""

import csv
import io
from dataclasses import asdict, dataclass

REQUIRED_FILES = ["de_genes.csv"]
PRECISION_THRESHOLD = 0.80
RECALL_THRESHOLD = 0.80


class ReferenceValidationError(RuntimeError):
    """The evaluator-controlled reference is not a valid nonempty gene list."""


@dataclass
class ScoreReport:
    score: float
    precision: float
    recall: float
    agent_gene_count: int
    reference_gene_count: int
    intersection_count: int
    false_positive_count: int
    false_negative_count: int
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_gene_set(raw: bytes, expected_column: str) -> set[str]:
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next((row for row in reader if row), None)
    if header is None:
        raise ValueError("CSV has no header row")
    header = [name.strip().lower() for name in header]
    indexed_reference = expected_column == "x" and header == ["", "x"]
    if header != [expected_column] and not indexed_reference:
        raise ValueError(f"Expected only '{expected_column}' column; got {header}")

    genes: set[str] = set()
    row_indices: set[str] = set()
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"Wrong column count at CSV line {reader.line_num}")
        if indexed_reference:
            row_index = row[0].strip()
            if (
                not row_index.isascii()
                or not row_index.isdecimal()
                or int(row_index) < 1
                or row_index in row_indices
            ):
                raise ValueError(f"Invalid reference row index at CSV line {reader.line_num}")
            row_indices.add(row_index)
        gene = row[-1].strip()
        if (
            not gene
            or not gene.isprintable()
            or any(character.isspace() or character in ',"' for character in gene)
        ):
            raise ValueError(f"Invalid gene symbol at CSV line {reader.line_num}")
        genes.add(gene)
    return genes


def score_submission(
    output_payloads: dict[str, bytes],
    reference_csv: bytes,
) -> ScoreReport:
    try:
        reference_genes = _parse_gene_set(reference_csv, "x")
        if not reference_genes:
            raise ValueError("Reference contains zero genes")
    except (ValueError, csv.Error) as exc:
        raise ReferenceValidationError(f"Invalid evaluator-controlled reference: {exc}") from exc

    agent_genes: set[str] = set()
    error = ""
    candidate = output_payloads.get("de_genes.csv")
    if candidate is None:
        error = "de_genes.csv missing from output payloads"
    else:
        try:
            agent_genes = _parse_gene_set(candidate, "gene")
        except (ValueError, csv.Error) as exc:
            error = f"Failed to parse agent output: {exc}"
        if not error and not agent_genes:
            error = "Agent output contains zero genes"

    intersection_count = len(agent_genes & reference_genes)
    precision = intersection_count / len(agent_genes) if agent_genes else 0.0
    recall = intersection_count / len(reference_genes)
    score = float(not error and precision >= PRECISION_THRESHOLD and recall >= RECALL_THRESHOLD)
    return ScoreReport(
        score=score,
        precision=precision,
        recall=recall,
        agent_gene_count=len(agent_genes),
        reference_gene_count=len(reference_genes),
        intersection_count=intersection_count,
        false_positive_count=len(agent_genes) - intersection_count,
        false_negative_count=len(reference_genes) - intersection_count,
        error=error,
    )
