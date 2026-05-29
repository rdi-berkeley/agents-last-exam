"""Local scoring helpers for the VMD protein solvation task."""

from dataclasses import dataclass
from typing import Any

WATER_RESIDUES = {"HOH", "WAT", "TIP3", "SOL", "TIP3W", "TIP"}
EXPECTED_LOG_LINE = "Solvate completed successfully."
MIN_PADDING_ANGSTROM = 4.9


@dataclass
class SolvateScoreResult:
    score: float
    passed: bool
    reason: str
    final_log_line: str | None = None
    paddings: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "final_log_line": self.final_log_line,
            "paddings": self.paddings,
        }


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _last_nonblank_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return None


def _parse_paddings(pdb_text: str) -> dict[str, float]:
    all_coords: list[tuple[float, float, float]] = []
    solute_coords: list[tuple[float, float, float]] = []

    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue

        try:
            resname = line[17:20].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except Exception as exc:  # pragma: no cover - defensive parser parity
            raise ValueError(f"failed to parse PDB coordinate line: {line!r}") from exc

        coords = (x, y, z)
        all_coords.append(coords)
        if resname not in WATER_RESIDUES:
            solute_coords.append(coords)

    if not all_coords or not solute_coords:
        raise ValueError("could not determine system and solute bounding boxes")

    xmin_all = min(coord[0] for coord in all_coords)
    xmax_all = max(coord[0] for coord in all_coords)
    ymin_all = min(coord[1] for coord in all_coords)
    ymax_all = max(coord[1] for coord in all_coords)
    zmin_all = min(coord[2] for coord in all_coords)
    zmax_all = max(coord[2] for coord in all_coords)

    xmin_solute = min(coord[0] for coord in solute_coords)
    xmax_solute = max(coord[0] for coord in solute_coords)
    ymin_solute = min(coord[1] for coord in solute_coords)
    ymax_solute = max(coord[1] for coord in solute_coords)
    zmin_solute = min(coord[2] for coord in solute_coords)
    zmax_solute = max(coord[2] for coord in solute_coords)

    return {
        "x_min": xmin_solute - xmin_all,
        "x_max": xmax_all - xmax_solute,
        "y_min": ymin_solute - ymin_all,
        "y_max": ymax_all - ymax_solute,
        "z_min": zmin_solute - zmin_all,
        "z_max": zmax_all - zmax_solute,
    }


def score_solvate_outputs(
    *,
    pdb_bytes: bytes | None,
    psf_exists: bool,
    log_bytes: bytes | None,
) -> SolvateScoreResult:
    if not psf_exists or pdb_bytes is None or log_bytes is None:
        return SolvateScoreResult(
            score=0.0,
            passed=False,
            reason="missing_required_output",
        )

    log_text = _decode_text(log_bytes)
    final_log_line = _last_nonblank_line(log_text)
    if final_log_line != EXPECTED_LOG_LINE:
        return SolvateScoreResult(
            score=0.0,
            passed=False,
            reason="unexpected_final_log_line",
            final_log_line=final_log_line,
        )

    try:
        paddings = _parse_paddings(_decode_text(pdb_bytes))
    except Exception as exc:
        return SolvateScoreResult(
            score=0.0,
            passed=False,
            reason=f"pdb_parse_error: {exc}",
            final_log_line=final_log_line,
        )

    if any(value < MIN_PADDING_ANGSTROM for value in paddings.values()):
        return SolvateScoreResult(
            score=0.0,
            passed=False,
            reason="padding_below_threshold",
            final_log_line=final_log_line,
            paddings=paddings,
        )

    return SolvateScoreResult(
        score=1.0,
        passed=True,
        reason="ok",
        final_log_line=final_log_line,
        paddings=paddings,
    )
