"""Layout-agnostic workbook scorer for business_finance/equity_research_summary.

Finds labels and section headers by scanning the worksheet instead of
requiring fixed cell coordinates. This allows the agent to use any layout.
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation
from io import BytesIO
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.cell import Cell
from openpyxl.formula import Tokenizer
from openpyxl.styles.colors import COLOR_INDEX
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.writer.theme import theme_xml

if __package__:
    from .formula_evaluator import FormulaCalculationError, FormulaEvaluationError, FormulaEvaluator
else:
    from formula_evaluator import FormulaCalculationError, FormulaEvaluationError, FormulaEvaluator


@dataclass
class WorkbookScoreResult:
    score: float
    passed: bool
    reasons: list[str]


FORMULA_RULES = {
    "Gross Profit": ("difference", "Total Revenue", "Cost of Revenue"),
    "Gross Margin %": ("ratio", "Gross Profit", "Total Revenue"),
    "Operating Income": ("difference", "Gross Profit", "Operating Expenses"),
    "Operating Margin %": ("ratio", "Operating Income", "Total Revenue"),
    "Net Margin %": ("ratio", "Net Income", "Total Revenue"),
    "Total Equity": ("difference", "Total Assets", "Total Liabilities"),
    "Debt / Equity": ("ratio", "Total Debt", "Total Equity"),
    "Free Cash Flow": ("difference", "Operating Cash Flow", "Capital Expenditures"),
    "FCF Margin %": ("ratio", "Free Cash Flow", "Total Revenue"),
    "Market Cap / Revenue": ("billions_to_millions_ratio", "Market Cap ($B)", "Total Revenue"),
    "Trailing P/E (x)": ("copy", "Trailing P/E"),
    "ROE": ("ratio", "Net Income", "Total Equity"),
    "Free Cash Flow Yield": ("ratio_to_billions", "Free Cash Flow", "Market Cap ($B)"),
    "Debt / Total Assets": ("ratio", "Total Debt", "Total Assets"),
}


def _expected_formula_value(label: str, values: dict[str, float]) -> float:
    operation, *operands = FORMULA_RULES[label]
    first = values[operands[0]]
    if operation == "copy":
        return first
    second = values[operands[1]]
    if operation == "difference":
        return first - second
    if operation == "ratio":
        return first / second
    if operation == "billions_to_millions_ratio":
        return first * 1000 / second
    if operation == "ratio_to_billions":
        return first / (second * 1000)
    raise ValueError(f"Unknown formula operation: {operation}")


def _rounded_expected(formula: str, expected: float) -> tuple[float, str | None]:
    match = re.fullmatch(
        r"\s*=\s*(ROUND|ROUNDUP|ROUNDDOWN|TRUNC)\s*\(.+,\s*([+-]?\d+(?:\.0+)?)\s*\)\s*",
        formula,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return expected, None
    try:
        tokens = Tokenizer(formula).items
    except Exception as error:
        raise FormulaEvaluationError("Cannot parse rounded formula") from error
    depth = 0
    separator = None
    for index, token in enumerate(tokens):
        if token.subtype == "OPEN":
            depth += 1
        elif token.subtype == "CLOSE":
            depth -= 1
            if depth == 0 and index != len(tokens) - 1:
                return expected, None
        elif token.type == "SEP" and token.subtype == "ARG" and depth == 1:
            separator = index
    operation, places = match.groups()
    places = int(float(places))
    if not -20 <= places <= 20:
        raise FormulaEvaluationError("Rounding precision is outside the supported range")
    mode = {
        "ROUND": ROUND_HALF_UP,
        "ROUNDUP": ROUND_UP,
        "ROUNDDOWN": ROUND_DOWN,
        "TRUNC": ROUND_DOWN,
    }
    try:
        result = Decimal(str(expected)).quantize(
            Decimal(1).scaleb(-places), rounding=mode[operation.upper()]
        )
        single_round = (
            len(
                re.findall(
                    r"\b(?:ROUND|ROUNDUP|ROUNDDOWN|TRUNC|MROUND|CEILING|FLOOR)\s*\(",
                    formula,
                    re.IGNORECASE,
                )
            )
            == 1
        )
        expression = (
            "".join(token.value for token in tokens[1:separator]).strip()
            if single_round and separator
            else None
        )
        return float(result), expression
    except InvalidOperation as error:
        raise FormulaEvaluationError("Cannot assess rounded formula") from error


def _parse_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.startswith("="):
            return None
        cleaned = stripped.replace(",", "").replace("$", "").replace("%", "")
        try:
            result = float(cleaned)
            return result if math.isfinite(result) else None
        except ValueError:
            return None
    return None


def _color_rgb(color, workbook) -> str | None:
    if color.type == "rgb":
        value = color.rgb
    elif color.type == "indexed":
        index = color.indexed
        palette = workbook._colors or COLOR_INDEX
        value = palette[index] if isinstance(index, int) and 0 <= index < len(palette) else None
    elif color.type == "theme":
        theme_order = (
            "lt1",
            "dk1",
            "lt2",
            "dk2",
            "accent1",
            "accent2",
            "accent3",
            "accent4",
            "accent5",
            "accent6",
            "hlink",
            "folHlink",
        )
        index = color.theme
        if not isinstance(index, int) or not 0 <= index < len(theme_order):
            return None
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        root = ElementTree.fromstring(workbook.loaded_theme or theme_xml)
        entry = root.find(f".//a:clrScheme/a:{theme_order[index]}", namespace)
        node = next(iter(entry), None) if entry is not None else None
        value = (node.get("lastClr") or node.get("val")) if node is not None else None
    else:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})", value):
        return None
    rgb = value[-6:].upper()
    if color.tint:
        channels = [int(rgb[offset : offset + 2], 16) / 255 for offset in (0, 2, 4)]
        hue, lightness, saturation = colorsys.rgb_to_hls(*channels)
        lightness = (
            lightness * (1 + color.tint)
            if color.tint < 0
            else lightness * (1 - color.tint) + color.tint
        )
        channels = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = "".join(f"{round(channel * 255):02X}" for channel in channels)
    return rgb


def _has_fill(cell) -> bool:
    if not cell.fill.fill_type:
        return False
    rgb = _color_rgb(cell.fill.fgColor, cell.parent.parent)
    return rgb is not None and rgb != "FFFFFF"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).strip().casefold())


def _build_label_map(ws) -> dict[str, tuple[str, Any]]:
    """Map normalized label text -> (value_cell_coordinate, value_cell_object).

    A label is any non-formula text cell. Its value cell is the cell in the
    next column of the same row.
    """
    label_map: dict[str, tuple[str, Any]] = {}
    for row in ws.iter_rows():
        for i, cell in enumerate(row):
            if cell.value and isinstance(cell.value, str) and not cell.value.startswith("="):
                norm = _normalize(cell.value)
                if norm in label_map:
                    continue
                if i + 1 < len(row):
                    val_cell = row[i + 1]
                    label_map[norm] = (val_cell.coordinate, val_cell)
    return label_map


def _find_label(label_map: dict[str, tuple[str, Any]], target: str) -> tuple[str, Any] | None:
    required_labels = {"Market Cap / Revenue": "Market Cap / Revenue (x)"}
    normalized = _normalize(required_labels.get(target, target))
    exact = label_map.get(normalized)
    if exact is not None:
        return exact
    pattern = re.compile(rf"^{re.escape(normalized)}\s+(?:—|–|-|\|)\s+(.+)$")
    for text, result in label_map.items():
        match = pattern.fullmatch(text)
        if match and not re.search(
            r"\b(?:no|not|prior|previous|forecast|projected|estimate)\b", match.group(1)
        ):
            return result
    return None


def _find_header_candidates(ws, target: str) -> list[Any]:
    normalized_target = _normalize(target)
    return [
        cell
        for row in ws.iter_rows()
        for cell in row
        if isinstance(cell.value, str)
        and not cell.value.startswith("=")
        and normalized_target in _normalize(cell.value)
    ]


def _match_distinct_headers(groups: list[list[Any]]) -> set[int]:
    assigned: dict[str, int] = {}

    def assign(index: int, visited: set[str]) -> bool:
        for candidate in groups[index]:
            coordinate = candidate.coordinate
            if coordinate in visited:
                continue
            visited.add(coordinate)
            if coordinate not in assigned or assign(assigned[coordinate], visited):
                assigned[coordinate] = index
                return True
        return False

    for index in range(len(groups)):
        assign(index, set())
    return set(assigned.values())


def score_workbook_bytes(agent_bytes: bytes, manifest: dict[str, Any]) -> WorkbookScoreResult:
    try:
        with ZipFile(BytesIO(agent_bytes)) as archive:
            members = archive.infolist()
            if (
                len(members) > 2000
                or sum(member.file_size for member in members) > 64 * 1024 * 1024
            ):
                raise FormulaEvaluationError("Workbook archive exceeds the evaluator's size limit")
        wb = load_workbook(BytesIO(agent_bytes), data_only=False)
    except (BadZipFile, InvalidFileException, ElementTree.ParseError) as exc:
        return WorkbookScoreResult(score=0.0, passed=False, reasons=[f"unreadable_workbook:{exc}"])

    ws = wb.active
    if ws.max_row * ws.max_column > 200_000:
        raise FormulaEvaluationError("Worksheet exceeds the evaluator's size limit")
    reasons: list[str] = []
    total = 0
    passed = 0

    sheet_check = manifest.get("sheet_name_contains", "")
    total += 1
    if sheet_check.lower() in (ws.title or "").lower():
        passed += 1
    else:
        reasons.append(f"sheet_name_missing:{sheet_check}")

    label_map = _build_label_map(ws)

    valid_headers = []
    for header in manifest.get("section_headers", []):
        total += 1
        text = header["text"]
        candidates = _find_header_candidates(ws, text)
        if not candidates:
            reasons.append(f"header_missing:{text}")
            continue
        candidates.sort(
            key=lambda candidate: (
                int(bool(header.get("bold")) and not (candidate.font and candidate.font.bold))
                + int(bool(header.get("fill")) and not _has_fill(candidate)),
                len(_normalize(candidate.value)),
            ),
        )
        cell = candidates[0]
        ok = True
        if header.get("bold") and not (cell.font and cell.font.bold):
            reasons.append(f"header_not_bold:{text}")
            ok = False
        if header.get("fill") and not _has_fill(cell):
            reasons.append(f"header_no_fill:{text}")
            ok = False
        if ok:
            valid_headers.append(
                (
                    text,
                    [
                        candidate
                        for candidate in candidates
                        if (not header.get("bold") or bool(candidate.font and candidate.font.bold))
                        and (not header.get("fill") or _has_fill(candidate))
                    ],
                )
            )

    matched_headers = _match_distinct_headers([candidates for _, candidates in valid_headers])
    passed += len(matched_headers)
    for index, (text, _) in enumerate(valid_headers):
        if index not in matched_headers:
            reasons.append(f"header_not_distinct:{text}")

    for entry in manifest.get("fixed_values", []):
        total += 1
        label = entry["label"]
        expected = entry["value"]
        tolerance = entry.get("tolerance", 1.0)
        result = _find_label(label_map, label)
        if result is None:
            reasons.append(f"fixed_label_missing:{label}")
            continue
        _, cell = result
        actual = _parse_numeric(cell.value)
        if actual is None:
            reasons.append(f"fixed_value_not_numeric:{label}")
            continue
        if abs(actual - expected) > tolerance:
            reasons.append(f"fixed_value_mismatch:{label}:{actual}!={expected}")
        else:
            passed += 1

    evaluator = FormulaEvaluator(wb, ws)
    scenarios = [{}]
    combined_scenarios = [{}, {}, {}, {}]
    generator = random.Random(hashlib.sha256(agent_bytes).digest())
    perturbations = [(1.137, 13.7), (0.863, 7.3)] + [
        (generator.uniform(0.9, 1.1), generator.uniform(3.0, 12.0)) for _ in range(2)
    ]
    for label in [entry["label"] for entry in manifest.get("fixed_values", [])] + manifest.get(
        "live_labels", []
    ):
        found = _find_label(label_map, label)
        if found is None:
            continue
        cell = found[1]
        value = _parse_numeric(cell.value)
        if value is not None:
            key = (ws.title, cell.coordinate)
            for index, (combined, (scale, offset)) in enumerate(
                zip(combined_scenarios, perturbations)
            ):
                replacement = value * scale + offset
                if index < 2:
                    scenarios.append({key: replacement})
                combined[key] = replacement
    scenarios.extend(combined_scenarios)
    scenario_caches = [{} for _ in scenarios]
    invalid_formula_cells = set()

    for entry in manifest.get("formula_cells", []) + manifest.get("dashboard_formulas", []):
        total += 1
        label = entry["label"]
        must_ref_labels = entry["must_reference"]
        result = _find_label(label_map, label)
        if result is None:
            reasons.append(f"formula_label_missing:{label}")
            continue
        _, cell = result
        if not isinstance(cell.value, str) or not cell.value.startswith("="):
            reasons.append(f"not_a_formula:{label}")
            continue
        inputs = {}
        for ref_label in must_ref_labels:
            ref_result = _find_label(label_map, ref_label)
            if ref_result is None:
                reasons.append(f"formula_ref_label_missing:{label}->{ref_label}")
                break
            if (ws.title, ref_result[1].coordinate) in invalid_formula_cells:
                reasons.append(f"formula_ref_invalid:{label}->{ref_label}")
                invalid_formula_cells.add((ws.title, cell.coordinate))
                break
            inputs[ref_label] = ref_result[1]
        else:
            raw_evaluator = FormulaEvaluator(wb, ws)
            for scenario_index, (overrides, cache) in enumerate(zip(scenarios, scenario_caches)):
                try:
                    values = {
                        name: evaluator.numeric(source, overrides, cache)
                        for name, source in inputs.items()
                    }
                    raw_expected = _expected_formula_value(label, values)
                    expected, rounded_expression = _rounded_expected(cell.value, raw_expected)
                    actual = evaluator.numeric(cell, overrides, cache)
                    if (
                        scenario_index == 0
                        and rounded_expression is not None
                        and expected == 0.0
                        and raw_expected != 0.0
                    ):
                        reasons.append(f"formula_value_mismatch:{label}")
                        break
                    if rounded_expression is not None:
                        raw_cell = Cell(
                            ws, row=cell.row, column=cell.column, value="=" + rounded_expression
                        )
                        raw_actual = raw_evaluator.numeric(raw_cell, overrides, {})
                        if not math.isclose(raw_actual, raw_expected, rel_tol=1e-7, abs_tol=1e-7):
                            if (ws.title, cell.coordinate) in raw_evaluator.rounding_cells:
                                raise FormulaEvaluationError(
                                    f"Cannot reliably assess rounded formula: {label}"
                                )
                            reasons.append(f"formula_value_mismatch:{label}")
                            break
                except FormulaCalculationError as error:
                    if scenario_index == 0:
                        reasons.append(f"formula_calculation_error:{label}")
                        invalid_formula_cells.add((ws.title, cell.coordinate))
                        break
                    raise FormulaEvaluationError(
                        f"Cannot assess perturbed formula: {label}"
                    ) from error
                except ZeroDivisionError as error:
                    if scenario_index == 0:
                        reasons.append(f"formula_calculation_error:{label}")
                        invalid_formula_cells.add((ws.title, cell.coordinate))
                        break
                    raise FormulaEvaluationError(
                        f"Cannot assess perturbed formula: {label}"
                    ) from error
                if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
                    key = (ws.title, cell.coordinate)
                    if key in evaluator.rounding_cells and (
                        rounded_expression is None or key in evaluator.nested_rounding_cells
                    ):
                        raise FormulaEvaluationError(
                            f"Cannot reliably assess rounded formula: {label}"
                        )
                    reasons.append(f"formula_value_mismatch:{label}")
                    break
            else:
                passed += 1

    for label in manifest.get("live_labels", []):
        total += 1
        result = _find_label(label_map, label)
        if result is None:
            reasons.append(f"live_label_missing:{label}")
            continue
        _, cell = result
        actual = _parse_numeric(cell.value)
        if actual is None:
            reasons.append(f"live_value_not_numeric:{label}")
        else:
            passed += 1

    score = round(passed / total, 4) if total > 0 else 0.0
    return WorkbookScoreResult(score=score, passed=(score >= 1.0), reasons=reasons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent")
    parser.add_argument("--manifest")
    parser.add_argument("--stdin", action="store_true")
    args = parser.parse_args()

    if args.stdin:
        if args.agent or args.manifest:
            parser.error("--stdin cannot be combined with file arguments")
        payload = json.load(sys.stdin)
        agent_bytes = base64.b64decode(payload["workbook"], validate=True)
        manifest = payload["manifest"]
    else:
        if not args.agent or not args.manifest:
            parser.error("--agent and --manifest are required without --stdin")
        with open(args.agent, "rb") as f:
            agent_bytes = f.read()
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    result = score_workbook_bytes(agent_bytes=agent_bytes, manifest=manifest)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
