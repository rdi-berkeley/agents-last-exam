"""Evaluate a bounded subset of workbook formulas without executing the workbook."""

from __future__ import annotations

import math
import re

import numpy as np
from formulas import Parser
from formulas.tokens.operand import XlError
from openpyxl.formula import Tokenizer
from openpyxl.utils.cell import range_boundaries

SUPPORTED_FUNCTIONS = frozenset(
    {
        "ABS",
        "AND",
        "AVERAGE",
        "CEILING",
        "EXP",
        "FLOOR",
        "FALSE",
        "IF",
        "IFERROR",
        "IFNA",
        "INDIRECT",
        "INT",
        "LEN",
        "LET",
        "LN",
        "LOG",
        "LOG10",
        "MAX",
        "MIN",
        "MOD",
        "MROUND",
        "N",
        "NOT",
        "OR",
        "POWER",
        "PRODUCT",
        "ROUND",
        "ROUNDDOWN",
        "ROUNDUP",
        "SIGN",
        "SQRT",
        "SUM",
        "SUMPRODUCT",
        "TRUNC",
        "TRUE",
    }
)


class FormulaEvaluationError(RuntimeError):
    """A workbook formula cannot be assessed reliably by this evaluator."""


class FormulaCalculationError(FormulaEvaluationError):
    """The workbook formula produces a definite spreadsheet calculation error."""


class FormulaEvaluator:
    def __init__(self, workbook, worksheet):
        self.workbook = workbook
        self.worksheet = worksheet
        self.compiled = {}
        self.rounding_cells = set()
        self.nested_rounding_cells = set()
        self.remaining_cells = 0

    def _reference_value(self, reference, overrides, cache, stack):
        if not isinstance(reference, str) or "[" in reference or "]" in reference:
            raise FormulaEvaluationError("External or unsupported workbook reference")
        sheet = self.worksheet
        if "!" in reference:
            sheet_name, reference = reference.rsplit("!", 1)
            sheet_name = sheet_name.strip("'").replace("''", "'")
            matches = [
                item
                for item in self.workbook.worksheets
                if item.title.casefold() == sheet_name.casefold()
            ]
            if len(matches) != 1:
                raise FormulaEvaluationError("Unknown worksheet reference")
            sheet = matches[0]
        try:
            min_col, min_row, max_col, max_row = range_boundaries(reference.replace("$", ""))
        except ValueError as error:
            raise FormulaEvaluationError("Unsupported named or dynamic reference") from error
        if None in (min_col, min_row, max_col, max_row):
            raise FormulaEvaluationError("Unbounded or unsupported reference")
        cell_count = (max_col - min_col + 1) * (max_row - min_row + 1)
        if cell_count > 10000 or cell_count > self.remaining_cells:
            raise FormulaEvaluationError("Formula reference is too large")
        self.remaining_cells -= cell_count
        values = [
            [
                self._cell_value(sheet._cells[(row, column)], overrides, cache, stack)
                if (row, column) in sheet._cells
                else 0
                for column in range(min_col, max_col + 1)
            ]
            for row in range(min_row, max_row + 1)
        ]
        return (
            values[0][0]
            if min_row == max_row and min_col == max_col
            else np.asarray(values, dtype=object)
        )

    def _cell_value(self, cell, overrides, cache, stack):
        key = (cell.parent.title, cell.coordinate)
        if key in overrides:
            return overrides[key]
        if key in cache:
            if key in self.rounding_cells:
                self.rounding_cells.update(stack)
                self.nested_rounding_cells.update(stack)
            return cache[key]
        if key in stack or len(stack) >= 100:
            raise FormulaEvaluationError("Circular or excessively deep formula dependency")
        value = cell.value
        if cell.data_type != "f":
            if cell.data_type == "e":
                return XlError(value)
            return 0 if value is None else value
        if not isinstance(value, str) or len(value) > 8192:
            raise FormulaEvaluationError("Unsupported or excessively long formula")
        stack.add(key)
        if re.search(
            r"\b(?:ROUND|ROUNDUP|ROUNDDOWN|TRUNC|MROUND|CEILING|FLOOR)\s*\(", value, re.IGNORECASE
        ):
            self.rounding_cells.update(stack)
            self.nested_rounding_cells.update(stack - {key})
        try:
            if key not in self.compiled:
                tokens = Tokenizer(value).items
                if len(tokens) > 1024:
                    raise FormulaEvaluationError("Formula has too many tokens")
                functions = {
                    token.value[:-1].upper().removeprefix("_XLFN.").removeprefix("_XLWS.")
                    for token in tokens
                    if token.type == "FUNC" and token.subtype == "OPEN"
                }
                if functions - SUPPORTED_FUNCTIONS:
                    raise FormulaEvaluationError("Unsupported spreadsheet function")
                for index, token in enumerate(tokens):
                    name = token.value[:-1].upper().removeprefix("_XLFN.")
                    if (
                        token.type != "FUNC"
                        or token.subtype != "OPEN"
                        or name not in {"ROUND", "ROUNDUP", "ROUNDDOWN", "TRUNC"}
                    ):
                        continue
                    depth = 1
                    separator = None
                    for position in range(index + 1, len(tokens)):
                        current = tokens[position]
                        if current.subtype == "OPEN":
                            depth += 1
                        elif current.subtype == "CLOSE":
                            depth -= 1
                            if depth == 0:
                                break
                        elif current.type == "SEP" and current.subtype == "ARG" and depth == 1:
                            separator = position
                    if separator is None:
                        if name == "TRUNC":
                            continue
                        raise FormulaEvaluationError("Unsupported rounding precision")
                    precision = "".join(
                        item.value for item in tokens[separator + 1 : position]
                    ).strip()
                    if (
                        not re.fullmatch(r"[+-]?\d{1,2}(?:\.0+)?", precision)
                        or abs(float(precision)) > 20
                    ):
                        raise FormulaEvaluationError("Unsupported rounding precision")
                self.compiled[key] = Parser().ast(value)[1].compile()
            function = self.compiled[key]
            arguments = [
                self._reference_value(ref, overrides, cache, stack) for ref in function.inputs
            ]
            result = function(*arguments)
            if isinstance(result, np.ndarray):
                if result.size != 1:
                    raise FormulaEvaluationError("Formula does not resolve to one value")
                result = result.item()
            cache[key] = result
            return result
        except FormulaEvaluationError:
            raise
        except Exception as error:
            raise FormulaEvaluationError(
                "Unsupported or unassessable spreadsheet formula"
            ) from error
        finally:
            stack.remove(key)

    def numeric(self, cell, overrides, cache):
        self.remaining_cells = 20000
        value = self._cell_value(cell, overrides, cache, set())
        if isinstance(value, XlError):
            if str(value) == "#DIV/0!":
                raise FormulaCalculationError("Formula divides by zero")
            raise FormulaEvaluationError(f"Formula returned a spreadsheet error: {value}")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            if cell.data_type == "f":
                raise FormulaCalculationError("Formula returned a non-numeric value")
            if isinstance(value, str):
                try:
                    value = float(value.strip().replace(",", "").replace("$", "").replace("%", ""))
                except ValueError as error:
                    raise FormulaCalculationError("Input is not a number") from error
            else:
                raise FormulaCalculationError("Input is not a number")
        result = float(value)
        if not math.isfinite(result):
            raise FormulaEvaluationError("Formula returned a non-finite number")
        return result
