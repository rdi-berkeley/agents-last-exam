#!/usr/bin/env python
"""Extract Chinese e-discovery evidence index workbook content as JSON.

This script runs on the VM and outputs the workbook's textual content
in a structured format. Actual scoring is done by the LLM judge in evaluate().
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Any


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return max(value - 1, 0)


def read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall(".//x:si", ns):
        strings.append("".join((node.text or "") for node in item.findall(".//x:t", ns)))
    return strings


def workbook_rows(workbook_path: Path) -> list[list[str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(workbook_path) as archive:
        shared = read_shared_strings(archive)
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(archive.read(sheet_name))
        rows: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", ns):
            values: list[str] = []
            for cell in row.findall("x:c", ns):
                idx = column_index(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join((node.text or "") for node in cell.findall(".//x:t", ns))
                else:
                    raw = cell.find("x:v", ns)
                    if raw is None or raw.text is None:
                        value = ""
                    elif cell_type == "s":
                        value = shared[int(raw.text)] if raw.text.isdigit() and int(raw.text) < len(shared) else raw.text
                    else:
                        value = raw.text
                values[idx] = cell_text(value)
            rows.append(values)
        return rows


def extract_workbook(workbook_path: Path) -> dict[str, Any]:
    """Extract workbook content as structured data for LLM judging."""
    if not workbook_path.exists():
        return {"ok": False, "error": "missing 证据目录.xlsx", "headers": [], "rows": []}

    try:
        sheet_rows = workbook_rows(workbook_path)
    except Exception as exc:
        return {"ok": False, "error": f"invalid workbook: {exc}", "headers": [], "rows": []}

    header_row_idx = None
    headers: list[str] = []
    for idx, values in enumerate(sheet_rows):
        joined = "".join(values)
        if "证据名称" in joined and "证明目的" in joined:
            header_row_idx = idx
            headers = [v for v in values if v]
            break

    if header_row_idx is None:
        return {"ok": False, "error": "no evidence-index header row found", "headers": [], "rows": []}

    data_rows: list[dict[str, str]] = []
    last_group = ""
    for values in sheet_rows[header_row_idx + 1:]:
        if not any(values):
            continue
        item: dict[str, str] = {}
        for i, h in enumerate(headers):
            item[h] = values[i] if i < len(values) else ""
        if item.get("分组"):
            last_group = item["分组"]
        elif last_group:
            item["分组"] = last_group
        data_rows.append(item)

    return {
        "ok": True,
        "error": None,
        "headers": headers,
        "rows": data_rows,
        "num_rows": len(data_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output
    candidates = sorted(output_dir.glob("*.xlsx"))
    target = output_dir / "证据目录.xlsx"
    if not target.exists() and len(candidates) == 1:
        target = candidates[0]
    print(json.dumps(extract_workbook(target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
