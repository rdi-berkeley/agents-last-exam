"""Stage the source-verified AE contract repair from the preserved original release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_LOCATORS = {
    "AECAT": "11:1",
    "AESPID": "11:2",
    "AETERM": "11:3",
    "AESTDTC": "11:4",
    "AEENRTPT": "11:5",
    "AEENTPT": "11:5",
    "AEENDTC": "11:5:1",
    "AETOXGR": "11:6",
    "AESER": "11:7",
    "AESCONG": "11:7:1",
    "AESDTH": "11:7:2",
    "AESHOSP": "11:7:3",
    "AESDISAB": "11:7:4",
    "AESLIFE": "11:7:5",
    "AESMIE": "11:7:6",
    "AEMERES": "11:8",
    "AEREL": "11:9",
    "AERELNST": "11:9:1",
    "AERELTXT": "11:9:2",
    "AEACN": "11:10",
    "AECONTRT": "12:11;12:12",
    "AECMGIV": "12:11",
    "AENDGIV": "12:12",
    "AEOUT": "12:13",
    "AESUBJDC": "12:14",
    "AEREFID": "12:15",
    "AELLT": "12:17",
    "AELLTCD": "12:18",
    "AEDECOD": "12:19",
    "AEPTCD": "12:20",
    "AEHLT": "12:21",
    "AEHLTCD": "12:22",
    "AEHLGT": "12:23",
    "AEHLGTCD": "12:24",
    "AEBODSYS": "12:25",
    "AESOC": "12:25",
    "AEBDSYCD": "12:26",
    "AESOCCD": "12:26",
}

CONTRACT_RULES = {
    "scope": "Map the numbered fields and their field-level AE/SUPPAE annotations on annotated_crf.pdf physical pages 11-12 (the ADVERSE EVENT REPORT form), including hidden coding fields. Exclude form-header/global identifiers not attached to a numbered question and fields marked NOT SUBMITTED. Include each distinct annotated target once, combining multiple source questions in one row when they share a target. Do not include variables merely because they exist elsewhere in define.xml.",
    "source_locator": "crf_item_or_placeholder is a source locator, not arbitrary prose: aCRF:<physical page>:<printed question number>. Use the page on which the question starts, even if it continues on the next page. For unnumbered follow-up fields append :<1-based ordinal> in reading order within that question. The parent field has no suffix. Subquestions, not individual response choices, receive ordinals. Multiple source fields use semicolon-separated locators; order is ignored. A different nonempty source descriptor loses this column's credit, not all row credit.",
    "field_label": "Copy the printed question/field label, excluding bracketed placeholders, question numbers, [hidden], response choices and explanatory instructions. Case, punctuation and whitespace (including PDF extraction spacing) are ignored. For multiple source fields join their full labels with ||; their order is ignored. crf_form is ADVERSE EVENT REPORT (AE).",
    "role": "Use these task role categories, not define.xml ItemRef Role values Req/Exp/Perm (which encode core status): Identifier for the source/event identifiers; Topic for the reported adverse event term; Grouping Qualifier for the event category; Synonym Qualifier for dictionary terms and their codes; Timing for dates and temporal reference points; Record Qualifier for other primary-event descriptors (flags, severity, causality, action and outcome); Supplemental Qualifier for every SUPPAE mapping. Case is ignored.",
    "origin": "Use the applicable sdtm_define.xml ItemDef Origin Type. Follow ValueListRef and WhereClause for the ADVERSE EVENT form/category when origin is conditional; use the QNAM-specific QVAL ItemDef for SUPPAE. Canonical names are CRF, Assigned and Derived. CRF/aCRF and aCRF are accepted aliases for CRF only, never Assigned. Formatting a collected date as ISO 8601 does not itself change its XML origin.",
    "controlled_values": "For finite controlled values, give the complete set of encoded target values for this form as a semicolon-separated list or JSON array of strings. Order/case/spacing is ignored; duplicates, empty elements, extra or missing terms are incorrect. Use define.xml CodedValue, not a display label such as YES/NO when target codes are Y/N. The complete Boolean code set may also be written Y/N or N/Y. Restrict choices/constants to those on this form (the XML covers other forms too). For unbounded text with no codelist use Free text; for datetime use ISO 8601 datetime (ISO 8601 date/time is equivalent). For MedDRA dictionary fields use MedDRA followed by LLT, PT, HLT, HLGT or SOC and append code for numeric code fields. Dictionary-Derived Term is the PT level. These dictionary-level labels identify types, not finite lists of dictionary entries. Put conditions in mapping_rule, not in this column.",
    "mapping_rule": "Write an explanation of the transfer, coding, constant or conditional logic, naming the exact sdtm_variable as a whole word (case-insensitive). At least two words are required. Wording is not compared with a hidden sentence. The deterministic prose check validates these format/content requirements, not unrestricted natural-language equivalence; source/target/role/origin/value accuracy is checked in the structured columns.",
    "notes": "Optional for every row, independently of whether the reference has a note. Leave blank or provide textual source/context information; bare TODO, TBD, N/A, null, numeric-only and punctuation-only placeholders are not explanatory notes. Note wording is never compared with a hidden sentence.",
    "comparison": "Each of the 11 columns has equal weight. Rows match by dataset and target variable, with each submitted row used at most once. Strict success additionally requires matching normalized source-locator composite keys and no extra rows. Coverage is matched rows divided by max(reference rows, submitted rows). Score = coverage * mean matched-row column accuracy, rounded to four decimals. Missing/extra rows and wrong source/role/origin/values retain partial credit for correct fields. Invalid headers/order, malformed CSV, duplicate or empty composite keys, forbidden datasets/flags or inconsistent flags score zero. SUPPAE should use QNAM, not QVAL; QVAL can receive partial credit only when its source cell explicitly contains the exact QNAM token, and still loses target/source-column credit.",
}


def build(original: Path, output: Path) -> None:
    if original.resolve() == output.resolve() or original.resolve() in output.resolve().parents:
        raise ValueError("Output must not overwrite or be inside the original backup")
    manifest = json.loads((original / "reference/source_manifest.json").read_text())
    before_hashes = {}
    for relative in [
        "input/output_contract.json",
        "input/task_brief.md",
        "reference/ae_mapping.csv",
        "reference/source_manifest.json",
    ]:
        before_hashes[relative] = hashlib.sha256((original / relative).read_bytes()).hexdigest()
    for filename, entry in manifest["raw_files"].items():
        relative = (
            f"reference/{filename}"
            if filename.endswith(".csv")
            else f"input/source_documents/{filename}"
        )
        if hashlib.sha256((original / relative).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Original source hash mismatch: {relative}")
    namespace = {"o": "http://www.cdisc.org/ns/odm/v1.3", "d": "http://www.cdisc.org/ns/def/v2.0"}
    root = ET.parse(original / "input/source_documents/sdtm_define.xml")
    items = {item.get("OID"): item for item in root.findall(".//o:ItemDef", namespace)}
    lists = {item.get("OID"): item for item in root.findall(".//o:CodeList", namespace)}
    with (original / "reference/ae_mapping.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    if set(SOURCE_LOCATORS) != {row["sdtm_variable"] for row in rows}:
        raise ValueError("Unexpected original target set")
    changes = []
    for row in rows:
        before = dict(row)
        variable = row["sdtm_variable"]
        row["crf_item_or_placeholder"] = ";".join(
            "aCRF:" + locator for locator in SOURCE_LOCATORS[variable].split(";")
        )
        item = items.get(f"IT.AE.{variable}")
        if item is not None:
            origin = item.find("d:Origin", namespace)
            row["origin"] = origin.get("Type") if origin is not None else "CRF"
        else:
            row["origin"] = "CRF"
        if row["role"] == "Result Qualifier":
            row["role"] = "Record Qualifier"
        if variable == "AECAT":
            row["role"] = "Grouping Qualifier"
        if variable == "AESCONG":
            row["crf_field_label"] = (
                "Is this serious event associated with congenital anomaly or birth defect?"
            )
        if variable == "AEREL":
            row["crf_field_label"] = "Is this event related to study treatment:"
        if variable == "AECONTRT":
            row["crf_field_label"] = (
                "Was a Concomitant Medication given? || Was a Non-Drug Treatment given?"
            )
            row["mapping_rule"] = (
                "Map the collected concomitant medication/non-drug treatment responses to the treatment-given flag AECONTRT using the target Boolean codes."
            )
            row["notes"] = (
                "Both questions are annotated AECONTRT on aCRF page 12; define.xml CL.NY provides N and Y."
            )
        if variable == "AEREFID":
            row["mapping_rule"] = (
                "Map the sponsor serious adverse event reference number directly to AEREFID."
            )
            row["notes"] = (
                "aCRF page 12 annotates question 15 to AEREFID; NOT SUBMITTED belongs to question 16, Comparison Term."
            )
        if variable == "AERELNST":
            row["mapping_rule"] = (
                "When the event is not related to study treatment, map the selected non-study cause to AERELNST."
            )
        code_ref = item.find("o:CodeListRef", namespace) if item is not None else None
        if code_ref is not None and code_ref.get("CodeListOID") != "CL.MedDRA":
            codes = [
                entry.get("CodedValue")
                for entry in lists[code_ref.get("CodeListOID")]
                if entry.get("CodedValue")
            ]
            if variable == "AECAT":
                codes = [code for code in codes if code == "ADVERSE EVENT"]
            elif variable == "AEACN":
                codes = [code for code in codes if code in {"DRUG WITHDRAWN", "NOT APPLICABLE"}]
            row["controlled_terms_or_expected_values"] = "; ".join(codes)
        elif variable in {"AECMGIV", "AENDGIV", "AESUBJDC", "AEMERES"}:
            row["controlled_terms_or_expected_values"] = "N; Y"
        elif variable in {"AESPID", "AEREFID", "AETERM", "AERELTXT"}:
            row["controlled_terms_or_expected_values"] = "Free text"
        elif variable == "AEENRTPT":
            row["controlled_terms_or_expected_values"] = "ONGOING"
        elif variable == "AEENTPT":
            row["controlled_terms_or_expected_values"] = "LAST SUBJECT ENCOUNTER"
        elif variable == "AEDECOD":
            row["controlled_terms_or_expected_values"] = "MedDRA PT"
        elif variable in {"AEBODSYS", "AESOC"}:
            row["controlled_terms_or_expected_values"] = "MedDRA SOC"
        elif variable in {"AEBDSYCD", "AESOCCD"}:
            row["controlled_terms_or_expected_values"] = "MedDRA SOC code"
        row["mapping_rule"] = f"{variable}: {row['mapping_rule']}"
        changes.append(
            {
                "target": variable,
                "changes": {
                    column: {"before": before[column], "after": row[column]}
                    for column in columns
                    if before[column] != row[column]
                },
            }
        )
    contract = json.loads((original / "input/output_contract.json").read_text())
    contract["contract_version"] = "20260907-source-verified"
    contract["column_notes"] = {
        "crf_form": CONTRACT_RULES["field_label"],
        "crf_field_label": CONTRACT_RULES["field_label"],
        "crf_item_or_placeholder": CONTRACT_RULES["source_locator"],
        "sdtm_variable": contract["column_notes"]["sdtm_variable"],
        "role": CONTRACT_RULES["role"],
        "origin": CONTRACT_RULES["origin"],
        "controlled_terms_or_expected_values": CONTRACT_RULES["controlled_values"],
        "mapping_rule": CONTRACT_RULES["mapping_rule"],
        "notes": CONTRACT_RULES["notes"],
    }
    contract["scope"] = CONTRACT_RULES["scope"]
    contract["comparison"] = CONTRACT_RULES["comparison"]
    contract["normalization"] = (
        "Trim/collapse whitespace globally; additionally apply the explicit column-specific rules. Dataset names, target variables and flags remain case-sensitive."
    )
    contract["example_row"] = {
        "_comment": "Format example; infer all mappings from the source files.",
        **next(row for row in rows if row["sdtm_variable"] == "AETERM"),
    }
    (output / "input").mkdir(parents=True, exist_ok=True)
    (output / "reference").mkdir(parents=True, exist_ok=True)
    (output / "input/output_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    brief = (original / "input/task_brief.md").read_text()
    brief += "\n## Scope and Output Semantics\n\n" + CONTRACT_RULES["scope"] + "\n\n"
    brief += "Follow output_contract.json for source locators, semantic roles, XML-derived origins, encoded values, required target-naming explanations and uniformly optional notes. The contract, not a hidden wording convention, defines these interfaces.\n"
    (output / "input/task_brief.md").write_text(brief)
    with (output / "reference/ae_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest["active_revision"] = {
        "id": contract["contract_version"],
        "original_file_hashes": before_hashes,
        "generated_file_hashes": {
            relative: hashlib.sha256((output / relative).read_bytes()).hexdigest()
            for relative in [
                "input/output_contract.json",
                "input/task_brief.md",
                "reference/ae_mapping.csv",
            ]
        },
        "source_evidence": "Original SDTM XML ItemDefs, conditional ValueLists/WhereClauses and codelists; annotated_crf.pdf physical pages 11-12, sample_crf.pdf pages 2-4. Raw source hashes above remain unchanged.",
        "corrections": changes,
        "annotation_evidence": {
            "AEREFID": "aCRF page 12 y=258-269 aligns question 15; NOT SUBMITTED y=305-315 aligns Comparison Term/question 16 y=300-322",
            "AECONTRT": "aCRF page 12 questions 11 and 12, XML IT.AE.AECONTRT -> CL.NY; non-study causes belong to IT.AE.AERELNST -> CL.AERELNST",
        },
    }
    (output / "reference/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.original, arguments.output)
