"""Public representation rules for the CM mapping specification."""

CM_RULES = {
    "scope": (
        "Map every distinct annotated CM/SUPPCM target for the CM forms in the supplied "
        "aCRF, including numbered hidden dictionary fields and conditional follow-up "
        "controls. Include both targets when a field has multiple annotations. Include "
        "study-specific category/subcategory constants annotated at form-header level "
        "as separate rows; never attribute a header constant to a different numbered "
        "question. Exclude items marked NOT SUBMITTED, global identifiers such as "
        "STUDYID/DOMAIN/USUBJID, and XML-only variables with no annotation on these forms. "
        "A field's inclusion does not depend on whether its target origin is CRF, "
        "Assigned or Derived. Do not add SUPPQUAL infrastructure rows."
    ),
    "source_text": (
        "Use the descriptive form title and parenthesized short form code, omitting "
        "the C4591001 study prefix and Repeating Form suffix. Use the printed field "
        "label without its question number, response options or explanatory instructions. "
        "Use its bracketed placeholder text without the brackets; for a conditional "
        "field with no placeholder, repeat its visible label. For a header constant use "
        "crf_field_label=Form header constant and crf_item_or_placeholder=Header. "
        "Use the sample CRF to cross-check broken aCRF text. Case, punctuation, spacing "
        "and [hidden] markers do not affect source-text matching; letters/digits and "
        "field identity do. A precise locator aCRF:<physical page>:<printed question "
        "number> may replace a damaged field label or placeholder; append :1 for the "
        "unnumbered End Date follow-up, and use :header for a header constant. A locator "
        "identifies the source only, never the target or its metadata."
    ),
    "role": (
        "Use the following task role convention, not XML ItemRef Role Req/Exp/Perm: "
        "Identifier for source medication identifiers; Topic for the reported medication "
        "name; Grouping Qualifier for category/subcategory; Synonym Qualifier for a "
        "standardized dictionary name; Record Qualifier for numeric/text dose amounts; "
        "Variable Qualifier for dose units, frequency and route; Timing for dates and "
        "temporal reference points; Supplemental Qualifier for every SUPPCM mapping. "
        "Role case and spacing do not affect matching."
    ),
    "origin": (
        "Use the target ItemDef Origin Type from sdtm_define.xml. For SUPPCM resolve "
        "QVAL through ValueListRef and the QNAM-specific WhereClause. Do not infer CRF "
        "origin merely from appearance on the aCRF, or Derived from date formatting. "
        "Use CRF, Assigned or Derived (case-insensitive); aCRF is an alias for CRF only. "
        "The separate ADaM file is not the authority for the SUPPCM value-level origin."
    ),
    "controlled_values": (
        "For form-restricted categories and annotated constants, give the complete "
        "set of encoded values shown for that form, not the broader study code list. "
        "Use a semicolon-separated list or JSON array of strings; order, letter case, "
        "spacing and balanced quotation marks around each value are ignored, but "
        "duplicates, empty members, extra or missing values are incorrect. For other "
        "linked code lists, including external dictionaries, write the exact CodeListOID "
        "from the applicable ItemDef rather than invented descriptions or unavailable "
        "dictionary entries. CodeListOID case and punctuation are significant. For "
        "datetime with no code list use ISO 8601 date/datetime (ISO 8601 datetime and "
        "ISO 8601 date/time are equivalent); with neither a code list nor an annotated "
        "constant use an empty cell. Put transfer conditions in mapping_rule."
    ),
    "prose": (
        "mapping_rule and notes must be nonempty and field-specific. mapping_rule must "
        "name the exact target variable as a whole word; notes must cite the source "
        "form/page or XML evidence. Their wording is not compared to hidden sentences. "
        "The deterministic prose checks enforce nonempty text and the target token, "
        "not unrestricted natural-language equivalence."
    ),
    "comparison": (
        "The original binary grading is unchanged: all required mappings and structured "
        "values must be correct, with no missing, extra or duplicate mappings. Keep the "
        "exact eleven-column header/order; standard CSV quoting, UTF-8 BOM and row order "
        "are accepted. Dataset names, target names and YES/NO flags are case-sensitive. "
        "Formatting normalization never substitutes one source field or SDTM variable "
        "for another."
    ),
}

CM_CONTRACT_TEXT = "\n\n".join(CM_RULES.values())

CM_FORMS = (
    (15, "BASELINE (CONMED BSL)", True, False, 12),
    (16, "NON STUDY VACCINATIONS (CONMED VAX)", False, False, 8),
    (76, "PROHIBITED (PROHIB CM)", True, True, 13),
    (110, "VASOPRESSORS (VASOPRESS)", False, True, 9),
)


def cm_source_fields():
    """Yield source identities transcribed from the four supplied CM forms."""
    for page, title, has_dose, has_ongoing, last_item in CM_FORMS:
        fields = [
            ("1", "What is the medication identifier?", "Sponsor-Defined Identifier"),
            ("2", "Category", "Category for Medication"),
            ("4", "Medication", "Name of Medication"),
        ]
        if has_dose:
            fields.extend(
                [
                    ("5", "Dose", "Dose Description"),
                    ("6", "Dose Unit", "Dose Unit"),
                    ("7", "Dose Frequency", "Dose Frequency"),
                    ("8", "Route", "Route"),
                ]
            )
        fields.append(
            ("9" if has_dose else "5", "Date" if page == 16 else "Start Date", "Start Date")
        )
        if has_ongoing:
            number = "10" if has_dose else "6"
            fields.extend(
                [(number, "Ongoing?", "Ongoing"), (number + ":1", "End Date", "End Date")]
            )
        fields.extend(
            [
                (
                    str(last_item - 1),
                    "Standardized Medication Name - Dictionary derived",
                    "Standardized Medication Name",
                ),
                (
                    str(last_item),
                    "Standardized Medication Code - Dictionary derived",
                    "Standardized Medication Code",
                ),
            ]
        )
        if page == 110:
            fields.append(("header", "Form header constant", "Header"))
        for number, label, placeholder in fields:
            yield {
                "form": "CONCOMITANT MEDICATIONS - " + title,
                "locator": f"aCRF:{page}:{number}",
                "label": label,
                "placeholder": placeholder,
            }
