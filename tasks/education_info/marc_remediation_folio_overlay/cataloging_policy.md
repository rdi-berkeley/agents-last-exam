# Cataloging Remediation Policy

Apply these rules to every visible and hidden batch.

1. Match incoming bibliographic records to FOLIO instances by normalized OCLC number, then LCCN, then ISBN. OCLC values must be emitted as `(OCoLC)<digits>` with no `ocm`, `ocn`, spaces, or punctuation other than the prefix.
2. When more than one incoming record has the same normalized OCLC, keep the most complete record: prefer records with a 100 field, a 300 field, and a non-VENDOR 040$a. Emit the weaker duplicates in `overlay_decisions.csv` with action `suppress_duplicate` and do not include them in remediated_records.xml or import_plan records.
3. Convert AACR2-era records to RDA by ensuring 040$b eng and 040$e rda, removing 245$h, and adding 336/337/338 fields. Use `text/txt`, `unmediated/n`, `volume/nc` for print records and `text/txt`, `computer/c`, `online resource/cr` for online records.
4. Use `authority_map.csv` to replace exact 100/700/650 `$a` values and add `$0` with the provided URI. Preserve relationship terms such as `$e author`.
5. For FOLIO actions, emit `update_instance` when a match exists and `create_instance` otherwise. Existing uncataloged matched instances must become status `Cataloged` in the import plan.
6. Map 949 fields to FOLIO holdings/items using `location_map.csv`: 949$a location, 949$b call number, 949$c material type hint, 949$d loan policy hint, 949$e barcode.

## Hidden-case rule coverage

Do not special-case record IDs. Apply the documented match precedence, duplicate survivor ranking, RDA carrier mapping, authority replacement, and 949 holdings/item mapping uniformly to every case directory passed through `--case-dir`.

## RDA carrier evidence and precedence

For this migration, `949$a=ONLINE` (case-insensitive, ignoring surrounding
whitespace) is authoritative evidence of online delivery, even when `300$a`
describes printed pages. Otherwise, `online resource` in `300$a` identifies an
online record; otherwise use print. Do not let a page count, the obsolete
`245$h`, or the material/loan hints in `949$c/$d` override online evidence.
Preserve the original `300` and `949` fields: this rule selects `337/338`, not
a rewrite of the physical description or the holdings data.

Emit the vocabulary source as well as the term/code: `336$a text $b txt
$2 rdacontent`; print uses `337$a unmediated $b n $2 rdamedia` and
`338$a volume $b nc $2 rdacarrier`; online uses `337$a computer $b c
$2 rdamedia` and `338$a online resource $b cr $2 rdacarrier`.
The same precedence and vocabulary rules apply to every case.
