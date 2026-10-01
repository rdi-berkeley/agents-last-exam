"""Scorer regressions and independent, read-only reference-provenance diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from tasks.life_sciences.genomic_interval_processing_1.scripts import score_outputs as scorer


TASK_DIR = Path(scorer.__file__).parents[1]
DATA_DIR = (
    TASK_DIR.parents[2] / "task-data-hf/extracted/life_sciences/genomic_interval_processing_1/base"
)
COUNTS = dict(zip(scorer.INPUT_BED_FILES, [2, 3, 4], strict=True))
REFERENCE = b"chr1\t10\t25\nchr10\t0\t4\nchr2\t30\t60\n"


@pytest.fixture
def outputs():
    return {
        "union_peaks.bed": REFERENCE,
        "commands.sh": b"#!/bin/sh\n# Recorded interval workflow: bedtools merge followed by sort.\n",
        "summary.json": json.dumps(
            {
                "input_files": scorer.INPUT_BED_FILES,
                "input_interval_counts": COUNTS,
                "total_input_intervals": 9,
                "output_file": "union_peaks.bed",
                "output_intervals": 3,
            }
        ).encode(),
    }


@pytest.mark.parametrize(
    "filename",
    [
        "union_peaks.bed",
        "./union_peaks.bed",
        "output/union_peaks.bed",
        "./output//union_peaks.bed",
        "../output/union_peaks.bed",
        "/media/user/data/agenthle/life_sciences/genomic_interval_processing_1/base/output/union_peaks.bed",
        "/local directory/union_peaks.bed",
    ],
)
def test_filename_label_accepts_local_posix_basename_equivalence(outputs, filename):
    summary = json.loads(outputs["summary.json"])
    summary["output_file"] = filename
    outputs["summary.json"] = json.dumps(summary).encode()
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 1.0 and report.passed
    assert report.details["exact_reference_match"]


@pytest.mark.parametrize(
    "filename",
    [
        None,
        12,
        [],
        {},
        True,
        "",
        "other.bed",
        "UNION_PEAKS.BED",
        "union_peaks.bed.backup",
        "not_union_peaks.bed",
        "output/union_peaks.bed/",
        "output/union_peaks.bed/.",
        "output/union_peaks.bed/..",
        "https://example.org/union_peaks.bed",
        "file:///output/union_peaks.bed",
        "gs://bucket/union_peaks.bed",
        "output\\union_peaks.bed",
        "output/union_peaks.bed\n",
        "bad\x00/union_peaks.bed",
        "bad\t/union_peaks.bed",
    ],
)
def test_wrong_or_malformed_filename_loses_only_existing_summary_credit(outputs, filename):
    summary = json.loads(outputs["summary.json"])
    summary["output_file"] = filename
    outputs["summary.json"] = json.dumps(summary).encode()
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 0.94 and not report.passed
    assert report.details["exact_reference_match"]


@pytest.mark.parametrize(
    "payload",
    [
        REFERENCE,
        b"\xef\xbb\xbf" + REFERENCE,
        REFERENCE.replace(b"\n", b"\r\n"),
        REFERENCE.replace(b"\n", b"\r"),
        REFERENCE.replace(b"\n", b"  \n") + b"\n\n",
    ],
)
def test_existing_bed_text_normalizations_preserve_full_credit(outputs, payload):
    outputs["union_peaks.bed"] = payload
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 1.0 and report.passed


@pytest.mark.parametrize(
    "payload",
    [
        b"chr1\t0010\t0025\nchr10\t000\t004\nchr2\t030\t060\n",
        REFERENCE.replace(b"\n", b"\n\n"),
        b"\xef\xbb\xbfchr1\t0010\t0025\r\n\r\nchr10\t0\t4\r\nchr2\t30\t60",
    ],
)
@pytest.mark.parametrize("reference_side", [False, True])
def test_identical_parsed_intervals_receive_identical_credit(outputs, payload, reference_side):
    reference = payload if reference_side else REFERENCE
    outputs["union_peaks.bed"] = REFERENCE if reference_side else payload
    report = scorer.score_submission(outputs, reference_bed=reference, input_counts=COUNTS)
    assert report.score == 1.0 and report.passed
    assert report.details["exact_reference_match"]


@pytest.mark.parametrize(
    "coordinate", ["1_0", "10.0", "+10", " 10", "\u0661\u0660", "\uff11\uff10"]
)
def test_python_integer_extensions_are_not_valid_bed_coordinates(outputs, coordinate):
    outputs["union_peaks.bed"] = REFERENCE.replace(b"\t10\t", f"\t{coordinate}\t".encode())
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score <= 0.15 and not report.passed
    assert not report.details["exact_reference_match"]
    assert any("non-integer coordinates" in reason for reason in report.reasons)


@pytest.mark.parametrize(
    "reference",
    [
        b"chr1\t10.5\t25\n",
        b"chr1\t25\t10\n",
        b"chr2\t10\t25\nchr1\t10\t25\n",
        b"chr1\t10\t25\nchr1\t20\t30\n",
        b"chr1\t10\t25\nchr1\t10\t25\n",
        b"\xff",
    ],
)
def test_invalid_reference_is_an_evaluator_error(outputs, reference):
    with pytest.raises(RuntimeError, match="reference"):
        scorer.score_submission(outputs, reference_bed=reference, input_counts=COUNTS)


@pytest.mark.parametrize(
    "payload",
    [
        b"chr1\t11\t25\nchr10\t0\t4\nchr2\t30\t60\n",
        b"chr1\t10\t24\nchr10\t0\t4\nchr2\t30\t60\n",
        b"chr1\t10\t25\nchr10\t0\t4\nchr3\t30\t60\n",
    ],
)
def test_same_counts_and_valid_filename_cannot_hide_wrong_intervals(outputs, payload):
    outputs["union_peaks.bed"] = payload
    summary = json.loads(outputs["summary.json"])
    summary["output_file"] = "/output/union_peaks.bed"
    outputs["summary.json"] = json.dumps(summary).encode()
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 0.25 and not report.passed
    assert not report.details["exact_reference_match"]


@pytest.mark.parametrize(
    "payload",
    [
        b"chr1\tNaN\t25\n",
        b"chr1\t10\tInfinity\n",
        b"chr1\t-1\t25\n",
        b"chr1\t25\t25\n",
        b"chr1\t30\t25\n",
        b"chr1 10 25\n",
        b"chr1\t10\t25\textra\n",
        b"\t10\t25\n",
        b"chr1\t10.5\t25\n",
    ],
)
def test_malformed_bed_cannot_receive_format_or_reference_credit(outputs, payload):
    outputs["union_peaks.bed"] = payload
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score <= 0.15 and not report.passed
    assert not report.details["exact_reference_match"]
    assert report.reasons


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b"chr2\t30\t60\nchr1\t10\t25\nchr10\t0\t4\n", "not sorted"),
        (b"chr1\t10\t25\nchr1\t20\t30\nchr2\t30\t60\n", "overlapping"),
        (b"chr1\t10\t25\nchr1\t10\t25\nchr2\t30\t60\n", "overlapping"),
    ],
)
def test_sorting_overlap_and_duplicate_checks_remain_active(outputs, payload, reason):
    outputs["union_peaks.bed"] = payload
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score < 0.25 and not report.passed
    assert any(reason in issue for issue in report.reasons)


@pytest.mark.parametrize("payload", [b"{", b"null", b"[]", b"42", b'"text"'])
def test_malformed_or_nonobject_summary_does_not_receive_summary_credit(outputs, payload):
    outputs["summary.json"] = payload
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 0.9 and not report.passed


@pytest.mark.parametrize(
    "key,value,score",
    [
        ("output_intervals", 4, 0.94),
        ("total_input_intervals", 8, 0.96),
        ("input_interval_counts", {**COUNTS, scorer.INPUT_BED_FILES[0]: 1}, 0.96),
    ],
)
def test_filename_equivalence_does_not_relax_count_checks(outputs, key, value, score):
    summary = json.loads(outputs["summary.json"])
    summary.update(output_file="output/union_peaks.bed")
    summary[key] = value
    outputs["summary.json"] = json.dumps(summary).encode()
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == score and not report.passed


@pytest.mark.parametrize("name", scorer.REQUIRED_FILES)
def test_missing_outputs_still_fail(outputs, name):
    del outputs[name]
    assert scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS).score == 0


def test_command_log_is_never_executed(outputs, tmp_path):
    sentinel = tmp_path / "must-not-exist"
    outputs["commands.sh"] = (
        f"#!/bin/sh\n# bedtools interval workflow\ntouch '{sentinel}'\n".encode()
    )
    report = scorer.score_submission(outputs, reference_bed=REFERENCE, input_counts=COUNTS)
    assert report.score == 1.0
    assert not sentinel.exists()


def test_runtime_and_card_agree_on_filename_label_contract():
    card = json.loads((TASK_DIR / "task_card.json").read_text())
    main_text = (TASK_DIR / "main.py").read_text()
    step = next(line for line in card["taskPrompt"].splitlines() if line.startswith("5. "))
    assert step in main_text
    assert "local POSIX path with that basename" in step
    assert "not an additional output location" in step


def read_intervals(path):
    records = [line.split("\t") for line in path.read_text().splitlines()]
    assert all(len(record) in (3, 10) for record in records)
    rows = [(record[0], int(record[1]), int(record[2])) for record in records]
    assert all(chrom and 0 <= start < end for chrom, start, end in rows)
    return rows


def merge_components(sources, *, touching):
    components = []
    labeled = sorted(
        (*row, 1 << source_index) for source_index, source in enumerate(sources) for row in source
    )
    for chrom, start, end, mask in labeled:
        if (
            components
            and components[-1][0] == chrom
            and (start <= components[-1][2] if touching else start < components[-1][2])
        ):
            previous = components[-1]
            components[-1] = (chrom, previous[1], max(end, previous[2]), previous[3] | mask)
        else:
            components.append((chrom, start, end, mask))
    return components


def coverage_intersection(sources):
    events = defaultdict(list)
    for source_index, source in enumerate(sources):
        for chrom, start, end in source:
            events[chrom].extend([(start, source_index, 1), (end, source_index, -1)])
    result = []
    for chrom in sorted(events):
        active = [0] * len(sources)
        previous = 0
        for position, source_index, change in sorted(events[chrom]):
            if position > previous and all(count > 0 for count in active):
                if result and result[-1][0] == chrom and result[-1][2] == previous:
                    result[-1] = (chrom, result[-1][1], position)
                else:
                    result.append((chrom, previous, position))
            active[source_index] += change
            previous = position
    return result


def test_independent_interval_diagnostics_distinguish_boundaries_labels_and_duplicates():
    sources = [[("chr1", 0, 5), ("chr1", 0, 5)], [("chr1", 5, 10)], [("chr1", 9, 12)]]
    assert merge_components(sources, touching=True) == [("chr1", 0, 12, 7)]
    assert merge_components(sources, touching=False) == [("chr1", 0, 5, 1), ("chr1", 5, 12, 6)]
    assert coverage_intersection(sources) == []
    sources[0].append(("chr1", 8, 11))
    assert coverage_intersection(sources) == [("chr1", 9, 10)]
    assert merge_components([[("chr2", 1, 2), ("chr10", 2, 3)]], touching=True) == [
        ("chr10", 2, 3, 1),
        ("chr2", 1, 2, 1),
    ]


@pytest.fixture(scope="module")
def staged_data():
    if not DATA_DIR.exists():
        pytest.skip("licensed/staged task data is not present")
    manifest = json.loads((DATA_DIR / "reference/source_manifest.json").read_text())
    for name, entry in manifest["raw_files"].items():
        assert (
            hashlib.sha256((DATA_DIR / "input" / name).read_bytes()).hexdigest() == entry["sha256"]
        )
    reference = DATA_DIR / "reference/union_ref.bed"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest["reference"]["sha256"]
    sources = [read_intervals(DATA_DIR / "input" / name) for name in scorer.INPUT_BED_FILES]
    return sources, read_intervals(reference)


def test_staged_integrity_manifest_matches_all_public_input_bytes(staged_data):
    expected = json.loads((DATA_DIR / "reference/input_hashes.json").read_text())
    actual = {
        path.relative_to(DATA_DIR / "input").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (DATA_DIR / "input").rglob("*")
        if path.is_file()
    }
    assert expected == actual


def test_independent_recomputation_does_not_substitute_a_convenient_gold(staged_data):
    sources, reference = staged_data
    assert list(map(len, sources)) == [15296, 34578, 34816]
    assert [len(source) - len(set(source)) for source in sources] == [0, 39, 130]
    components = merge_components(sources, touching=True)
    consensus = [row[:3] for row in components if row[3] == 7]
    intersection = coverage_intersection(sources)
    assert (len(components), len(consensus), len(intersection), len(reference)) == (
        36175,
        10984,
        11016,
        10984,
    )
    assert len(set(consensus) & set(reference)) == 10984
    assert consensus == reference and intersection != reference
    diagnostic_bed = "".join(
        f"{chrom}\t{start}\t{end}\n" for chrom, start, end in consensus
    ).encode()
    assert (
        hashlib.sha256(diagnostic_bed).hexdigest()
        == "e68ec0faf522c063ed4be56c3541ff116a0ba2d4ac3dbbb84ce68d5101bf732f"
    )


def test_source_gold_satisfies_visible_all_input_support_requirement(staged_data):
    sources, reference = staged_data
    unsupported = []
    for source_index, source in enumerate(sources):
        merged = merge_components([source], touching=True)
        cursor = 0
        for chrom, start, end in reference:
            while cursor < len(merged) and (
                merged[cursor][0] < chrom
                or (merged[cursor][0] == chrom and merged[cursor][2] <= start)
            ):
                cursor += 1
            if not (
                cursor < len(merged)
                and merged[cursor][0] == chrom
                and merged[cursor][1] < end
                and merged[cursor][2] > start
            ):
                unsupported.append((scorer.INPUT_BED_FILES[source_index], chrom, start, end))
    assert not unsupported, f"{len(unsupported)} missing input overlaps: {unsupported}"
