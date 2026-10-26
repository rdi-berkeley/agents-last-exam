from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import NormalDist

PRIMARY_METRIC = "opened_rate"
SECONDARY_METRICS = ["clicked_rate", "converted_rate", "unsubscribed_rate"]
METRIC_COLUMN_MAP = {
    "opened_rate": "opened",
    "clicked_rate": "clicked",
    "converted_rate": "converted",
    "unsubscribed_rate": "unsubscribed",
}
Z_975 = NormalDist().inv_cdf(0.975)
POWER_SAMPLE_SIZE_LOWER_TOLERANCE = 2
ASSIGNMENT_WEIGHT = 0.2
RESULT_METRIC_WEIGHT = 0.15
REPORT_WEIGHT = 0.2
UNDEFINED_NUMERIC_VALUES = {"", "na", "n/a", "nan", "none", "null", "undefined"}
RECOMMENDATION_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?recommendation\s*:?(?:\*\*|__)?\s*(.*)$",
    re.IGNORECASE,
)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
RECOMMENDATION_TOKEN_RE = re.compile(r"\b(ship|hold)\b", re.IGNORECASE)
NEGATED_TOKEN_RE = re.compile(
    r"(?:\b(?:not|never|cannot|can't|won't|avoid|without)\b|\brather\s+than\b|"
    r"\binstead\s+of\b)(?:\s+(?:yet|currently|immediately|now|safely))?\s*$",
    re.IGNORECASE,
)
LIFT_UNITS = (
    r"percentage[ -]+points?|percent[ -]+points?|p\.?p\.?|"
    r"basis[ -]+points?|bps?|percent|%"
)
LIFT_UNIT_RE = re.compile(
    rf"\s*(?P<unit>{LIFT_UNITS})(?!\w)",
    re.IGNORECASE,
)
REPORT_NUMBER_RE = re.compile(
    r"(?<![\w.,+-])(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))"
    rf"(?:\.\d+)?(?:e[+-]?\d+)?)(?![\d_]|,\d|\.\d)(?=$|[^\w]|(?:{LIFT_UNITS})(?!\w))",
    re.IGNORECASE,
)
LIFT_LABEL_RE = re.compile(
    r"\b(?:(?P<kind>absolute|relative)\s+)?(?:lift|difference|delta)\b",
    re.IGNORECASE,
)
PRIMARY_CONTEXT_RE = re.compile(r"\b(?:primary|open(?:ed|ing)?(?:[_ -]?rate)?)\b", re.IGNORECASE)
OTHER_METRIC_RE = re.compile(
    r"\b(?:click\w*|unsubscrib\w*|convert\w*|conversion|revenue|income|sales)\b",
    re.IGNORECASE,
)
SAMPLE_LABEL_RE = re.compile(r"\b(?:sample\s+size|sample\s+per\s+(?:arm|group))\b", re.IGNORECASE)
SAMPLE_REQUIREMENT_RE = re.compile(
    r"\b(?:requir(?:e[ds]?|ing)|need(?:s|ed)?|minimum|target|planned)\b", re.IGNORECASE
)
PER_ARM_RE = re.compile(
    r"\b(?:per[\s-]+(?:arm|group)|(?:in|for)\s+each\s+(?:arm|group))\b", re.IGNORECASE
)
NONCURRENT_CLAIM_RE = re.compile(
    r"\b(?:previous|historical|earlier|prior|hypothetical|target|goal|minimum|"
    r"expected|anticipated)\s+(?:(?:primary|absolute|observed|open(?:ed)?(?:[ _-]rate)?)\s+)*$",
    re.IGNORECASE,
)
NONCURRENT_SAMPLE_CLAIM_RE = re.compile(
    r"\b(?:previous|historical|earlier|prior|hypothetical)\s+"
    r"(?:(?:required|per[ -]arm)\s+)*(?:sample\s+size|requirement)\b|"
    r"\bpreviously\s+(?:required|needed)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-report", action="store_true")
    parser.add_argument("--report-byte-limit", type=int)
    return parser.parse_args()


def to_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_bool(value: str | None) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def read_csv_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sample_size_required(historical_metrics: Path) -> int:
    rows = read_csv_rows(historical_metrics)
    baseline = sum(float(row["open_rate"]) for row in rows) / len(rows)
    mde = 0.03
    alpha = 0.05
    power = 0.80
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    p1 = baseline
    p2 = baseline + mde
    p_bar = (p1 + p2) / 2
    n = (
        (
            z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        ** 2
    ) / (mde**2)
    return math.ceil(n)


def sample_size_bounds(historical_metrics: Path, required_n: int) -> tuple[int, int]:
    rows = read_csv_rows(historical_metrics)
    baseline = sum(float(row["open_rate"]) for row in rows) / len(rows)
    treatment = baseline + 0.03
    average = (baseline + treatment) / 2
    pooled_variance = 2 * average * (1 - average)
    unpooled_variance = baseline * (1 - baseline) + treatment * (1 - treatment)
    z_sum = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
    effect = 2 * (math.asin(math.sqrt(treatment)) - math.asin(math.sqrt(baseline)))
    estimates = (
        required_n - POWER_SAMPLE_SIZE_LOWER_TOLERANCE,
        required_n,
        math.ceil(z_sum**2 * pooled_variance / 0.03**2),
        math.ceil(z_sum**2 * unpooled_variance / 0.03**2),
        math.ceil(2 * z_sum**2 / effect**2),
    )
    return max(0, min(estimates)), max(estimates)


def metric_stats(results_raw: Path) -> dict[str, dict[str, float | bool]]:
    rows = read_csv_rows(results_raw)
    stats: dict[str, dict[str, float | bool]] = {}
    for metric_name, column_name in METRIC_COLUMN_MAP.items():
        control = [
            int(r[column_name]) for r in rows if r["variant"] == "control" and r["delivered"] == "1"
        ]
        treatment = [
            int(r[column_name])
            for r in rows
            if r["variant"] == "treatment" and r["delivered"] == "1"
        ]
        n_c = len(control)
        n_t = len(treatment)
        x_c = sum(control)
        x_t = sum(treatment)
        p_c = x_c / n_c
        p_t = x_t / n_t
        diff = p_t - p_c
        rel_lift = 0.0 if p_c == 0 and p_t == 0 else (diff / p_c * 100.0 if p_c else float("inf"))
        se_unpooled = math.sqrt((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t))
        ci_lower = diff - Z_975 * se_unpooled
        ci_upper = diff + Z_975 * se_unpooled
        pooled = (x_c + x_t) / (n_c + n_t)
        se_pooled = (
            math.sqrt(pooled * (1 - pooled) * ((1 / n_c) + (1 / n_t)))
            if pooled not in {0.0, 1.0}
            else 0.0
        )
        z_stat = diff / se_pooled if se_pooled else 0.0
        p_val = 2 * (1 - NormalDist().cdf(abs(z_stat)))
        stats[metric_name] = {
            "control_rate": p_c,
            "treatment_rate": p_t,
            "absolute_lift": diff,
            "relative_lift_pct": rel_lift,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "z_statistic": z_stat,
            "p_value_raw": p_val,
            "significant_at_05": p_val < 0.05,
            "is_primary": metric_name == PRIMARY_METRIC,
        }
    return stats


def bh_correct(p_values: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    metric_count = len(ordered)
    last_passing_rank = max(
        (
            rank
            for rank, (_, p_value) in enumerate(ordered, 1)
            if p_value <= rank / metric_count * 0.05
        ),
        default=0,
    )
    out: dict[str, dict[str, float | bool]] = {}
    for rank, (metric, _) in enumerate(ordered, start=1):
        threshold = rank / metric_count * 0.05
        out[metric] = {
            "bh_rank": float(rank),
            "bh_threshold": threshold,
            "bh_significant": rank <= last_passing_rank,
        }
    return out


def floats_close(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) <= tol


def numeric_value_matches(
    value: str | None,
    expected: float,
    *,
    tol: float = 0.001,
    allow_undefined: bool = False,
) -> bool:
    text = str(value or "").strip().lower()
    if text in UNDEFINED_NUMERIC_VALUES:
        return allow_undefined
    try:
        observed = float(text)
        return math.isfinite(observed) and floats_close(observed, expected, tol=tol)
    except ValueError:
        return False


def rounding_allowance(value: str, cap: float, *, normalize: bool = False) -> float:
    try:
        number = Decimal(value.strip())
        decimal_tuple = number.as_tuple()
        exponent = decimal_tuple.exponent
    except (AttributeError, InvalidOperation, ValueError):
        return 0.0
    if not isinstance(exponent, int):
        return 0.0
    if normalize:
        if number.is_zero():
            exponent = 0
        else:
            for digit in reversed(decimal_tuple.digits):
                if digit:
                    break
                exponent += 1
    return min(cap, 0.5 * 10.0 ** max(-16, min(4, exponent))) + 1e-12


def report_has_acceptable_sample_size(
    text: str, required_n: int, sample_bounds: tuple[int, int] | None = None
) -> bool:
    minimum_n, maximum_n = sample_bounds or (
        max(0, required_n - POWER_SAMPLE_SIZE_LOWER_TOLERANCE),
        required_n,
    )
    for match in REPORT_NUMBER_RE.finditer(unicodedata.normalize("NFKC", text)):
        value = float(match["number"].replace(",", ""))
        if math.isfinite(value) and value.is_integer() and minimum_n <= value <= maximum_n:
            return True
    return False


def sample_size_is_per_arm(clause: str, number: re.Match, section: str) -> bool:
    before = clause[: number.start()]
    after = clause[number.end() :]
    if re.search(r"\b(?:not|never|isn't|wasn't|cannot|can't)\b[^.;]{0,40}$", before, re.IGNORECASE):
        return False
    if re.match(r"\s*(?:total|overall)\b", after, re.IGNORECASE):
        return False
    return bool(
        PER_ARM_RE.search(before)
        or PER_ARM_RE.search(section)
        or re.match(
            r"\s*(?:delivered\s+)?(?:emails?|subscribers?|customers?|participants?|"
            r"samples?|observations?)?\s*(?:per|in each|for each)\s+(?:arm|group)\b",
            after,
            re.IGNORECASE,
        )
    )


def report_has_required_sample_size(
    text: str, required_n: int, sample_bounds: tuple[int, int] | None = None
) -> bool:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[`*\"\u201c\u201d]", "", text)
    section = ""
    candidates = []
    claims = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = paragraph.splitlines()
        if lines and MARKDOWN_HEADING_RE.match(lines[0]):
            section = lines.pop(0)
        if not lines:
            continue
        segments = (
            lines
            if any("|" in line for line in lines)
            else [
                " ".join(part.splitlines()) for part in re.split(r"\n\s*[-*]\s+", "\n".join(lines))
            ]
        )
        candidates.extend((section, segment) for segment in segments)
    for section, segment in candidates:
        for clause in re.split(r";|(?<=[.!?])\s+(?=[A-Z])", segment):
            labels = list(SAMPLE_LABEL_RE.finditer(clause))
            for label in labels:
                prefix = clause[: label.start()]
                if re.search(r"\b(?:observed|actual|achieved|total)\s*$", prefix, re.IGNORECASE):
                    continue
                number = REPORT_NUMBER_RE.search(clause, label.end())
                if number and sample_size_is_per_arm(clause, number, section):
                    claims.append((number, clause))
            requirements = list(SAMPLE_REQUIREMENT_RE.finditer(clause))
            if re.search(r"\b(?:power|sample)\b", section, re.IGNORECASE):
                requirements.extend(re.finditer(r"\brequirement\b", clause, re.IGNORECASE))
                requirements.extend(re.finditer(r"\brounds?\s+up\s+to\b", clause, re.IGNORECASE))
            if re.search(r"\b(?:power|sample)\b", f"{section} {clause}", re.IGNORECASE):
                requirements.extend(re.finditer(r"\bn\s*[:=]\s*", clause, re.IGNORECASE))
            for requirement in requirements:
                if re.fullmatch(r"n\s*[:=]\s*", requirement.group(), re.IGNORECASE):
                    number = REPORT_NUMBER_RE.match(clause, requirement.end())
                else:
                    number = REPORT_NUMBER_RE.search(clause, requirement.end())
                if number and sample_size_is_per_arm(clause, number, section):
                    claims.append((number, clause))
    found_correct = False
    found_conflict = False
    for number, clause in claims:
        if NONCURRENT_SAMPLE_CLAIM_RE.search(clause[: number.start()]):
            continue
        value = float(number.group().replace(",", ""))
        if math.isfinite(value) and value.is_integer():
            if report_has_acceptable_sample_size(number.group(), required_n, sample_bounds):
                found_correct = True
            else:
                found_conflict = True
    return found_correct and not found_conflict


def report_has_primary_lift(text: str, expected_lift: float) -> bool:
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[`*\"\u201c\u201d]", "", text)
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
    candidates = []
    for paragraph in re.split(r"\n\s*\n", text):
        prose = []
        table_header = None
        for line in paragraph.splitlines():
            if "|" not in line:
                prose.append(line.strip())
                continue
            cells = next(csv.reader([line.strip().strip("|")], delimiter="|"))
            cells = [cell.strip() for cell in cells]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            if table_header is None:
                table_header = cells
                continue
            if len(cells) != len(table_header):
                continue
            row_context = " ".join(cells)
            if re.search(r"open|primary", row_context, re.IGNORECASE):
                for header, cell in zip(table_header, cells):
                    if LIFT_LABEL_RE.search(header):
                        candidates.append(f"Primary {header}: {cell}")
            for cell in cells:
                if LIFT_LABEL_RE.search(cell):
                    candidates.append(" ".join(cells))
                    break
        candidates.extend(re.split(r";|(?<=[.!?])\s+(?=[A-Z])", " ".join(prose)))

    found_correct = False
    found_conflict = False
    for candidate in candidates:
        direction = re.search(
            r"\b(?P<verb>increased|improved|rose|decreased|declined|dropped|fell)\b"
            r"(?P<before>[^.;]{0,60}?)\bby\s+",
            candidate,
            re.IGNORECASE,
        )
        if direction and PRIMARY_CONTEXT_RE.search(candidate[: direction.end()]):
            nearby_other = OTHER_METRIC_RE.search(direction["before"])
            relative_amount = re.match(
                r"\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:%|percent)\s+relativ",
                candidate[direction.end() :],
                re.IGNORECASE,
            )
            if not nearby_other and not relative_amount:
                sign = (
                    "-"
                    if direction["verb"].lower() in {"decreased", "declined", "dropped", "fell"}
                    else ""
                )
                candidate += f"; Primary absolute lift: {sign}{candidate[direction.end() :]}"
        comparison = re.search(
            r"\b(higher|lower)\s+than\s+(?:the\s+)?control\b", candidate, re.IGNORECASE
        )
        if comparison:
            prefix = candidate[: comparison.start()]
            primary = list(PRIMARY_CONTEXT_RE.finditer(prefix))
            other = list(OTHER_METRIC_RE.finditer(prefix))
            numbers = list(REPORT_NUMBER_RE.finditer(prefix))
            if primary and numbers and (not other or other[-1].start() < primary[-1].start()):
                number = numbers[-1]
                unit = LIFT_UNIT_RE.match(prefix, number.end())
                if (
                    primary[-1].end() <= number.start()
                    and unit
                    and not prefix[unit.end() :].strip()
                    and not re.search(
                        r"\brelative(?:ly)?\b", prefix[primary[-1].start() :], re.IGNORECASE
                    )
                ):
                    sign = "-" if comparison.group(1).lower() == "lower" else ""
                    candidate += f"; Primary absolute lift: {sign}{number.group()}{unit.group()}"
        labels = list(LIFT_LABEL_RE.finditer(candidate))
        for index, label in enumerate(labels):
            if (label["kind"] or "").lower() == "relative":
                continue
            prefix = candidate[: label.start()]
            if re.search(r"\b(?:ci|confidence interval)\s+(?:for\s+)?$", prefix, re.IGNORECASE):
                continue
            primary = list(PRIMARY_CONTEXT_RE.finditer(prefix))
            other = list(OTHER_METRIC_RE.finditer(prefix))
            if other and (not primary or other[-1].start() > primary[-1].start()):
                continue
            end = labels[index + 1].start() if index + 1 < len(labels) else len(candidate)
            quantity = candidate[label.end() : end]
            match = REPORT_NUMBER_RE.search(quantity)
            if match is None:
                continue
            if re.search(
                r"\b(?:not|never|isn't|wasn't)\b", quantity[: match.start()], re.IGNORECASE
            ):
                continue
            value = float(match["number"].replace(",", ""))
            unit_match = LIFT_UNIT_RE.match(quantity, match.end())
            if unit_match is None:
                unit_match = re.search(
                    rf"(?<!\w)(?P<unit>{LIFT_UNITS})(?!\w)",
                    quantity[: match.start()],
                    re.IGNORECASE,
                )
            unit = unit_match["unit"].lower() if unit_match else ""
            if unit.startswith(("basis", "bp")):
                value /= 10_000
            elif unit:
                value /= 100
            if not math.isfinite(value):
                continue
            if NONCURRENT_CLAIM_RE.search(prefix):
                continue
            if math.isclose(value, expected_lift, rel_tol=0, abs_tol=0.00005):
                found_correct = True
            elif primary:
                found_conflict = True
    return found_correct and not found_conflict


def extract_recommendation(text: str) -> str | None:
    lines = text.splitlines()
    decisions = set()
    for index, line in enumerate(lines):
        heading = RECOMMENDATION_HEADING_RE.match(line)
        if heading is None:
            continue

        section_lines = []
        inline_text = heading.group(1).strip()
        if inline_text:
            section_lines.append(inline_text)
        for following in lines[index + 1 :]:
            if MARKDOWN_HEADING_RE.match(following):
                break
            if not following.strip():
                continue
            section_lines.append(following.strip())

        section = re.sub(r"\*\*|__|[`*]", "", " ".join(section_lines))
        for token in RECOMMENDATION_TOKEN_RE.finditer(section):
            decision = token.group(1).lower()
            prefix = section[max(0, token.start() - 48) : token.start()]
            if NEGATED_TOKEN_RE.search(prefix):
                decision = "hold" if decision == "ship" else "ship"
            decisions.add(decision)
    return decisions.pop() if len(decisions) == 1 else None


def validate_metric_row(
    metric: str,
    row: dict[str, str],
    expected: dict[str, float | bool],
    bh_expected: dict[str, dict[str, float | bool]],
) -> bool:
    try:
        if parse_bool(row["is_primary"]) is not bool(expected["is_primary"]):
            return False
        for key in (
            "control_rate",
            "treatment_rate",
            "absolute_lift",
            "ci_lower_95",
            "ci_upper_95",
        ):
            if not numeric_value_matches(row[key], float(expected[key])):
                return False
            if key in {"control_rate", "treatment_rate"} and not 0 <= float(row[key]) <= 1:
                return False

        if float(row["ci_lower_95"]) > float(row["ci_upper_95"]):
            return False
        rate_fields = ("control_rate", "treatment_rate", "absolute_lift")
        rounding_error = sum(rounding_allowance(row[key], 0.001) for key in rate_fields)
        if (
            abs(
                float(row["treatment_rate"])
                - float(row["control_rate"])
                - float(row["absolute_lift"])
            )
            > rounding_error
        ):
            return False

        control_rate = float(expected["control_rate"])
        treatment_rate = float(expected["treatment_rate"])
        if row["relative_lift_pct"] is None:
            return False
        if control_rate == 0.0 and treatment_rate > 0.0:
            relative_text = row["relative_lift_pct"].strip().lower()
            if relative_text not in UNDEFINED_NUMERIC_VALUES:
                try:
                    relative_value = float(relative_text)
                except (TypeError, ValueError):
                    return False
                if relative_value != math.inf:
                    return False
        elif not numeric_value_matches(
            row["relative_lift_pct"],
            float(expected["relative_lift_pct"]),
            tol=0.05,
            allow_undefined=control_rate == 0.0,
        ):
            return False
        degenerate = control_rate == treatment_rate and control_rate in {0.0, 1.0}
        z_statistic = float(expected["z_statistic"])
        if not any(
            numeric_value_matches(row["z_statistic"], value, tol=0.01, allow_undefined=degenerate)
            for value in (z_statistic, -z_statistic)
        ):
            return False
        if not numeric_value_matches(
            row["p_value_raw"],
            float(expected["p_value_raw"]),
            tol=rounding_allowance(row["p_value_raw"], 0.001, normalize=True),
            allow_undefined=degenerate,
        ):
            return False
        if row["p_value_raw"].strip().lower() not in UNDEFINED_NUMERIC_VALUES and not (
            0 <= float(row["p_value_raw"]) <= 1
        ):
            return False
        if parse_bool(row["significant_at_05"]) is not bool(expected["significant_at_05"]):
            return False
        if metric == PRIMARY_METRIC:
            return not any(
                str(row[key]).strip() for key in ["bh_rank", "bh_threshold", "bh_significant"]
            )

        bh = bh_expected[metric]
        if (
            not numeric_value_matches(row["bh_rank"], float(bh["bh_rank"]))
            or not float(row["bh_rank"]).is_integer()
        ):
            return False
        if not numeric_value_matches(row["bh_threshold"], float(bh["bh_threshold"])) or not (
            0 <= float(row["bh_threshold"]) <= 1
        ):
            return False
        return parse_bool(row["bh_significant"]) is bool(bh["bh_significant"])
    except (KeyError, TypeError, ValueError):
        return False


def validate_result_rows(
    output_path: Path, expected_stats: dict[str, dict[str, float | bool]]
) -> dict[str, bool]:
    checks = dict.fromkeys(METRIC_COLUMN_MAP, False)
    try:
        with output_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
                return checks
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return checks
    if len(rows) != 4:
        return checks
    by_metric = {row.get("metric"): row for row in rows}
    if set(by_metric) != set(METRIC_COLUMN_MAP):
        return checks

    bh_expected = bh_correct(
        {metric: expected_stats[metric]["p_value_raw"] for metric in SECONDARY_METRICS}
    )
    for p_value in {expected_stats[metric]["p_value_raw"] for metric in SECONDARY_METRICS}:
        tied = [
            metric
            for metric in SECONDARY_METRICS
            if expected_stats[metric]["p_value_raw"] == p_value
        ]
        if len(tied) < 2:
            continue
        try:
            ranks = {metric: float(by_metric[metric]["bh_rank"]) for metric in tied}
            valid_ranks = {float(bh_expected[metric]["bh_rank"]) for metric in tied}
            if set(ranks.values()) != valid_ranks:
                continue
            for metric, rank in ranks.items():
                bh_expected[metric]["bh_rank"] = rank
                bh_expected[metric]["bh_threshold"] = rank / len(SECONDARY_METRICS) * 0.05
        except (KeyError, TypeError, ValueError):
            continue
    for metric, expected in expected_stats.items():
        row = by_metric[metric]
        if None not in row and None not in row.values():
            checks[metric] = validate_metric_row(metric, row, expected, bh_expected)
    return checks


def validate_results_tsv(
    output_path: Path, expected_stats: dict[str, dict[str, float | bool]]
) -> bool:
    return all(validate_result_rows(output_path, expected_stats).values())


def validate_assignment_csv(output_path: Path, results_raw: Path) -> bool:
    try:
        with output_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if len(reader.fieldnames or []) != 2 or set(reader.fieldnames) != {"metric", "value"}:
                return False
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return False
    if any(None in row or None in row.values() for row in rows):
        return False
    try:
        values = {row["metric"]: row["value"] for row in rows}
    except (KeyError, TypeError):
        return False
    required = {"n_control", "n_treatment", "ratio", "srm_chi2", "srm_pvalue", "srm_pass"}
    if len(rows) != len(required) or set(values) != required:
        return False
    raw_rows = read_csv_rows(results_raw)
    n_control = sum(row["variant"] == "control" for row in raw_rows)
    n_treatment = sum(row["variant"] == "treatment" for row in raw_rows)
    if not n_control or not n_treatment:
        return False
    chi2 = (n_treatment - n_control) ** 2 / (n_control + n_treatment)
    pvalue = math.erfc(math.sqrt(chi2 / 2))
    try:
        for key, expected in [("n_control", n_control), ("n_treatment", n_treatment)]:
            observed = float(values[key])
            if not math.isfinite(observed) or not observed.is_integer() or observed != expected:
                return False
        if not numeric_value_matches(values["ratio"], n_treatment / n_control):
            return False
        if float(values["srm_chi2"]) < 0 or not numeric_value_matches(values["srm_chi2"], chi2):
            return False
        if not 0 <= float(values["srm_pvalue"]) <= 1 or not numeric_value_matches(
            values["srm_pvalue"], pvalue
        ):
            return False
    except (TypeError, ValueError):
        return False
    flag = values["srm_pass"].strip().lower()
    if flag not in {"true", "false", "1", "0", "yes", "no"}:
        return False
    return to_bool(flag) == (pvalue > 0.01)


def validate_report_md(
    output_path: Path,
    expected_stats: dict[str, dict[str, float | bool]],
    required_n: int,
    sample_bounds: tuple[int, int] | None = None,
) -> bool:
    try:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    guardrail_lift_pp = expected_stats["unsubscribed_rate"]["absolute_lift"] * 100
    guardrail_pass = guardrail_lift_pp <= 0.5
    primary_sig = bool(expected_stats[PRIMARY_METRIC]["significant_at_05"])
    expected_recommendation = "ship" if primary_sig and guardrail_pass else "hold"
    if extract_recommendation(text) != expected_recommendation:
        return False
    if not report_has_required_sample_size(text, required_n, sample_bounds):
        return False
    return report_has_primary_lift(text, float(expected_stats[PRIMARY_METRIC]["absolute_lift"]))


def component_scores(
    assignment_ok: bool, result_checks: dict[str, bool], report_ok: bool
) -> dict[str, float]:
    scores = {"assignment": ASSIGNMENT_WEIGHT if assignment_ok else 0.0}
    scores.update(
        {
            f"results_{metric}": RESULT_METRIC_WEIGHT if ok else 0.0
            for metric, ok in result_checks.items()
        }
    )
    scores["report"] = REPORT_WEIGHT if report_ok else 0.0
    return scores


def compare_to_reference(output_path: Path, reference_path: Path) -> bool:
    return (
        output_path.read_text(encoding="utf-8", errors="replace").strip()
        == reference_path.read_text(encoding="utf-8", errors="replace").strip()
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    payload = {"score": 0.0}

    required_files = {
        "assignment": output_dir / "randomization_assignment.csv",
        "results": output_dir / "experiment_results.tsv",
        "report": output_dir / "experiment_report.md",
    }
    payload["missing_files"] = [name for name, path in required_files.items() if not path.is_file()]

    expected_stats = metric_stats(input_dir / "experiment_results_raw.csv")
    required_n = sample_size_required(input_dir / "historical_metrics.csv")
    sample_bounds = sample_size_bounds(input_dir / "historical_metrics.csv", required_n)
    assignment_ok = required_files["assignment"].is_file() and validate_assignment_csv(
        required_files["assignment"], input_dir / "experiment_results_raw.csv"
    )
    if required_files["results"].is_file():
        result_checks = validate_result_rows(required_files["results"], expected_stats)
    else:
        result_checks = dict.fromkeys(METRIC_COLUMN_MAP, False)
    results_ok = all(result_checks.values())
    report_size = (
        required_files["report"].stat().st_size if required_files["report"].is_file() else 0
    )
    report_ok = (
        not args.external_report
        and required_files["report"].is_file()
        and validate_report_md(required_files["report"], expected_stats, required_n, sample_bounds)
    )
    reference_match = (
        (args.report_byte_limit is None or report_size <= args.report_byte_limit)
        and not payload["missing_files"]
        and all(
            compare_to_reference(required_files[key], reference_dir / required_files[key].name)
            for key in required_files
        )
    )
    scores = component_scores(assignment_ok, result_checks, report_ok)

    payload.update(
        {
            "assignment_ok": assignment_ok,
            "results_ok": results_ok,
            "result_checks": result_checks,
            "report_ok": report_ok,
            "reference_match": reference_match,
            "component_scores": scores,
            "required_n": required_n,
            "sample_bounds": sample_bounds,
            "primary_absolute_lift": expected_stats[PRIMARY_METRIC]["absolute_lift"],
            "expected_recommendation": "ship"
            if expected_stats[PRIMARY_METRIC]["significant_at_05"]
            and expected_stats["unsubscribed_rate"]["absolute_lift"] <= 0.005
            else "hold",
            "report_size": report_size,
            "open_rate_lift_pp": round(expected_stats[PRIMARY_METRIC]["absolute_lift"] * 100, 3),
            "unsubscribe_lift_pp": round(
                expected_stats["unsubscribed_rate"]["absolute_lift"] * 100, 3
            ),
        }
    )
    payload["score"] = round(sum(scores.values()), 10)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
