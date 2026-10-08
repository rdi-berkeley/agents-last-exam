"""Original rubric applied to independently replayed physical observations."""

import csv
import io
import math
import re

import numpy as np

from .verification import (
    CASES,
    DERIVED,
    REQUIRED_IMAGES,
    RESPONSES,
    parse_json,
    read_answer,
    read_csv,
)


def reference_answer(cases):
    if set(cases) != set(CASES):
        raise ValueError("All five independently replayed cases are required")
    result = {}
    minima = {}
    for quantity in ("settlement", "disp"):
        values = [float(cases[case][f"max_{quantity}_mm"]) for case in CASES]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Replay contains nonpositive or nonfinite response magnitudes")
        for case, value in zip(CASES, values):
            result[f"{case}_max_{quantity}_mm"] = value
        minimum = min(values)
        minima[f"best_support_position_for_minimum_{quantity}_m"] = [
            position for position, value in enumerate(values) if value == minimum
        ]
    settlement = [result[f"{case}_max_settlement_mm"] for case in CASES]
    result[DERIVED[0]] = round((settlement[0] - settlement[2]) / settlement[0] * 100, 1)
    result[DERIVED[1]] = round((settlement[4] - settlement[2]) / settlement[2] * 100, 1)
    result.update({key: positions[0] for key, positions in minima.items()})
    return result, minima


def normalize_monitor_table(content, suffix):
    if suffix == ".json":
        data = parse_json(content)
        return {table: data[table] for table in ("stations", "profiles")}
    reader = csv.DictReader(io.StringIO(content), delimiter="\t" if suffix == ".tsv" else ",")
    required = {"monitor", "x_m", "y_m", "z_m", "ux_mm", "uy_mm", "uz_mm"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Unsupported raw monitor table columns")
    result = {"stations": [], "profiles": []}
    for row in reader:
        name = row["monitor"]
        table = "stations" if name.startswith("DBC") else "profiles"
        result[table].append(
            {
                "monitor": name,
                "xyz_m": [float(row[f"{axis}_m"]) for axis in "xyz"],
                "u_m": [float(row[f"u{axis}_mm"]) / 1000 for axis in "xyz"],
            }
        )
    return result


def compare_monitors(reported, native):
    """Allow different sampling densities while checking full physical profiles."""
    for table, prefix in (("stations", "DBC"), ("profiles", "CX")):
        rows = reported.get(table)
        if not isinstance(rows, list) or {row.get("monitor") for row in rows} != {
            f"{prefix}{number}" for number in range(1, 11)
        }:
            raise ValueError(f"Missing or wrong physical {prefix} monitor")
        for number in range(1, 11):
            name = f"{prefix}{number}"
            expected = [row for row in native[table] if row["monitor"] == name]
            actual = [row for row in rows if row["monitor"] == name]
            points = np.asarray([row["xyz_m"] for row in actual], dtype=float)
            values = np.asarray([row["u_m"] for row in actual], dtype=float)
            if (
                points.shape != (len(actual), 3)
                or values.shape != points.shape
                or not np.isfinite(points).all()
                or not np.isfinite(values).all()
                or not np.allclose(points[:, :2], expected[0]["xyz_m"][:2], rtol=0, atol=1e-6)
            ):
                raise ValueError(f"Wrong reported location or nonfinite values at {name}")
            expected_points = np.asarray([row["xyz_m"] for row in expected])
            expected_values = np.asarray([row["u_m"] for row in expected])
            if table == "stations":
                if len(actual) != 1 or abs(points[0, 2] - expected_points[0, 2]) > 1e-6:
                    raise ValueError(f"Wrong station height or duplicate at {name}")
                targets = expected_values
            else:
                order = np.argsort(expected_points[:, 2], kind="stable")
                levels = expected_points[order, 2]
                ordered_values = expected_values[order]
                if (
                    abs(points[:, 2].min() - levels[0]) > 1e-6
                    or abs(points[:, 2].max() - levels[-1]) > 1e-6
                ):
                    raise ValueError(f"Incomplete wall top-to-toe coverage at {name}")
                targets = []
                for elevation in points[:, 2]:
                    matches = expected_values[np.abs(expected_points[:, 2] - elevation) < 1e-8]
                    if not len(matches):
                        lower = np.flatnonzero(levels < elevation)[-1]
                        upper = np.flatnonzero(levels > elevation)[0]
                        fraction = (elevation - levels[lower]) / (levels[upper] - levels[lower])
                        matches = np.array(
                            [
                                (1 - fraction) * ordered_values[lower]
                                + fraction * ordered_values[upper]
                            ]
                        )
                    targets.append(matches)
                observed_maximum = np.linalg.norm(values[:, :2], axis=1).max() * 1000
                native_maximum = np.linalg.norm(expected_values[:, :2], axis=1).max() * 1000
                if abs(observed_maximum - native_maximum) > max(0.30, 0.05 * native_maximum):
                    raise ValueError(f"Reported profile misses the native maximum at {name}")
            if table == "stations":
                targets = [targets]
            for value, alternatives in zip(values, targets):
                tolerance = np.maximum(0.30 / 1000, 0.05 * np.abs(alternatives))
                if not np.any(np.all(np.abs(value - alternatives) <= tolerance, axis=1)):
                    raise ValueError(f"Reported physical displacement disagrees at {name}")
    return True


def assess_presentation(files, cases):
    """Apply the disclosed PNG and structured-memo checks after native replay."""
    from PIL import Image

    images = {}
    for name in REQUIRED_IMAGES:
        try:
            with Image.open(io.BytesIO(files[name])) as picture:
                picture.verify()
            with Image.open(io.BytesIO(files[name])) as picture:
                valid = picture.format == "PNG" and picture.width >= 320 and picture.height >= 240
                extrema = picture.convert("RGB").getextrema() if valid else ()
                images[name] = bool(valid and any(high - low >= 16 for low, high in extrema))
        except (OSError, ValueError, Image.DecompressionBombError):
            images[name] = False
    memo_checks = dict.fromkeys(
        (
            "comparisons",
            "minima",
            "recommendation",
            "assumptions",
            "limitations",
            "installation_and_rerun",
        ),
        False,
    )
    try:
        memo = files["support_position_engineering_memo.md"].decode("utf-8")
        blocks = re.findall(
            r"^```support-decision\s*\n(.*?)^```\s*$", memo, re.MULTILINE | re.DOTALL
        )
        if len(blocks) != 1:
            raise ValueError("Memo needs one support-decision JSON block")
        decision = parse_json(blocks[0])
        reference, minima = reference_answer(cases)
        memo_checks["comparisons"] = all(
            type(decision.get(key)) in (int, float)
            and math.isfinite(decision[key])
            and abs(decision[key] - reference[key]) <= max(0.3, 0.08 * abs(reference[key]))
            for key in DERIVED[:2]
        )
        memo_checks["minima"] = all(
            type(decision.get(key)) is int and decision[key] in minima[key] for key in DERIVED[2:]
        )
        memo_checks["recommendation"] = (
            type(decision.get("recommended_position_m")) is int
            and decision["recommended_position_m"] in range(5)
            and isinstance(decision.get("recommendation_reason"), str)
            and len(decision["recommendation_reason"].strip()) >= 40
        )
        for key in ("assumptions", "limitations"):
            memo_checks[key] = (
                isinstance(decision.get(key), str) and len(decision[key].strip()) >= 40
            )
        memo_checks["installation_and_rerun"] = all(
            isinstance(decision.get(key), str) and len(decision[key].strip()) >= 40
            for key in ("installation_tradeoff", "rerun_instructions")
        )
    except (UnicodeDecodeError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return {
        "images_fraction": sum(images.values()) / len(images),
        "memo_fraction": sum(memo_checks.values()) / len(memo_checks),
        "images": images,
        "memo": memo_checks,
        "scope": "Decoded image and structured reporting checks; no semantic image/prose judgment",
    }


def score_delivery(answer_text, summary_text, cases, engineering):
    reference, minima = reference_answer(cases)
    try:
        answer = read_answer(answer_text)
    except (ValueError, TypeError, KeyError):
        return {"score": 0.0, "reason": "Invalid answer schema", "numeric_subscore": 0.0}
    fields = {}
    reported_derived, _ = reference_answer(
        {
            case: {
                f"max_{quantity}_mm": answer[f"{case}_max_{quantity}_mm"]
                for quantity in ("settlement", "disp")
            }
            for case in CASES
        }
    )
    for name in RESPONSES + DERIVED:
        weight = 0.0359 if name == DERIVED[-1] else 0.0357
        if name in minima:
            passed = answer[name] in minima[name]
        else:
            tolerance = max(
                0.30, (0.08 if name.endswith("_percent") else 0.05) * abs(reference[name])
            )
            passed = abs(answer[name] - reference[name]) <= tolerance
            if name.endswith("_percent"):
                passed = passed and abs(answer[name] - reported_derived[name]) <= 0.100000001
        fields[name] = {"passed": bool(passed), "weight": weight}
    try:
        read_csv(summary_text, answer)
        summary_passed = True
    except (ValueError, TypeError, KeyError):
        summary_passed = False
    if not engineering.get("required_files_present"):
        return {"score": 0.0, "reason": "Missing original deliverable", "numeric_subscore": 0.0}
    numeric = sum(row["weight"] for row in fields.values() if row["passed"])
    components = {
        "schema": 0.08,
        "completeness": 0.12,
        "native_project": 0.06 if engineering.get("native_results_agree") else 0.0,
        "summary": 0.08 if summary_passed and engineering.get("raw_monitors_agree") else 0.0,
        "images": 0.10 * engineering["images_fraction"],
        "memo": 0.06 * engineering["memo_fraction"],
    }
    return {
        "score": round(numeric + sum(components.values()), 4),
        "numeric_subscore": round(numeric, 4),
        "engineering_components": components,
        "fields": fields,
        "reference": reference,
        "equal_native_minima": minima,
        "authority": "Independent native replay of this submitted source-compliant family",
    }
