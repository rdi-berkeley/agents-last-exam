"""
verify_process.py — Secondary evaluation script (runs on remote Windows VM)

Reads a Moldex3D .pro file (INI-style) and compares key process parameters
against a reference JSON. Used for partial credit when simulation results
are incomplete or missing.

Requirements:
    Python 3.x (no extra packages needed)

Usage:
    python verify_process.py --agent-pro PATH_TO_PRO --ref PATH_TO_REF_JSON

Output:
    JSON to stdout, e.g.:
    {"score": 0.75, "matched_params": 6, "total_params": 8, "param_details": {...}}

Exit codes:
    0 = success
    1 = error
"""

import sys
import json
import argparse
import os
import configparser
import re


# Parameters to check and their locations in the .pro file
# Format: (json_key, pro_section, pro_key, tolerance)
PROCESS_PARAMS = [
    ("melt_temperature_C", "FlowCTL", "MeltTemperature", 0.01),
    ("mold_temperature_C", "FlowCTL", "MoldTemperature", 0.01),
    ("injection_time_sec", "Flow-1", "FillTime", 0.05),
    ("vp_switch_volume_pct", "FlowCTL", "VolumeFilled", 0.01),
    ("packing_time_sec", "Pack-1", "PackTime", 0.01),
    ("packing_pressure_MPa", "Pack-1", "PackPres", 0.01),
    ("cooling_time_sec", "Cool", "CoolTime", 0.01),
    ("coolant_temperature_C", "Cool", "CoolantTemp", 0.01),
    ("coolant_flow_rate", "Cool", "CoolantFR", 0.05),
    ("eject_temperature_C", "Cool", "EjectTemp", 0.01),
    ("open_time_sec", "Cool", "OpenTime", 0.01),
]


def parse_pro_file(pro_path):
    """Parse a Moldex3D .pro file (INI-style) into a dict of dicts.

    The .pro file uses INI-like format with [Section] headers and Key = Value lines.
    Some keys have multiple space-separated values.
    """
    sections = {}
    current_section = None

    # Try multiple encodings since Moldex3D files may use GBK
    content = None
    for enc in ["utf-8", "gbk", "latin-1"]:
        try:
            with open(pro_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        raise ValueError(f"Could not decode .pro file: {pro_path}")

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        # Section header
        section_match = re.match(r"\[(.+)\]", line)
        if section_match:
            current_section = section_match.group(1)
            sections[current_section] = {}
            continue

        # Key = Value
        if current_section and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            sections[current_section][key] = value

    return sections


def verify_process(pro_path, ref_path):
    """Compare .pro file parameters against reference JSON."""

    # Load reference
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    # Parse .pro file
    pro_sections = parse_pro_file(pro_path)

    param_details = {}
    matched = 0
    total = 0

    for json_key, pro_section, pro_key, tolerance in PROCESS_PARAMS:
        ref_val = ref_data.get(json_key)
        if ref_val is None:
            continue

        total += 1

        # Get value from .pro file
        agent_val = None
        if pro_section in pro_sections and pro_key in pro_sections[pro_section]:
            raw = pro_sections[pro_section][pro_key]
            # Take first value if space-separated
            try:
                agent_val = float(raw.split()[0])
            except (ValueError, IndexError):
                agent_val = None

        if agent_val is None:
            param_details[json_key] = {"match": False, "detail": "not found in .pro"}
            continue

        ref_float = float(ref_val)
        if ref_float == 0:
            match = abs(agent_val) < 1e-6
        else:
            rel_error = abs(agent_val - ref_float) / abs(ref_float)
            match = rel_error <= tolerance

        param_details[json_key] = {
            "match": match,
            "detail": f"agent={agent_val}, ref={ref_float}, tol={tolerance}",
        }
        if match:
            matched += 1

    score = matched / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "matched_params": matched,
        "total_params": total,
        "param_details": param_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Moldex3D .pro process config vs reference"
    )
    parser.add_argument("--agent-pro", required=True, help="Path to agent's .pro file")
    parser.add_argument("--ref", required=True, help="Path to reference process JSON")
    args = parser.parse_args()

    for path, label in [
        (args.agent_pro, "agent .pro"),
        (args.ref, "reference process JSON"),
    ]:
        if not os.path.exists(path):
            result = {"score": 0.0, "error": f"{label} not found: {path}"}
            print(json.dumps(result))
            sys.exit(1)

    try:
        result = verify_process(args.agent_pro, args.ref)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        result = {"score": 0.0, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
