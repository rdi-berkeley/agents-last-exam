# -*- coding: utf-8 -*-
"""
check_collision.py -- Evaluation gate check (runs on remote Windows VM, PowerMill must be open)

For each toolpath in the agent's PM project, checks whether PowerMill flags it
as having collisions or gouging. If any toolpath fails, the overall gate returns
failure and the agent's score is set to 0.

Strategy:
    1. Open the agent's PM project via COM
    2. Apply collision checking (EDIT COLLISION APPLY) to every toolpath
    3. Export a SetupSheet CSV with per-toolpath collision/gouge status
    4. Parse the CSV to count collision/gouge flags

Note: The exact column names in the SetupSheet CSV export may vary across
PowerMill versions. The current implementation looks for "Collision" and "Gouge"
columns, but this may need adjustment.

Requirements:
    pip install pywin32

Usage:
    python check_collision.py --project PATH_TO_PM_PROJECT

Output (JSON to stdout):
    {"passed": true,  "collision_count": 0, "gouge_count": 0, "details": [...]}
    {"passed": false, "collision_count": 2, "gouge_count": 1, "details": [...]}

Exit codes:
    0 = no collisions/gouges (gate passes)
    2 = collisions/gouges found (gate fails)
    1 = error (cannot connect to PM, project not found, etc.)
"""

import os
import sys
import json
import tempfile
import time
import argparse

try:
    import win32com.client
except ImportError:
    print('{"error": "pywin32 not installed"}')
    sys.exit(1)


def get_pm_connection():
    """Connect to the running PowerMill instance via COM."""
    try:
        return win32com.client.GetActiveObject("pmill.Document")
    except Exception:
        pass
    try:
        pm = win32com.client.Dispatch("pmill.Document")
        time.sleep(5)
        return pm
    except Exception:
        return None


def check_collision(pm_project_path: str) -> dict:
    """Open PM project, apply collision checking, and parse results.

    The macro:
    1. Opens the project
    2. Iterates over all toolpaths, applying collision checking
    3. Exports a SetupSheet CSV containing collision/gouge columns
    4. Parses the CSV for any positive flags

    Args:
        pm_project_path: absolute path to the agent's PM project directory
    Returns:
        dict with 'passed' (bool), 'collision_count', 'gouge_count', 'details'
    """
    pm = get_pm_connection()
    if pm is None:
        return {
            "passed": False,
            "error": "Cannot connect to PowerMill",
            "collision_count": 0,
            "gouge_count": 0,
        }

    # Convert paths for PowerMill macro (forward slashes)
    pm_path_fwd = pm_project_path.replace("\\", "/")
    export_dir = tempfile.mkdtemp().replace("\\", "/")
    export_csv = (export_dir + "/collision_check.csv").replace("\\", "/")

    # PowerMill macro:
    # - Opens the project
    # - Applies collision checking to each toolpath
    # - Exports program info (includes collision status) to CSV
    macro_code = f"""
ECHO OFF
DIALOGS MESSAGE OFF
DIALOGS ERROR OFF

PROJECT OPEN "{pm_path_fwd}"

// Apply collision checking to all toolpaths
FOREACH tp IN folder('Toolpath') {{
    ACTIVATE TOOLPATH $tp.Name
    EDIT COLLISION APPLY
}}

// Export program info (includes per-toolpath collision and gouge flags)
EDIT SETUP_SHEET EXPORT_CNC_PROGRAM_INFO "{export_csv}"

DIALOGS MESSAGE ON
DIALOGS ERROR ON
ECHO ON
"""

    # Write macro with GBK encoding (PowerMill on Chinese Windows)
    temp_mac = os.path.join(tempfile.gettempdir(), "check_collision_temp.mac").replace("\\", "/")
    with open(temp_mac, "w", encoding="gbk") as f:
        f.write(macro_code)

    try:
        pm.Execute(f'MACRO "{temp_mac}"')
    except Exception as e:
        return {
            "passed": False,
            "error": f"Macro failed: {e}",
            "collision_count": 0,
            "gouge_count": 0,
        }

    # ---------------------------------------------------------------------------
    # Parse the exported CSV for collision/gouge flags
    # ---------------------------------------------------------------------------
    collision_count = 0
    gouge_count = 0
    details = []

    csv_path = export_csv.replace("/", "\\")
    if os.path.exists(csv_path):
        try:
            import csv

            with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                print(f"CSV columns found: {columns}", file=sys.stderr)

                # Find collision/gouge columns by name (handles English and Chinese PM)
                col_collision = None
                col_gouge = None
                for col in columns:
                    col_lower = col.strip().lower()
                    if col_lower in ("collision", "碰撞"):
                        col_collision = col
                    elif col_lower in ("gouge", "过切"):
                        col_gouge = col

                if col_collision is None and col_gouge is None:
                    print(
                        f"WARNING: Neither Collision nor Gouge column found in CSV. "
                        f"Available columns: {columns}",
                        file=sys.stderr,
                    )

                for row in reader:
                    tp_name = row.get("Toolpath", row.get("Name", row.get("刀路", "unknown")))
                    # Check collision/gouge columns
                    has_collision = False
                    has_gouge = False
                    if col_collision:
                        has_collision = str(row.get(col_collision, "")).strip().lower() in (
                            "yes", "true", "1", "collision", "碰撞",
                        )
                    if col_gouge:
                        has_gouge = str(row.get(col_gouge, "")).strip().lower() in (
                            "yes", "true", "1", "gouge", "过切",
                        )

                    if has_collision:
                        collision_count += 1
                    if has_gouge:
                        gouge_count += 1

                    details.append(
                        {
                            "toolpath": tp_name,
                            "collision": has_collision,
                            "gouge": has_gouge,
                        }
                    )
        except Exception as e:
            details.append({"parse_error": str(e)})
    else:
        # SetupSheet CSV export failed -- cannot verify, fail-safe to reject
        details.append(
            {"error": "SetupSheets CSV not exported -- cannot verify collision status"}
        )
        # Fail-safe: if we can't check, assume collisions exist
        passed = False
        return {
            "passed": passed,
            "collision_count": collision_count,
            "gouge_count": gouge_count,
            "details": details,
        }

    passed = collision_count == 0 and gouge_count == 0
    return {
        "passed": passed,
        "collision_count": collision_count,
        "gouge_count": gouge_count,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check PowerMill project for collisions and gouges"
    )
    parser.add_argument("--project", required=True, help="Path to agent's PM project folder")
    args = parser.parse_args()

    if not os.path.exists(args.project):
        result = {
            "passed": False,
            "error": f"PM project not found: {args.project}",
            "collision_count": 0,
            "gouge_count": 0,
        }
        print(json.dumps(result))
        sys.exit(1)

    result = check_collision(args.project)
    print(json.dumps(result))
    sys.exit(0 if result.get("passed") else 2)


if __name__ == "__main__":
    main()
