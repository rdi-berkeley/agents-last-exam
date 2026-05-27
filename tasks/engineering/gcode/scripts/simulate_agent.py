# -*- coding: utf-8 -*-
"""
simulate_agent.py -- Evaluation helper (runs on remote Windows VM)

Connects to the running PowerMill instance, opens the agent's PM project,
runs stock model simulation across all toolpaths, and exports agent_sim.stl
to the output directory.

Called by evaluate() in main.py via session.run_command().

Requirements:
    - PowerMill must be running
    - pip install pywin32

Usage:
    python simulate_agent.py --project PATH_TO_PM_PROJECT --output PATH_TO_OUTPUT_DIR

Exit codes:
    0 = success (STL written to output_dir/agent_sim.stl)
    1 = failure (no PM connection, no toolpaths, export failed, etc.)
"""

import os
import sys
import tempfile
import time
import argparse


def get_pm_connection():
    """Connect to the running PowerMill instance via COM.

    Returns the COM object or None if PM is not available.
    """
    import win32com.client

    try:
        return win32com.client.GetActiveObject("pmill.Document")
    except Exception:
        pass
    try:
        pm = win32com.client.Dispatch("pmill.Document")
        time.sleep(5)
        return pm
    except Exception as e:
        print(f"[ERROR] Cannot connect to PowerMill: {e}", file=sys.stderr)
        return None


def simulate_and_export(pm, pm_project_path: str, output_dir: str) -> bool:
    """Open PM project, simulate all toolpaths into a stock model, export STL.

    The macro:
    1. Opens the agent's PM project (which should have toolpaths designed by the agent)
    2. Creates a stock model named "Agent_Sim"
    3. Attaches block, inserts all toolpaths, calculates simulation
    4. Exports the visual mesh as STL

    Args:
        pm: PowerMill COM object
        pm_project_path: absolute path to the agent's PM project directory
        output_dir: directory to write agent_sim.stl to
    Returns:
        True if STL was successfully exported, False otherwise
    """
    output_stl = os.path.join(output_dir, "agent_sim.stl")
    pm_path_fwd = pm_project_path.replace("\\", "/")
    stl_path_fwd = output_stl.replace("\\", "/")

    # PowerMill macro for stock model simulation and STL export
    # Note: DIALOGS MESSAGE/ERROR OFF suppresses all popups during automation
    macro_code = f"""
ECHO OFF
DIALOGS MESSAGE OFF
DIALOGS ERROR OFF

PROJECT OPEN "{pm_path_fwd}"

// Abort if no toolpaths -- nothing to simulate
IF NOT entity_exists('toolpath', '*') {{
    PRINT "ERROR: No toolpaths found in project"
    EXIT
}}

// Remove any previous simulation result
IF entity_exists('stockmodel', 'Agent_Sim') {{
    DELETE STOCKMODEL "Agent_Sim"
}}

// Create new stock model with bounding-box block
CREATE STOCKMODEL "Agent_Sim"
EDIT BLOCKTYPE BOX
EDIT BLOCK RESET
BLOCK ACCEPT

// Attach the block to the stock model
ACTIVATE STOCKMODEL "Agent_Sim"
EDIT STOCKMODEL ; BLOCK ;

// Insert all toolpaths in order and apply collision checking
FOREACH tp IN folder('Toolpath') {{
    ACTIVATE TOOLPATH $tp.Name
    EDIT COLLISION APPLY
    EDIT STOCKMODEL "Agent_Sim" INSERT_INPUT Toolpath $tp.Name LAST
}}

// Run the cutting simulation (this is the CPU-intensive step)
EDIT STOCKMODEL "Agent_Sim" CALCULATE

// Export the visual mesh as STL
ACTIVATE STOCKMODEL "Agent_Sim"
EXPORT STOCKMODEL_SHADING ; "{stl_path_fwd}"

DIALOGS MESSAGE ON
DIALOGS ERROR ON
ECHO ON
"""

    # Write macro with GBK encoding (required by PowerMill on Chinese Windows)
    temp_mac = os.path.join(tempfile.gettempdir(), "simulate_agent_temp.mac").replace("\\", "/")
    with open(temp_mac, "w", encoding="gbk") as f:
        f.write(macro_code)

    try:
        pm.Execute(f'MACRO "{temp_mac}"')
    except Exception as e:
        print(f"[ERROR] Macro execution failed: {e}", file=sys.stderr)
        return False

    # Verify STL was created and is non-trivially sized (> 1KB)
    if os.path.exists(output_stl) and os.path.getsize(output_stl) > 1024:
        print(f"[OK] agent_sim.stl exported ({os.path.getsize(output_stl)//1024} KB): {output_stl}", file=sys.stderr)
        return True
    else:
        print(f"[ERROR] STL file not found or empty at: {output_stl}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simulate agent toolpath in PowerMill and export STL"
    )
    parser.add_argument("--project", required=True, help="Path to the agent's PM project folder")
    parser.add_argument("--output", required=True, help="Output directory for agent_sim.stl")
    args = parser.parse_args()

    if not os.path.exists(args.project):
        print(f"[ERROR] PM project not found: {args.project}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    pm = get_pm_connection()
    if pm is None:
        sys.exit(1)

    ok = simulate_and_export(pm, args.project, args.output)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
