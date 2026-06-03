"""
generate_ref_stls.py — One-time setup (runs on remote Windows VM, PowerMill must be open)

For each task, opens the expert reference PM project in PowerMill, runs the
stock model simulation across all toolpaths, and exports the result as
reference/reference_sim.stl.

The resulting STL serves as the ground-truth for evaluation: the agent's
simulated stock model is compared against this reference.

Requirements:
    - PowerMill must be running (manual open or already running)
    - pip install pywin32

Usage:
    python generate_ref_stls.py                        # process all 18 tasks
    python generate_ref_stls.py --task 125162_319     # single task only
    python generate_ref_stls.py --force               # regenerate even if STL exists
"""

import os
import sys
import tempfile
import time
import argparse

try:
    import win32com.client
except ImportError:
    print("ERROR: pywin32 not installed. Run: pip install pywin32")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
TASK_ROOT = os.environ.get(
    "AGENTHLE_STAGE1_TASK_ROOT",
    r"E:\agenthle\engineering\gcode",
)

# All 18 task tags in processing order
TASK_TAGS = [
    "125162_319",
    "A125117_301",
    "A125138_301",
    "A125138_302",
    "MDBZDHZJ25_SKC_1_NCSM_T",
    "MR250692C00_M2",
    "MR250696C00_F1",
    "MR250696C00_S5",
    "MR250697C00_M1",
    "MR250697C00_S1",
    "MR250697C00_S2",
    "MR250698C00_F3",
    "MR250698C00_P6",
    "MR250698C00_U005",
    "T29153_050",
    # New 4 variants (added 2026-03-25)
    "MDB240386_S2",
    "MM250645B00_S2",
    "MM250645B00_S3",
    # NOTE: MM250689C00_M1 excluded — raw data has no .pmlprj (not a valid PM project)
]


def get_pm_connection():
    """Connect to the running PowerMill instance via COM.

    Tries GetActiveObject first (attach to existing), then Dispatch (launch new).
    Returns the PM COM object or None on failure.
    """
    try:
        pm = win32com.client.GetActiveObject("pmill.Document")
        return pm
    except Exception:
        pass
    try:
        pm = win32com.client.Dispatch("pmill.Document")
        time.sleep(5)  # Wait for PM to initialize
        return pm
    except Exception as e:
        print(f"ERROR: Cannot connect to PowerMill: {e}")
        return None


def generate_ref_stl(pm, task_tag: str, force: bool = False) -> bool:
    """Open the reference PM project, simulate, and export reference_sim.stl.

    The PowerMill macro:
    1. Opens the reference PM project
    2. Creates a new stock model ("Ref_Sim_Result")
    3. Attaches the block (bounding box around the workpiece)
    4. Inserts ALL toolpaths into the stock model in order
    5. Runs the stock model calculation (physical cutting simulation)
    6. Exports the resulting mesh as STL

    Args:
        pm: PowerMill COM object
        task_tag: e.g. "125162_319"
        force: if True, regenerate even if STL already exists
    Returns:
        True on success, False on failure
    """
    task_dir = os.path.join(TASK_ROOT, task_tag)
    ref_pm_dir = os.path.join(task_dir, "reference", "ref_pm_project")
    output_stl = os.path.join(task_dir, "reference", "reference_sim.stl")

    # Check if reference PM project exists
    if not os.path.exists(ref_pm_dir):
        print(f"  ERROR: ref_pm_project not found: {ref_pm_dir}")
        return False

    # Find the .pmlprj file (PM project identifier)
    pmlprj_files = [f for f in os.listdir(ref_pm_dir) if f.endswith(".pmlprj")]
    if not pmlprj_files:
        print(f"  ERROR: no .pmlprj in {ref_pm_dir}")
        return False

    # Skip if STL already exists (unless --force)
    if os.path.exists(output_stl) and not force:
        size_kb = os.path.getsize(output_stl) // 1024
        print(f"  SKIP: reference_sim.stl already exists ({size_kb} KB)")
        return True

    # Convert paths to forward-slash for PowerMill macro compatibility
    pm_path = ref_pm_dir.replace("\\", "/")
    stl_path = output_stl.replace("\\", "/")

    # Build the PowerMill macro
    # Key commands:
    #   PROJECT OPEN          — loads the PM project
    #   CREATE STOCKMODEL     — creates a new stock model entity
    #   EDIT BLOCKTYPE BOX    — set block type to bounding box
    #   EDIT BLOCK RESET      — reset block to workpiece extents
    #   EDIT STOCKMODEL ; BLOCK ; — attach the block to the stock model
    #   INSERT_INPUT Toolpath — add a toolpath to the simulation queue
    #   CALCULATE             — run the actual cutting simulation
    #   EXPORT STOCKMODEL_SHADING — export the visual mesh as STL
    macro_code = f"""
ECHO OFF
DIALOGS MESSAGE OFF
DIALOGS ERROR OFF

PROJECT OPEN "{pm_path}"

IF entity_exists('stockmodel', 'Ref_Sim_Result') {{
    DELETE STOCKMODEL "Ref_Sim_Result"
}}

CREATE STOCKMODEL "Ref_Sim_Result"
EDIT BLOCKTYPE BOX
EDIT BLOCK RESET
BLOCK ACCEPT

ACTIVATE STOCKMODEL "Ref_Sim_Result"
EDIT STOCKMODEL ; BLOCK ;

FOREACH tp IN folder('Toolpath') {{
    ACTIVATE TOOLPATH $tp.Name
    EDIT COLLISION APPLY
    EDIT STOCKMODEL "Ref_Sim_Result" INSERT_INPUT Toolpath $tp.Name LAST
}}

EDIT STOCKMODEL "Ref_Sim_Result" CALCULATE

ACTIVATE STOCKMODEL "Ref_Sim_Result"
EXPORT STOCKMODEL_SHADING ; "{stl_path}"

DIALOGS MESSAGE ON
DIALOGS ERROR ON
ECHO ON
"""

    # Write macro to temp file with GBK encoding
    # (PowerMill on Chinese Windows expects GBK, NOT UTF-8)
    temp_mac = os.path.join(tempfile.gettempdir(), "gen_ref_stl_temp.mac").replace("\\", "/")
    with open(temp_mac, "w", encoding="gbk") as f:
        f.write(macro_code)

    # Execute the macro via COM
    try:
        print(f"  Running simulation... (this may take several minutes)")
        pm.Execute(f'MACRO "{temp_mac}"')
    except Exception as e:
        print(f"  ERROR: Macro execution failed: {e}")
        return False

    # Verify the output STL was created and is non-empty
    if os.path.exists(output_stl) and os.path.getsize(output_stl) > 1024:
        size_kb = os.path.getsize(output_stl) // 1024
        print(f"  OK: reference_sim.stl exported ({size_kb} KB)")
        return True
    else:
        print(f"  ERROR: STL not found or empty at: {output_stl}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference_sim.stl for all (or specific) tasks"
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Process a single task tag (e.g. 125162_319)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate STL even if it already exists"
    )
    args = parser.parse_args()

    # Determine which tasks to process
    if args.task:
        if args.task not in TASK_TAGS:
            print(f"ERROR: Unknown task tag: {args.task}")
            print(f"Available: {TASK_TAGS}")
            sys.exit(1)
        tasks = [args.task]
    else:
        tasks = TASK_TAGS

    # Connect to PowerMill
    pm = get_pm_connection()
    if pm is None:
        print("ERROR: PowerMill is not running. Please open it first.")
        sys.exit(1)

    print("=" * 60)
    print(f"GCode Task -- Generate Reference STLs ({len(tasks)} tasks)")
    print("=" * 60)

    ok = 0
    fail = 0
    for i, tag in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {tag}")
        if generate_ref_stl(pm, tag, force=args.force):
            ok += 1
        else:
            fail += 1

    print("\n" + "=" * 60)
    print(f"Done: {ok} succeeded, {fail} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
