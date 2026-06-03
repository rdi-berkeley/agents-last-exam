"""
setup_test_dirs.py — One-time setup (runs on remote Windows VM)

Creates output_test_pos/ and output_test_neg/ directories for each task,
populated with test STL files for validating the evaluation pipeline:

  output_test_pos/agent_sim.stl
      Copy of the task's own reference_sim.stl.
      Self-comparison should produce score ~1.0.

  output_test_neg/agent_sim.stl
      Copy of a DIFFERENT task's reference_sim.stl (round-robin assignment).
      Cross-workpiece comparison should produce a low score (~0.17 observed).

Prerequisites:
    generate_ref_stls.py must have been run first to create all reference_sim.stl files.

Usage:
    python setup_test_dirs.py            # create all test dirs
    python setup_test_dirs.py --dry-run  # preview only
"""

import os
import shutil
import argparse

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
TASK_ROOT = os.environ.get(
    "AGENTHLE_STAGE1_TASK_ROOT",
    r"E:\agenthle\engineering\gcode",
)

# Ordered list of task tags — used for round-robin negative sample assignment
# (each task's negative test uses the NEXT task's reference STL)
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


def setup_test_dirs(dry_run: bool = False):
    """Create positive and negative test directories for all tasks."""
    print("=" * 60)
    print("GCode Task -- Setup Test Directories")
    print(f"{'[DRY RUN] ' if dry_run else ''}")
    print("=" * 60)

    n = len(TASK_TAGS)

    for i, task_tag in enumerate(TASK_TAGS):
        task_dir = os.path.join(TASK_ROOT, task_tag)

        # Positive source: this task's own reference STL
        ref_stl = os.path.join(task_dir, "reference", "reference_sim.stl")

        # Negative source: next task's reference STL (wraps around)
        neg_source_tag = TASK_TAGS[(i + 1) % n]
        neg_stl = os.path.join(TASK_ROOT, neg_source_tag, "reference", "reference_sim.stl")

        # Destination paths
        pos_dir = os.path.join(task_dir, "output_test_pos")
        neg_dir = os.path.join(task_dir, "output_test_neg")
        pos_dst = os.path.join(pos_dir, "agent_sim.stl")
        neg_dst = os.path.join(neg_dir, "agent_sim.stl")

        print(f"\n[{task_tag}]")
        print(f"  pos src: {ref_stl}")
        print(f"  neg src: {neg_stl} (from {neg_source_tag})")

        # Validate source files exist
        if not os.path.exists(ref_stl):
            print(f"  WARN: reference_sim.stl not found -- run generate_ref_stls.py first")
            continue

        if not os.path.exists(neg_stl):
            print(f"  WARN: neg source STL not found ({neg_source_tag})")
            continue

        if not dry_run:
            os.makedirs(pos_dir, exist_ok=True)
            os.makedirs(neg_dir, exist_ok=True)
            shutil.copy2(ref_stl, pos_dst)
            shutil.copy2(neg_stl, neg_dst)

        print(f"  OK pos: {pos_dst}")
        print(f"  OK neg: {neg_dst}")

    print("\n" + "=" * 60)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create positive and negative test directories for all GCode tasks"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview operations without copying files"
    )
    args = parser.parse_args()
    setup_test_dirs(dry_run=args.dry_run)
