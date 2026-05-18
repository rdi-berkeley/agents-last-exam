#!/usr/bin/env python
"""Build and sync canonical assets for demo/demo_desktop_note."""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
from pathlib import Path


DOMAIN_NAME = "demo"
TASK_NAME = "demo_desktop_note"
VARIANT_NAME = "base"

TASK_ROOT = Path(__file__).resolve().parents[1]
STAGED_ROOT = TASK_ROOT / "tmp" / "staged" / VARIANT_NAME
GCS_ROOT = f"gs://agenthle/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}"
WORKER_NAME = "agenthle-nested-kvm-03"
WORKER_PROJECT = "sunblaze-4"
WORKER_ZONE = "us-central1-a"
WORKER_TASK_PARENT = f"/mnt/agenthle-task-data/{DOMAIN_NAME}/{TASK_NAME}"

EXPECTED_TEXT = "\r\n".join(
    [
        "AgentHLE Desktop Plugin Demo",
        "This note was created through the desktop UI.",
    ]
)

NOTE_REQUEST = """AgentHLE Desktop Plugin Demo

Create a plain text note in Notepad with exactly these two lines:
AgentHLE Desktop Plugin Demo
This note was created through the desktop UI.

Save it as:
E:\\agenthle\\demo\\demo_desktop_note\\base\\output\\agenthle_desktop_plugin_demo.txt
"""

# Windows Shell Link generated on a Windows VM with WScript.Shell.CreateShortcut.
# target: C:\Windows\System32\notepad.exe
# working_dir/icon: C:\Windows\System32
NOTEPAD_LNK_B64 = (
    "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"
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_staged() -> None:
    reset_dir(STAGED_ROOT)
    for subdir in ("input", "reference", "software", "output", "output_test_pos", "output_test_neg"):
        (STAGED_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    (STAGED_ROOT / "input" / "note_request.txt").write_text(NOTE_REQUEST, encoding="utf-8")
    (STAGED_ROOT / "reference" / "expected_note.txt").write_text(
        EXPECTED_TEXT + "\r\n",
        encoding="utf-8",
    )
    (STAGED_ROOT / "software" / "Notepad.lnk").write_bytes(base64.b64decode(NOTEPAD_LNK_B64))

    (STAGED_ROOT / "output_test_pos" / "agenthle_desktop_plugin_demo.txt").write_text(
        EXPECTED_TEXT + "\r\n",
        encoding="utf-8",
    )
    (STAGED_ROOT / "output_test_neg" / "agenthle_desktop_plugin_demo.txt").write_text(
        "AgentHLE Desktop Plugin Demo\r\nWrong second line.\r\n",
        encoding="utf-8",
    )


def sync_gcs() -> None:
    run(["gsutil", "-m", "rsync", "-r", str(STAGED_ROOT), GCS_ROOT])


def sync_worker() -> None:
    worker_variant_root = f"{WORKER_TASK_PARENT}/{VARIANT_NAME}"
    remote_command = (
        f"mkdir -p {worker_variant_root} && "
        f"gsutil -m rsync -r {GCS_ROOT} {worker_variant_root}/"
    )
    run(
        [
            "gcloud",
            "compute",
            "ssh",
            WORKER_NAME,
            "--project",
            WORKER_PROJECT,
            "--zone",
            WORKER_ZONE,
            "--command",
            remote_command,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-gcs", action="store_true")
    parser.add_argument("--sync-worker", action="store_true")
    args = parser.parse_args()

    build_staged()
    if args.sync_gcs:
        sync_gcs()
    if args.sync_worker:
        sync_worker()


if __name__ == "__main__":
    main()
