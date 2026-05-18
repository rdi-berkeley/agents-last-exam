#!/usr/bin/env python
"""Build and sync canonical assets for demo/demo_desktop_note_linux."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


DOMAIN_NAME = "demo"
TASK_NAME = "demo_desktop_note_linux"
VARIANT_NAME = "base"

TASK_ROOT = Path(__file__).resolve().parents[1]
STAGED_ROOT = TASK_ROOT / "tmp" / "staged" / VARIANT_NAME
GCS_ROOT = f"gs://agenthle/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}"
VM_CUA_URL = "http://34.102.84.188:5000"
VM_TASK_PARENT = f"/media/user/data/agenthle/{DOMAIN_NAME}/{TASK_NAME}"

EXPECTED_TEXT = "\n".join(
    [
        "AgentHLE Desktop Plugin Demo",
        "This note was created through the desktop UI.",
    ]
)

REMOTE_OUTPUT_FILE = (
    "/media/user/data/agenthle/demo/demo_desktop_note_linux/base/"
    "output/agenthle_desktop_plugin_demo.txt"
)

NOTE_REQUEST = f"""AgentHLE Desktop Plugin Demo

Create a plain text note in a Linux text editor with exactly these two lines:
AgentHLE Desktop Plugin Demo
This note was created through the desktop UI.

Save it as:
{REMOTE_OUTPUT_FILE}
"""

OPEN_TEXT_EDITOR = """#!/usr/bin/env bash
set -euo pipefail

for editor in gedit xed mousepad leafpad kate kwrite; do
  if command -v "$editor" >/dev/null 2>&1; then
    nohup "$editor" >/tmp/agenthle-demo-note-editor.log 2>&1 &
    exit 0
  fi
done

if command -v xdg-open >/dev/null 2>&1; then
  tmp="${TMPDIR:-/tmp}/agenthle-demo-note-edit.txt"
  : > "$tmp"
  nohup xdg-open "$tmp" >/tmp/agenthle-demo-note-editor.log 2>&1 &
  exit 0
fi

echo "No graphical text editor found. Try nano or vi in a terminal." >&2
exit 1
"""


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
        EXPECTED_TEXT + "\n",
        encoding="utf-8",
    )
    launcher = STAGED_ROOT / "software" / "open_text_editor.sh"
    launcher.write_text(OPEN_TEXT_EDITOR, encoding="utf-8")
    launcher.chmod(0o755)

    (STAGED_ROOT / "output_test_pos" / "agenthle_desktop_plugin_demo.txt").write_text(
        EXPECTED_TEXT + "\n",
        encoding="utf-8",
    )
    (STAGED_ROOT / "output_test_neg" / "agenthle_desktop_plugin_demo.txt").write_text(
        "AgentHLE Desktop Plugin Demo\nWrong second line.\n",
        encoding="utf-8",
    )


def sync_gcs() -> None:
    run(["gsutil", "-m", "rsync", "-r", str(STAGED_ROOT), GCS_ROOT])


def sync_vm(cua_url: str) -> None:
    vm_variant_root = f"{VM_TASK_PARENT}/{VARIANT_NAME}"
    remote_command = (
        f"mkdir -p {vm_variant_root} && "
        f"gsutil -m rsync -r {GCS_ROOT} {vm_variant_root}/ && "
        f"chmod +x {vm_variant_root}/software/open_text_editor.sh"
    )
    print("+ CUA", cua_url, remote_command)
    payload = json.dumps({"command": "run_command", "params": {"command": remote_command}}).encode()
    req = urllib.request.Request(
        f"{cua_url.rstrip('/')}/cmd",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = None
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data: "):
                data = json.loads(line[6:])
                break
    if data is None:
        raise RuntimeError("CUA sync command returned no response")
    rc = data.get("return_code", data.get("returncode", 0))
    if rc != 0:
        raise RuntimeError(
            f"CUA sync command failed rc={rc}: {data.get('stderr', '') or data.get('error', '')}"
        )
    stdout = data.get("stdout", "") or data.get("output", "")
    if stdout:
        print(stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-gcs", action="store_true")
    parser.add_argument("--sync-vm", action="store_true")
    parser.add_argument("--vm-cua-url", default=VM_CUA_URL)
    args = parser.parse_args()

    build_staged()
    if args.sync_gcs:
        sync_gcs()
    if args.sync_vm:
        sync_vm(args.vm_cua_url)


if __name__ == "__main__":
    main()
