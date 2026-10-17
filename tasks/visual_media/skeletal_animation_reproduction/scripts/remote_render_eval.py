from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--renderer-script", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def launch_job(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ack_path = output_dir / "launch.json"
    try:
        reservation = ack_path.open("x", encoding="utf-8")
    except FileExistsError:
        print(json.dumps({"status": "already_submitted", "ack_path": str(ack_path)}))
        return 0
    with reservation:
        reservation.write(json.dumps({"status": "launching", "launcher_pid": os.getpid()}))
    if (output_dir / "render_report.json").exists():
        ack = {"status": "report_exists", "launcher_pid": os.getpid()}
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--blend",
            args.blend,
            "--output-dir",
            str(output_dir),
            "--renderer-script",
            args.renderer_script,
            "--evaluation-config",
            args.evaluation_config,
        ]
        options = {"close_fds": True, "cwd": str(Path(__file__).resolve().parent)}
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            options["start_new_session"] = True
        try:
            with (
                (output_dir / "job_stdout.txt").open("ab") as stdout,
                (output_dir / "job_stderr.txt").open("ab") as stderr,
            ):
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **options
                )
            ack = {"status": "started", "pid": process.pid}
        except Exception as error:
            ack = {"status": "launch_failed", "error": str(error)}
    temporary = ack_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ack), encoding="utf-8")
    temporary.replace(ack_path)
    print(json.dumps(ack))
    return 0 if ack.get("status") in {"started", "report_exists"} else 1


def main() -> int:
    args = parse_args()
    if args.launch:
        return launch_job(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "render_report.json"
    config = json.loads(Path(args.evaluation_config).read_text(encoding="utf-8"))
    blender_binary = os.environ.get("BLENDER_BINARY")
    if not blender_binary:
        failure = {
            "validity_gate_passed": False,
            "gate_fail_reasons": ["missing_blender_binary_env"],
            "view_paths": {},
            "silhouette_paths": [],
            "sample_positions": [],
        }
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    sample_count = int(config.get("sample_count", 10))
    image_width = int(config.get("image_width", 512))
    image_height = int(config.get("image_height", 500))

    cmd = [
        blender_binary,
        "--background",
        args.blend,
        "--python-exit-code",
        "1",
        "--python",
        args.renderer_script,
        "--",
        "--output-dir",
        str(output_dir),
        "--sample-count",
        str(sample_count),
        "--image-width",
        str(image_width),
        "--image-height",
        str(image_height),
        "--evaluation-config",
        args.evaluation_config,
    ]
    report_path.unlink(missing_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not report_path.is_file():
        failure = {
            "validity_gate_passed": False,
            "gate_fail_reasons": [
                "blender_render_failed" if result.returncode != 0 else "missing_render_report"
            ],
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "view_paths": {},
            "silhouette_paths": [],
            "sample_positions": [],
        }
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
