#!/usr/bin/env python3
"""Recreate the benchmark-owned /workspace state for n8n_rss_monitoring_workflow_1."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path


def _count_sqlite_rows(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f"select count(*) from {table}")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input-dir", required=True)
    parser.add_argument("--workspace-input-dir", required=True)
    parser.add_argument("--workspace-website-dir", required=True)
    parser.add_argument("--runtime-db", required=True)
    parser.add_argument("--baseline-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_input_dir = Path(args.source_input_dir)
    workspace_input_dir = Path(args.workspace_input_dir)
    workspace_website_dir = Path(args.workspace_website_dir)
    runtime_db = Path(args.runtime_db)
    baseline_file = Path(args.baseline_file)

    if workspace_input_dir.exists():
        shutil.rmtree(workspace_input_dir)
    shutil.copytree(source_input_dir, workspace_input_dir)

    if workspace_website_dir.exists():
        shutil.rmtree(workspace_website_dir)
    (workspace_website_dir / "news" / "ja").mkdir(parents=True, exist_ok=True)
    (workspace_website_dir / "news" / "en").mkdir(parents=True, exist_ok=True)

    for keep_path in [
        workspace_website_dir / "news" / "ja" / ".gitkeep",
        workspace_website_dir / "news" / "en" / ".gitkeep",
    ]:
        keep_path.write_text("", encoding="utf-8")

    _run_git(["git", "init", "-b", "main"], cwd=workspace_website_dir)
    _run_git(["git", "config", "user.name", "AgentHLE Benchmark"], cwd=workspace_website_dir)
    _run_git(
        ["git", "config", "user.email", "agenthle+n8n-benchmark@example.com"],
        cwd=workspace_website_dir,
    )
    _run_git(["git", "add", "."], cwd=workspace_website_dir)
    _run_git(["git", "commit", "-m", "Initial website scaffold"], cwd=workspace_website_dir)

    baseline = {
        "workflow_count": _count_sqlite_rows(runtime_db, "workflow_entity"),
        "execution_count": _count_sqlite_rows(runtime_db, "execution_entity"),
        "credential_count": _count_sqlite_rows(runtime_db, "credentials_entity"),
        "git_branch": _run_git(["git", "branch", "--show-current"], cwd=workspace_website_dir),
        "git_commit_count": int(
            _run_git(["git", "rev-list", "--count", "HEAD"], cwd=workspace_website_dir)
        ),
        "git_head": _run_git(["git", "rev-parse", "HEAD"], cwd=workspace_website_dir),
    }
    conn = sqlite3.connect(runtime_db)
    try:
        cur = conn.cursor()
        baseline["workflow_ids"] = [row[0] for row in cur.execute("select id from workflow_entity")]
        baseline["max_execution_id"] = cur.execute(
            "select coalesce(max(id), 0) from execution_entity"
        ).fetchone()[0]
    finally:
        conn.close()
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps(baseline))


if __name__ == "__main__":
    main()
