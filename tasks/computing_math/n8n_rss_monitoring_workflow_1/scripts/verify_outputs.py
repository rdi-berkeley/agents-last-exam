#!/usr/bin/env python3
"""VM-side verifier for n8n_rss_monitoring_workflow_1."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

RSS_NS = {
    "rss": "http://purl.org/rss/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
IGNORE_FILES = {".gitkeep", ".gitignore"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--workspace-website-dir", required=True)
    parser.add_argument("--runtime-db", required=True)
    parser.add_argument("--baseline-file", required=True)
    return parser.parse_args()


def parse_rss_items(input_dir: Path) -> list[dict[str, str]]:
    root = ET.fromstring((input_dir / "rss_snapshot.xml").read_text(encoding="utf-8"))
    items: list[dict[str, str]] = []
    for idx, item in enumerate(root.findall("rss:item", RSS_NS), start=1):
        published_raw = item.findtext("dc:date", default="", namespaces=RSS_NS)
        published = published_raw[:10]
        items.append(
            {
                "index": str(idx),
                "filename": f"{published}-{idx}.md",
                "published": published,
                "source_url": item.findtext("rss:link", default="", namespaces=RSS_NS).strip(),
                "headline_ja": item.findtext("rss:title", default="", namespaces=RSS_NS).strip(),
            }
        )
    return items


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing yaml frontmatter")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("frontmatter terminator not found")
    frontmatter_block = parts[0].splitlines()[1:]
    body = parts[1]
    frontmatter: dict[str, str] = {}
    for raw in frontmatter_block:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    return frontmatter, body


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def _cjk_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    cjk_count = sum(1 for char in meaningful if _contains_cjk(char))
    return cjk_count / len(meaningful)


def validate_markdown_file(path: Path, *, expected: dict[str, str], lang: str) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]
    try:
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path.name}: invalid markdown structure ({exc})"]

    for field in ["headline", "published", "excerpt", "source_url", "lang"]:
        if not frontmatter.get(field):
            problems.append(f"{path.name}: missing frontmatter field {field}")

    if frontmatter.get("published") != expected["published"]:
        problems.append(f"{path.name}: published mismatch")
    if frontmatter.get("source_url") != expected["source_url"]:
        problems.append(f"{path.name}: source_url mismatch")
    if frontmatter.get("lang") != lang:
        problems.append(f"{path.name}: lang mismatch")
    if lang == "ja" and frontmatter.get("headline") != expected["headline_ja"]:
        problems.append(f"{path.name}: ja headline mismatch")
    if not frontmatter.get("headline", "").strip():
        problems.append(f"{path.name}: headline empty")
    excerpt = frontmatter.get("excerpt", "")
    if not excerpt.strip():
        problems.append(f"{path.name}: excerpt empty")
    if len(excerpt) > 200:
        problems.append(f"{path.name}: excerpt too long")
    if "## " not in body:
        problems.append(f"{path.name}: missing level-2 heading")
    if not re.search(r"(?im)^##\s+About\b", body):
        problems.append(f"{path.name}: missing About section")
    else:
        about_section = re.split(r"(?im)^##\s+About\b.*$", body, maxsplit=1)
        if len(about_section) >= 2:
            about_text = about_section[1]
            info_lines = [
                line.strip()
                for line in about_text.splitlines()
                if line.strip() and (":" in line or "：" in line)
            ]
            if len(info_lines) < 5:
                problems.append(f"{path.name}: About section missing company information details")
            if "http://" not in about_text and "https://" not in about_text:
                problems.append(f"{path.name}: About section missing company URL")
    if lang == "ja":
        if not _contains_cjk(body):
            problems.append(f"{path.name}: ja body missing CJK content")
    else:
        if _cjk_ratio(body) >= 0.10:
            problems.append(f"{path.name}: en body contains too much CJK content")
    return problems


def validate_sync_log(path: Path, candidate_root: Path, expected_items: list[dict[str, str]]) -> list[str]:
    if not path.exists():
        return [f"missing sync log: {path}"]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    article_lines = [
        line
        for line in non_empty_lines
        if "total" not in line.lower() and "articles processed" not in line.lower()
    ]
    total_lines = [
        line
        for line in non_empty_lines
        if "total" in line.lower() or "articles processed" in line.lower()
    ]
    if len(article_lines) < len(expected_items):
        problems.append("sync_log has fewer article lines than expected")
    for item in expected_items:
        matching_line = next((line for line in article_lines if item["filename"] in line), "")
        if not matching_line:
            problems.append(f"sync_log missing line for {item['filename']}")
            continue
        accepted_headlines = {item["headline_ja"]}
        for lang in ["ja", "en"]:
            md_path = candidate_root / "news" / lang / item["filename"]
            if not md_path.exists():
                continue
            try:
                frontmatter, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            headline = frontmatter.get("headline", "").strip()
            if headline:
                accepted_headlines.add(headline)
        if not any(headline in matching_line for headline in accepted_headlines):
            problems.append(f"sync_log line missing headline for {item['filename']}")
    if not total_lines:
        problems.append("sync_log missing total count line")
    elif not any(re.search(r"\b3\b", line) for line in total_lines):
        problems.append("sync_log total count line does not mention 3")
    return problems


def validate_output_tree(candidate_root: Path, expected_filenames: list[str]) -> list[str]:
    problems: list[str] = []
    for lang in ["ja", "en"]:
        lang_dir = candidate_root / "news" / lang
        if not lang_dir.exists():
            problems.append(f"missing directory: {lang_dir}")
            continue
        actual = sorted(
            path.name
            for path in lang_dir.iterdir()
            if path.is_file() and path.name not in IGNORE_FILES
        )
        if actual != expected_filenames:
            problems.append(f"{lang_dir}: expected files {expected_filenames}, got {actual}")
    return problems


def validate_language_pairs(candidate_root: Path, expected_filenames: list[str]) -> list[str]:
    problems: list[str] = []
    for filename in expected_filenames:
        ja_path = candidate_root / "news" / "ja" / filename
        en_path = candidate_root / "news" / "en" / filename
        if not ja_path.exists() or not en_path.exists():
            continue
        try:
            ja_frontmatter, _ = parse_frontmatter(ja_path.read_text(encoding="utf-8"))
            en_frontmatter, _ = parse_frontmatter(en_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{filename}: unable to compare ja/en frontmatter ({exc})")
            continue
        for field in ["published", "source_url"]:
            if ja_frontmatter.get(field) != en_frontmatter.get(field):
                problems.append(f"{filename}: ja/en {field} mismatch")
    return problems


def git_output_checks(repo_dir: Path, expected_paths: list[str], baseline: dict[str, int | str]) -> list[str]:
    problems: list[str] = []
    if not (repo_dir / ".git").exists():
        return [f"missing git repo: {repo_dir}"]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    try:
        branch = git("branch", "--show-current")
        if branch != "main":
            problems.append(f"git branch is {branch}, expected main")
        commit_count = int(git("rev-list", "--count", "HEAD"))
        if commit_count <= int(baseline["git_commit_count"]):
            problems.append("git repo has no post-setup commit")
        baseline_head = str(baseline.get("git_head", "")).strip()
        if baseline_head:
            new_commits = [
                line.strip()
                for line in git("rev-list", "--reverse", "HEAD", f"^{baseline_head}").splitlines()
                if line.strip()
            ]
            if not new_commits:
                problems.append("git repo has no commit after baseline HEAD")
            else:
                expected_news_paths = sorted(path for path in expected_paths if path.startswith("news/"))
                added_news_commit_found = False
                for commit in new_commits:
                    diff_lines = [
                        line.strip()
                        for line in git("diff-tree", "--no-commit-id", "--name-status", "-r", commit).splitlines()
                        if line.strip()
                    ]
                    added_news_paths = sorted(
                        line.split("\t", 1)[1]
                        for line in diff_lines
                        if line.startswith("A\t") and "\t" in line and line.split("\t", 1)[1].startswith("news/")
                    )
                    if added_news_paths == expected_news_paths:
                        added_news_commit_found = True
                        break
                if not added_news_commit_found:
                    problems.append("no post-setup commit added exactly the 6 required news files")
        status = git("status", "--porcelain")
        if status.strip():
            problems.append("git working tree is not clean")
        for rel_path in expected_paths:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel_path],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if tracked.returncode != 0:
                problems.append(f"git does not track {rel_path}")
    except Exception as exc:
        problems.append(f"git check failed: {exc}")
    return problems


def _load_json_blob(raw: str | None) -> Any:
    if not raw:
        return None
    return json.loads(raw)


def _count_connections(connections: Any) -> int:
    if isinstance(connections, dict):
        total = 0
        for value in connections.values():
            total += _count_connections(value)
        return total
    if isinstance(connections, list):
        total = 0
        for value in connections:
            total += _count_connections(value)
        return total
    if isinstance(connections, tuple):
        return sum(_count_connections(value) for value in connections)
    if isinstance(connections, dict) and "node" in connections:
        return 1
    return 1 if isinstance(connections, dict) and "node" in connections else 0


def _count_edge_targets(connections: Any) -> int:
    if isinstance(connections, list):
        total = 0
        for value in connections:
            if isinstance(value, dict) and "node" in value:
                total += 1
            else:
                total += _count_edge_targets(value)
        return total
    if isinstance(connections, dict):
        return sum(_count_edge_targets(value) for value in connections.values())
    return 0


def _is_llm_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type", "")).lower()
    if any(token in node_type for token in ["openai", "anthropic", "langchain", "gemini", "ollama", "lmchat"]):
        return True
    if "httprequest" in node_type or "http-request" in node_type:
        parameters = node.get("parameters", {}) or {}
        url = str(parameters.get("url", "")).lower()
        body = json.dumps(parameters, ensure_ascii=False).lower()
        llm_markers = [
            "api.openai.com",
            "/v1/chat/completions",
            "/v1/responses",
            "openai",
            "anthropic",
            "generativelanguage",
            "gemini",
        ]
        return any(marker in url or marker in body for marker in llm_markers)
    return False


def _is_file_or_shell_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("type", "")).lower()
    return any(
        token in node_type
        for token in [
            "executecommand",
            "readwritefile",
            "readbinaryfile",
            "writebinaryfile",
            "localfiletrigger",
        ]
    )


def runtime_usage_checks(runtime_db: Path, baseline: dict[str, Any]) -> list[str]:
    conn = sqlite3.connect(runtime_db)
    try:
        cur = conn.cursor()
        cur.execute("select count(*) from workflow_entity")
        workflow_count = int(cur.fetchone()[0])
        cur.execute("select count(*) from execution_entity")
        execution_count = int(cur.fetchone()[0])
        cur.execute("select count(*) from credentials_entity where name = 'OpenAI API' and type = 'openAiApi'")
        credential_count = int(cur.fetchone()[0])
        workflow_rows = cur.execute(
            "select id, name, nodes, connections from workflow_entity"
        ).fetchall()
        execution_rows = cur.execute(
            "select id, workflowId, finished, status from execution_entity order by id asc"
        ).fetchall()
    finally:
        conn.close()

    problems: list[str] = []
    if workflow_count <= int(baseline["workflow_count"]):
        problems.append("n8n workflow count did not increase")
    if execution_count <= int(baseline["execution_count"]):
        problems.append("n8n execution count did not increase")
    if credential_count < 1:
        problems.append("OpenAI API credential missing at evaluation time")

    baseline_workflow_ids = set(str(value) for value in baseline.get("workflow_ids", []))
    new_workflows = [row for row in workflow_rows if str(row[0]) not in baseline_workflow_ids]
    if not new_workflows:
        problems.append("no new workflow saved after setup baseline")
        return problems

    successful_execution_workflow_ids = {
        str(workflow_id)
        for execution_id, workflow_id, finished, status in execution_rows
        if int(execution_id) > int(baseline.get("max_execution_id", 0))
        and bool(finished)
        and str(status).lower() == "success"
    }
    if not successful_execution_workflow_ids:
        problems.append("no successful workflow execution after setup baseline")

    qualifying_workflow_found = False
    for workflow_id, workflow_name, nodes_raw, connections_raw in new_workflows:
        nodes = _load_json_blob(nodes_raw) or []
        connections = _load_json_blob(connections_raw) or {}
        node_count = len(nodes) if isinstance(nodes, list) else 0
        connection_count = _count_edge_targets(connections)
        has_llm_node = any(isinstance(node, dict) and _is_llm_node(node) for node in nodes)
        has_file_or_shell_node = any(
            isinstance(node, dict) and _is_file_or_shell_node(node) for node in nodes
        )
        has_success = str(workflow_id) in successful_execution_workflow_ids
        if node_count >= 5 and connection_count >= 4 and has_llm_node and has_file_or_shell_node and has_success:
            qualifying_workflow_found = True
            break

    if not qualifying_workflow_found:
        problems.append(
            "no new workflow satisfied all runtime checks: >=5 nodes, >=4 connections, "
            "LLM node, file/shell node, and successful execution"
        )
    return problems


def verify(mode: str, input_dir: Path, candidate_root: Path, workspace_website_dir: Path, runtime_db: Path, baseline_file: Path) -> dict:
    items = parse_rss_items(input_dir)
    expected_filenames = [item["filename"] for item in items]
    candidate_news_root = candidate_root / "news"

    checks: list[dict[str, object]] = []
    problems: list[str] = []
    problems.extend(validate_output_tree(candidate_root, expected_filenames))

    for item in items:
        problems.extend(
            validate_markdown_file(
                candidate_news_root / "ja" / item["filename"],
                expected=item,
                lang="ja",
            )
        )
        problems.extend(
            validate_markdown_file(
                candidate_news_root / "en" / item["filename"],
                expected=item,
                lang="en",
            )
        )

    problems.extend(validate_sync_log(candidate_root / "sync_log.txt", candidate_root, items))
    problems.extend(validate_language_pairs(candidate_root, expected_filenames))

    checks.append({"check": "markdown_and_log_structure", "passed": not problems, "message": "; ".join(problems[:10])})

    if mode == "output":
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        expected_git_paths = ["sync_log.txt"]
        for item in items:
            expected_git_paths.append(f"news/ja/{item['filename']}")
            expected_git_paths.append(f"news/en/{item['filename']}")
        git_problems = git_output_checks(workspace_website_dir, expected_git_paths, baseline)
        runtime_problems = runtime_usage_checks(runtime_db, baseline)
        checks.append({"check": "git_commit_evidence", "passed": not git_problems, "message": "; ".join(git_problems[:10])})
        checks.append({"check": "n8n_runtime_evidence", "passed": not runtime_problems, "message": "; ".join(runtime_problems[:10])})

    score = 1.0 if all(bool(check["passed"]) for check in checks) else 0.0
    return {"score": score, "checks": checks, "mode": mode}


def main() -> None:
    args = parse_args()
    payload = verify(
        mode=args.mode,
        input_dir=Path(args.input_dir),
        candidate_root=Path(args.candidate_root),
        workspace_website_dir=Path(args.workspace_website_dir),
        runtime_db=Path(args.runtime_db),
        baseline_file=Path(args.baseline_file),
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
