from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_task_data.sh"
REVISION = "a" * 40
TASK = Path("life_sciences/tcga_brca_deg_analysis/base")
RUNTIME = TASK / "input/runtime_env/.venv"
LINKS = {
    "bin/python": "/home/user/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14",
    "bin/python3": "python",
    "bin/python3.14": "python",
    "lib64": "lib",
}


@pytest.fixture
def fetch_fixture(tmp_path):
    tree = tmp_path / "source"
    (tree / RUNTIME / "bin").mkdir(parents=True)
    (tree / RUNTIME / "lib").mkdir()
    (tree / TASK / "input/public.txt").write_text("fixture input\n")
    (tree / TASK / "reference").mkdir()
    (tree / TASK / "reference/gold.csv").write_text("fixture,reference\n")
    (tree / TASK / "software").mkdir()
    executable = tree / TASK / "software/run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o751)
    for relative, target in LINKS.items():
        (tree / RUNTIME / relative).symlink_to(target)
    archive = tmp_path / "fixture.tar.gz"
    subprocess.run(
        ["tar", "-czf", str(archive), "-C", str(tree), "."],
        check=True,
        env={**os.environ, "TAR_OPTIONS": ""},
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "huggingface-cli"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json, os, pathlib, shutil, sys\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('command')\n"
        "parser.add_argument('repo')\n"
        "parser.add_argument('filename')\n"
        "parser.add_argument('--repo-type', required=True)\n"
        "parser.add_argument('--revision', required=True)\n"
        "parser.add_argument('--local-dir', required=True)\n"
        "args = parser.parse_args()\n"
        "record = vars(args).copy()\n"
        "record['paths'] = {name: os.environ.get(name) for name in "
        "('TMPDIR', 'TMP', 'TEMP', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_HOME')}\n"
        "with open(os.environ['FETCH_STUB_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "assert args.command == 'download'\n"
        "assert args.repo == 'agents-last-exam/agents-last-exam-data-archive'\n"
        "assert args.filename == 'ale-tasks-data.tar.gz' and args.repo_type == 'dataset'\n"
        "if args.revision != os.environ['FETCH_STUB_REVISION']:\n"
        "    sys.exit(9)\n"
        "download = pathlib.Path(args.local_dir)\n"
        "source = pathlib.Path(os.environ['FETCH_STUB_ARCHIVE'])\n"
        "if os.environ.get('FETCH_STUB_ARCHIVE_LINK'):\n"
        "    (download / args.filename).symlink_to(source)\n"
        "else:\n"
        "    shutil.copyfile(source, download / args.filename)\n"
        "metadata = download / '.cache/huggingface/download' / (args.filename + '.metadata')\n"
        "metadata.parent.mkdir(parents=True)\n"
        "if not os.environ.get('FETCH_STUB_NO_METADATA'):\n"
        "    revision = os.environ.get('FETCH_STUB_RECEIPT_REVISION', args.revision)\n"
        "    metadata.write_text(revision + '\\nfixture-etag\\n1.0\\n')\n"
        "if os.environ.get('FETCH_STUB_RACE_DEST'):\n"
        "    destination = pathlib.Path(os.environ['FETCH_STUB_RACE_DEST'])\n"
        "    destination.mkdir()\n"
        "    (destination / 'foreign-file').write_text('retain me')\n"
        "print(download / args.filename)\n"
    )
    stub.chmod(0o755)
    mount = binary / "findmnt"
    mount.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = -n ]; then\n'
        "  printf '%s\\n' \"${FETCH_STUB_FSTYPE:-ext4}\"\n"
        "else\n"
        "  printf '%s\\n' 'fixture persistent filesystem'\n"
        "fi\n"
    )
    mount.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "FETCH_STUB_LOG": str(tmp_path / "calls.jsonl"),
        "FETCH_STUB_ARCHIVE": str(archive),
        "FETCH_STUB_REVISION": REVISION,
        "HF_HOME": str(tmp_path / "existing-auth-home"),
        "HF_HUB_CACHE": "/dev/shm/forbidden-hub-cache",
        "HF_XET_CACHE": "/dev/shm/forbidden-xet-cache",
        "TMPDIR": "/dev/shm",
        "TAR_OPTIONS": "--strip-components=99",
    }
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return tmp_path, archive, env, ["--revision", REVISION, "--sha256", digest]


def run_fetch(fixture, arguments=None, extra_env=None):
    root, _, env, pins = fixture
    return subprocess.run(
        ["bash", str(SCRIPT), *(pins if arguments is None else arguments)],
        cwd=root,
        env={**env, **(extra_env or {})},
        text=True,
        capture_output=True,
        timeout=30,
    )


def calls(root):
    log = root / "calls.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_pinned_default_extracts_native_links_and_modes(fetch_fixture):
    root, _, _, _ = fetch_fixture
    result = run_fetch(fetch_fixture)
    assert result.returncode == 0, result.stderr
    destination = root / "task-data-v1.1"
    for relative, target in LINKS.items():
        assert os.readlink(destination / RUNTIME / relative) == target
    assert (destination / TASK / "reference/gold.csv").read_text() == "fixture,reference\n"
    assert stat.S_IMODE((destination / TASK / "software/run.sh").stat().st_mode) == 0o751
    assert not (root / "task-data").exists()
    assert not list(root.glob(".ale-fetch.*"))
    assert not (destination / "ale-tasks-data.tar.gz").exists()
    request = calls(root)[0]
    assert request["revision"] == REVISION
    work = Path(request["local_dir"]).parent
    assert work.parent == root
    for name in ("TMPDIR", "TMP", "TEMP", "HF_HUB_CACHE", "HF_XET_CACHE"):
        assert Path(request["paths"][name]).is_relative_to(work)
    assert request["paths"]["HF_HOME"] == str(root / "existing-auth-home")


@pytest.mark.parametrize("revision", ["main", "v1.1", "a" * 39, "g" * 40, "<COMMIT_SHA>"])
def test_mutable_or_invalid_revision_fails_before_download(fetch_fixture, revision):
    root, _, _, pins = fetch_fixture
    result = run_fetch(fetch_fixture, ["--revision", revision, *pins[2:]])
    assert result.returncode != 0
    assert "40-hex HF commit" in result.stderr
    assert calls(root) == []
    assert not (root / "task-data-v1.1").exists()


@pytest.mark.parametrize("arguments", [[], ["--revision", REVISION], ["--legacy"]])
def test_missing_pins_never_select_defaults(fetch_fixture, arguments):
    root, _, _, _ = fetch_fixture
    assert run_fetch(fetch_fixture, arguments).returncode != 0
    assert calls(root) == []


def test_unavailable_exact_revision_has_no_fallback(fetch_fixture):
    root, _, _, pins = fetch_fixture
    result = run_fetch(fetch_fixture, ["--revision", "b" * 40, *pins[2:]])
    assert result.returncode != 0
    assert "no fallback" in result.stderr
    assert [request["revision"] for request in calls(root)] == ["b" * 40]
    assert not (root / "task-data-v1.1").exists()


def test_mismatched_revision_receipt_fails_before_extraction(fetch_fixture):
    root, _, _, _ = fetch_fixture
    result = run_fetch(fetch_fixture, extra_env={"FETCH_STUB_RECEIPT_REVISION": "b" * 40})
    assert result.returncode != 0
    assert "revision mismatch" in result.stderr
    assert not (root / "task-data-v1.1").exists()
    assert all(not list(path.iterdir()) for path in root.glob(".ale-fetch.*/extracted"))


def test_wrong_checksum_never_extracts_or_changes_empty_destination(fetch_fixture):
    root, _, _, pins = fetch_fixture
    destination = root / "empty"
    destination.mkdir()
    result = run_fetch(fetch_fixture, [*pins[:2], "--sha256", "0" * 64, str(destination)])
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert not list(destination.iterdir())
    assert all(not list(path.iterdir()) for path in root.glob(".ale-fetch.*/extracted"))


@pytest.mark.parametrize("existing", ["old.txt", ".hidden-old"])
def test_nonempty_destination_is_untouched_and_not_downloaded(fetch_fixture, existing):
    root, _, _, pins = fetch_fixture
    destination = root / "stale"
    destination.mkdir()
    (destination / existing).write_text("existing data")
    result = run_fetch(fetch_fixture, [*pins, str(destination)])
    assert result.returncode != 0
    assert "nonempty destination" in result.stderr
    assert (destination / existing).read_text() == "existing data"
    assert calls(root) == []


def test_explicit_legacy_uses_separate_historical_destination(fetch_fixture):
    root, _, _, pins = fetch_fixture
    result = run_fetch(fetch_fixture, ["--legacy", *pins])
    assert result.returncode == 0, result.stderr
    assert (root / "task-data" / TASK / "input/public.txt").exists()
    assert not (root / "task-data-v1.1").exists()
    assert calls(root)[0]["revision"] == REVISION


def test_empty_custom_destination_with_spaces_is_supported(fetch_fixture):
    root, _, _, pins = fetch_fixture
    destination = root / "empty directory"
    destination.mkdir()
    result = run_fetch(fetch_fixture, [*pins, str(destination)])
    assert result.returncode == 0, result.stderr
    assert os.readlink(destination / RUNTIME / "bin/python3") == "python"


@pytest.mark.parametrize("filesystem", ["tmpfs", "ramfs", "devtmpfs", "overlay"])
def test_unverified_storage_fails_before_download(fetch_fixture, filesystem):
    root, _, _, _ = fetch_fixture
    result = run_fetch(fetch_fixture, extra_env={"FETCH_STUB_FSTYPE": filesystem})
    assert result.returncode != 0
    assert "memory-backed filesystem" in result.stderr
    assert calls(root) == []
    assert not list(root.glob(".ale-fetch.*"))


def test_symlink_destination_is_rejected(fetch_fixture):
    root, _, _, pins = fetch_fixture
    actual = root / "real-directory"
    actual.mkdir()
    alias = root / "alias"
    alias.symlink_to(actual)
    result = run_fetch(fetch_fixture, [*pins, str(alias)])
    assert result.returncode != 0
    assert "must not be a symlink" in result.stderr
    assert not list(actual.iterdir())
    assert calls(root) == []


@pytest.mark.parametrize("setting", ["FETCH_STUB_ARCHIVE_LINK", "FETCH_STUB_NO_METADATA"])
def test_missing_receipt_or_external_archive_link_is_rejected(fetch_fixture, setting):
    root, _, _, _ = fetch_fixture
    result = run_fetch(fetch_fixture, extra_env={setting: "1"})
    assert result.returncode != 0
    assert not (root / "task-data-v1.1").exists()
    assert all(not list(path.iterdir()) for path in root.glob(".ale-fetch.*/extracted"))


def test_extraction_failure_does_not_publish_partial_tree(fetch_fixture):
    root, archive, _, pins = fetch_fixture
    archive.write_bytes(b"not a tar archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = run_fetch(fetch_fixture, [*pins[:2], "--sha256", digest])
    assert result.returncode != 0
    assert not (root / "task-data-v1.1").exists()
    assert "Staged evidence retained" in result.stderr


def test_destination_created_during_download_is_not_overwritten(fetch_fixture):
    root, _, _, _ = fetch_fixture
    destination = root / "task-data-v1.1"
    result = run_fetch(fetch_fixture, extra_env={"FETCH_STUB_RACE_DEST": str(destination)})
    assert result.returncode != 0
    assert "nonempty destination" in result.stderr
    assert (destination / "foreign-file").read_text() == "retain me"
    assert not (destination / TASK).exists()
