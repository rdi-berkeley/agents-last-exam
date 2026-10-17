from __future__ import annotations

import importlib.util
import base64
import io
import json
import fnmatch
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "ale_run/environments/images/ale_ubuntu22_docker"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = load_module("bootstrap_ssh")
policy = load_module("rootfs_policy")
audit = load_module("runtime_audit")
build = load_module("local_build")
PUBLIC_KEY = (
    "ssh-ed25519 "
    + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + b"a" * 32).decode()
)
TASK_RESIDUE_ROOTS = (
    "/workspace",
    "/reference",
    "/protected",
    "/output",
    "/output_test_pos",
    "/output_test_neg",
)


@pytest.fixture
def contract():
    return {
        "required_paths": ["/opt/runtime"],
        "preserve_cache_paths": [],
        "probes": [["/usr/bin/python3", "-c", "pass"], ["/usr/bin/Rscript", "--version"]],
    }


@pytest.fixture
def root(tmp_path, contract):
    guest = tmp_path / "guest"
    for path in [*policy.REQUIRED, "/opt/runtime"]:
        target = guest / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("runtime")
    return guest


@pytest.fixture
def args(tmp_path, contract):
    source = tmp_path / "sealed.qcow2"
    source.write_bytes(b"small synthetic qcow2")
    data = tmp_path / "data.json"
    data.write_text('{"release":"v1.1"}')
    contract.update(
        release="v1.1", source_sha256=build.sha256(source), data_manifest_sha256=build.sha256(data)
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(contract))
    key = tmp_path / "ssh-key"
    key.write_text("fake key, never used")
    return SimpleNamespace(
        source_qcow2=source,
        source_sha256=build.sha256(source),
        data_manifest=data,
        runtime_manifest=runtime,
        release="v1.1",
        image="localhost:5000/ale:candidate-unit",
        ssh_key=key,
        ssh_user="user",
        cpus=2,
        memory_gib=4,
        headroom_gib=16,
        workdir=tmp_path / "work",
        boot_timeout=1,
        preflight_only=False,
        overlay_budget_gib=16,
        export_budget_gib=120,
        docker_budget_gib=260,
        uefi_code=None,
        uefi_vars=None,
        builder_backend="native",
        runner_image="agentslastexam/ale-qemu:0.2.0",
        runner_image_id="sha256:" + "a" * 64,
        stage="all",
    )


def make_archive(root, contract):
    manifest = io.BytesIO()
    policy.export_manifest(root, contract, manifest)
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for item in manifest.getvalue().split(b"\0"):
            if item:
                name = item.decode()
                archive.add(root / name[2:], arcname=name, recursive=False)
    result.seek(0)
    return result


@pytest.mark.parametrize(
    "path",
    [
        "/media/user/data/agenthle/task/input/a",
        "/media/user/data/agenthle/task/reference.7z",
        "/home/user/.ale/task/output/result",
        "/home/user/.ale-src/secret/.env",
        "/etc/agenthle/gcs-reader.json",
        "/etc/boto.cfg",
        "/tmp/agenthle/gcs-reader.json",
        "/home/user/.config/gcloud/credentials.db",
        "/root/.ssh/id_ed25519",
        "/home/user/.ssh/authorized_keys",
        "/home/other/answers",
        "/root/.aws/credentials",
        "/home/user/project/.env.production",
        "/home/user/.codex/auth.json",
        "/home/user/.codex/config.toml",
        "/home/user/.codex/state_5.sqlite-wal",
        "/home/user/.codex/sessions/a",
        "/home/user/.hermes/auth.json",
        "/home/user/.hermes/state.db-wal",
        "/home/user/.hermes/config.yaml",
        "/home/user/.claude/settings.json",
        "/home/user/.claude/settings.local.json",
        "/home/user/.grok/user-settings.json",
        "/home/user/.gemini/settings.json",
        "/home/user/.openhands/conversations/task/result",
        "/home/user/.openclaw/openclaw.json.bak",
        "/home/user/.kimi/config.toml",
        "/home/user/.claude/projects/state",
        "/home/user/.openclaw/agents/default/session",
        "/home/user/.config/google-chrome/Profile 2/a",
        "/home/user/.local/share/docker/overlay2/layer",
        "/var/lib/docker/data",
        "/var/lib/containerd/state",
        "/var/lib/dind/layer",
        "/opt/ale-docker-images/a.tar",
        "/home/user/.bash_history",
        "/home/user/.agenthle_hidden_eval_assets/a",
        "/var/lib/cloud/instance/user-data.txt",
        "/etc/ssh/ssh_host_ed25519_key",
        "/var/lib/ale-export-manifest.abc",
        "/home/user/.cache/pip/cache",
    ],
)
def test_sensitive_paths_excluded_before_archive(root, contract, path):
    sensitive = root / path.lstrip("/")
    sensitive.parent.mkdir(parents=True, exist_ok=True)
    sensitive.write_text("synthetic-secret")
    manifest = io.BytesIO()
    policy.export_manifest(root, contract, manifest)
    assert ("." + path).encode() not in manifest.getvalue().split(b"\0")
    assert policy.forbidden(path, contract)
    assert policy.inspect_archive(make_archive(root, contract), contract)["members"] > 0


@pytest.mark.parametrize("path", TASK_RESIDUE_ROOTS)
def test_task_residue_roots_pruned_before_export(root, contract, path):
    residue = root / path.lstrip("/")
    (residue / "empty").mkdir(parents=True)
    (residue / "payload").write_text("synthetic task input or reference")
    manifest = io.BytesIO()
    policy.export_manifest(root, contract, manifest)
    names = manifest.getvalue().decode().split("\0")
    assert not any(name == "." + path or name.startswith("." + path + "/") for name in names)
    assert policy.forbidden(path, contract)
    assert policy.forbidden("." + path + "/payload", contract)
    assert policy.inspect_archive(make_archive(root, contract), contract)["members"] > 0
    assert audit.inspect_tree(root, contract)["export_ready"]
    assert (residue / "payload").read_text() == "synthetic task input or reference"
    contract["required_paths"].append(path)
    with pytest.raises(ValueError, match="required runtime excluded"):
        policy.export_manifest(root, contract, io.BytesIO())


@pytest.mark.parametrize("path", TASK_RESIDUE_ROOTS)
@pytest.mark.parametrize("relative", [False, True])
def test_runtime_links_cannot_retain_task_residue(root, contract, path, relative):
    payload = root / path.lstrip("/") / "payload"
    payload.parent.mkdir()
    payload.write_text("synthetic task payload")
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    target = "../.." + path + "/payload" if relative else path + "/payload"
    (runtime / "task-alias").symlink_to(target)
    with pytest.raises(ValueError, match="excluded or missing symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())
    assert not audit.inspect_tree(root, contract)["export_ready"]


@pytest.mark.parametrize("path", TASK_RESIDUE_ROOTS)
@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_archive_rejects_links_into_task_residue(contract, path, member_type):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        member = tarfile.TarInfo("./opt/task-alias")
        member.type = member_type
        member.linkname = (
            ".." + path + "/payload" if member_type == tarfile.SYMTYPE else path[1:] + "/payload"
        )
        output.addfile(member)
    archive.seek(0)
    with pytest.raises(ValueError, match="archive link targets excluded content"):
        policy.inspect_archive(archive, contract)


@pytest.mark.parametrize(
    "path",
    [
        "/opt/science/model.bin",
        "/usr/lib/R/library/stats/R/stats.rdb",
        "/home/user/R/x86_64-pc-linux-gnu-library/4.3/package/libs/package.so",
        "/home/user/.local/share/uv/python/cpython/bin/python3",
        "/home/user/.hermes/venv/bin/python",
        "/home/user/.openclaw/node_modules/runtime.js",
        "/home/user/.openclaw/extensions/plugin/node_modules/dependency/index.js",
        "/home/user/.grok/install.json",
        "/home/user/.grok/bin/grok",
        "/home/user/.cargo/bin/rustc",
        "/home/user/.npm-global/bin/agent",
        "/usr/share/doc/tool/reference/usage.md",
        "/opt/runtime/workspace/model.bin",
        "/opt/runtime/protected/api.py",
        "/opt/runtime/output/template.json",
        "/workspace-tools/bin/runtime",
        "/reference-library/package.py",
        "/protected-runtime/bin/tool",
        "/output-format/schema.json",
        "/output_test_positive/docs",
        "/output_test_negative/docs",
    ],
)
def test_runtime_assets_not_blanket_removed(contract, path):
    assert not policy.forbidden(path, contract)


def test_vm_floppy_alias_and_mount_are_excluded(root, contract):
    media = root / "media"
    media.mkdir()
    (media / "floppy0").mkdir()
    (media / "floppy0" / "vm-media").write_text("VM-only fixture")
    (media / "floppy").symlink_to("floppy0")
    archive = make_archive(root, contract)
    assert policy.inspect_archive(archive, contract)["members"] > 0
    archive.seek(0)
    with tarfile.open(fileobj=archive) as exported:
        assert not any(member.name.startswith("./media/floppy") for member in exported)
    contract["required_paths"].append("/media/floppy")
    with pytest.raises(ValueError, match="required runtime excluded"):
        policy.export_manifest(root, contract, io.BytesIO())


def test_cache_backed_runtime_requires_explicit_audited_preservation(root, contract):
    target = root / "home/user/.cache/uv/python/bin/python"
    target.parent.mkdir(parents=True)
    target.write_text("python-runtime")
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.symlink_to("/home/user/.cache/uv/python/bin/python")
    with pytest.raises(ValueError, match="required runtime excluded"):
        policy.export_manifest(root, contract, io.BytesIO())
    contract["preserve_cache_paths"] = ["/home/user/.cache/uv/python"]
    archive = make_archive(root, contract)
    assert policy.inspect_archive(archive, contract)["members"] > 0
    assert not policy.forbidden("/home/user/.cache/uv/python/bin/python", contract)
    assert policy.forbidden("/home/user/.cache/uv/unneeded-cache", contract)
    assert policy.forbidden("/home/user/.cache/uv/python/.env", contract)


@pytest.mark.parametrize(
    "target", ["/run/service/socket", "/proc/self/mounts", "/dev/log", "/usr/share/doc/optional"]
)
def test_nonruntime_os_links_do_not_require_live_targets(root, contract, target):
    link = root / "etc/optional-alias"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    archive = make_archive(root, contract)
    assert policy.inspect_archive(archive, contract)["members"] > 0
    assert audit.inspect_tree(root, contract)["export_ready"]
    archive.seek(0)
    with tarfile.open(fileobj=archive) as exported:
        assert exported.getmember("./etc/optional-alias").linkname == target


@pytest.mark.parametrize(
    "target", ["/run/missing-runtime", "/usr/lib/missing-runtime", "/home/user/.ssh/id_ed25519"]
)
def test_required_runtime_links_still_fail_closed(root, contract, target):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    (runtime / "dependency").symlink_to(target)
    with pytest.raises(ValueError, match="symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())
    assert not audit.inspect_tree(root, contract)["export_ready"]


def test_exact_source_baseline_missing_link_is_preserved(root, contract):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    (runtime / "unused-alias").symlink_to("missing-original-target")
    contract["baseline_missing_links"] = {
        "/opt/runtime/unused-alias": "/opt/runtime/missing-original-target"
    }
    archive = make_archive(root, contract)
    assert policy.inspect_archive(archive, contract)["members"] > 0
    assert audit.inspect_tree(root, contract)["export_ready"]
    archive.seek(0)
    with tarfile.open(fileobj=archive) as exported:
        assert (
            exported.getmember("./opt/runtime/unused-alias").linkname == "missing-original-target"
        )
    (runtime / "new-broken-dependency").symlink_to("missing-original-target")
    with pytest.raises(ValueError, match="symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())


def test_changed_baseline_target_is_rejected(root, contract):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    (runtime / "alias").symlink_to("different-missing-target")
    contract["baseline_missing_links"] = {"/opt/runtime/alias": "/opt/runtime/original-target"}
    with pytest.raises(ValueError, match="symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())
    assert not audit.inspect_tree(root, contract)["export_ready"]


def test_archive_cannot_change_allowlisted_missing_target(root, contract):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    (runtime / "alias").symlink_to("original-target")
    contract["baseline_missing_links"] = {"/opt/runtime/alias": "/opt/runtime/original-target"}
    source = make_archive(root, contract)
    changed = io.BytesIO()
    with (
        tarfile.open(fileobj=source) as original,
        tarfile.open(fileobj=changed, mode="w") as output,
    ):
        for member in original:
            if member.name == "./opt/runtime/alias":
                member.linkname = "different-target"
            output.addfile(member, original.extractfile(member) if member.isfile() else None)
    changed.seek(0)
    with pytest.raises(ValueError, match="missing runtime or link target"):
        policy.inspect_archive(changed, contract)


@pytest.mark.parametrize(
    "baseline",
    [
        [],
        {"/opt/runtime": "/usr/lib/absent"},
        {"relative": "/usr/lib/absent"},
        {"/opt/runtime/alias": "/home/user/.ssh/private"},
        {"/opt/runtime/alias": "/media/user/data/agenthle/reference"},
        {"/opt/runtime/alias": "/opt/../escape"},
    ],
)
def test_baseline_cannot_relax_required_or_excluded_paths(contract, baseline):
    contract["baseline_missing_links"] = baseline
    with pytest.raises(ValueError):
        policy.validate_contract(contract)


def test_required_r_library_symlinks_are_checked_recursively(root, contract):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.mkdir()
    (runtime / "R-package").symlink_to("/media/user/data/agenthle/package")
    with pytest.raises(ValueError, match="symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())


def test_relative_runtime_symlinks_are_preserved(root, contract):
    runtime = root / "opt/runtime"
    runtime.unlink()
    runtime.symlink_to("../usr/local/bin/node")
    assert policy.inspect_archive(make_archive(root, contract), contract)["members"] > 0


def test_missing_runtime_fails_before_export(root, contract):
    (root / "opt/runtime").unlink()
    with pytest.raises(ValueError, match="missing runtime"):
        policy.export_manifest(root, contract, io.BytesIO())


def test_unreadable_export_tree_is_fatal(root, contract, monkeypatch):
    def broken_walk(*argv, **kwargs):
        kwargs["onerror"](PermissionError("unreadable runtime tree"))

    monkeypatch.setattr(policy.os, "walk", broken_walk)
    with pytest.raises(SystemExit, match="unreadable runtime tree"):
        policy.export_manifest(root, contract, io.BytesIO())


@pytest.mark.parametrize("path", ["/root/.ssh", "/home/user/.cache", "/media/user/data"])
def test_cannot_whitelist_secret_or_general_cache(contract, path):
    contract["preserve_cache_paths"] = [path]
    with pytest.raises(ValueError, match="only audited"):
        policy.validate_contract(contract)


@pytest.mark.parametrize("name", ["../secret", "./a/../b", "a\nname", "a\0name"])
def test_unsafe_names_rejected(name):
    with pytest.raises(ValueError, match="unsafe"):
        policy.canonical(name)


def test_dotfiles_are_not_renamed():
    assert policy.canonical("./.cache/a") == "/.cache/a"
    assert policy.canonical("/.cache/a") == "/.cache/a"


def test_compiled_component_filter_matches_original_glob_contract():
    samples = {"normal.py", ".envexample", ".cache", "reference", "task-data", "nested/path"}
    for pattern in policy.COMPONENTS:
        sample = pattern.replace("*", "test")
        samples.update((pattern, sample, sample + ".bak", "prefix" + sample, sample + "/nested"))
    for sample in samples:
        expected = any(fnmatch.fnmatchcase(sample, pattern) for pattern in policy.COMPONENTS)
        assert bool(policy.COMPONENT_PATTERN.fullmatch(sample)) == expected


@pytest.mark.parametrize(
    "name",
    ["./etc/agenthle/key", "./home/user/.env", "../escape"]
    + ["." + path for path in TASK_RESIDUE_ROOTS]
    + ["." + path + "/payload" for path in TASK_RESIDUE_ROOTS],
)
def test_archive_rejects_forbidden_members(contract, name):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        output.addfile(tarfile.TarInfo(name))
    archive.seek(0)
    with pytest.raises(ValueError):
        policy.inspect_archive(archive, contract)


def test_archive_rejects_missing_runtime(contract):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        output.addfile(tarfile.TarInfo("./only-one-file"))
    archive.seek(0)
    with pytest.raises(ValueError, match="missing runtime"):
        policy.inspect_archive(archive, contract)


def test_archive_rejects_broken_required_symlink(root, contract):
    archive = make_archive(root, contract)
    data = archive.getvalue()
    altered = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(data)) as original,
        tarfile.open(fileobj=altered, mode="w") as output,
    ):
        for member in original:
            if member.name == "./opt/runtime":
                member.type = tarfile.SYMTYPE
                member.linkname = "/absent/python"
                member.size = 0
                output.addfile(member)
            else:
                output.addfile(member, original.extractfile(member) if member.isfile() else None)
    altered.seek(0)
    with pytest.raises(ValueError, match="missing runtime"):
        policy.inspect_archive(altered, contract)


def test_cache_requires_identity_hash_and_size(tmp_path):
    archive = tmp_path / "rootfs.tar.zst"
    receipt = tmp_path / "export.json"
    identity = {"release": "v1.1", "source_sha256": "source", "scripts": {"build": "hash"}}
    assert not build.cached_export(archive, receipt, identity)
    archive.write_bytes(b"tiny archive")
    with pytest.raises(ValueError, match="incomplete"):
        build.cached_export(archive, receipt, identity)
    inventory = tmp_path / "packages.before.txt"
    inventory.write_text("python\t1\n")
    build.write_json(
        receipt,
        {
            "identity": identity,
            "sha256": build.sha256(archive),
            "size": archive.stat().st_size,
            "builder_evidence": str(tmp_path),
            "package_inventory_sha256": build.sha256(inventory),
        },
    )
    assert build.cached_export(archive, receipt, identity)
    for key in ("release", "source_sha256", "scripts"):
        with pytest.raises(ValueError, match="identity mismatch"):
            build.cached_export(archive, receipt, {**identity, key: "wrong"})
    archive.write_bytes(b"tiny archivf")
    with pytest.raises(ValueError, match="integrity"):
        build.cached_export(archive, receipt, identity)
    archive.write_bytes(b"short")
    with pytest.raises(ValueError, match="integrity"):
        build.cached_export(archive, receipt, identity)


def test_source_and_release_binding(args, monkeypatch):
    monkeypatch.setattr(build, "persistent_mount", lambda *argv, **kwargs: {})
    monkeypatch.setattr(
        build, "capture", lambda *argv: json.dumps([{"format": "qcow2", "virtual-size": 1024}])
    )
    identity, _, size = build.source_identity(args)
    assert size == 1024
    assert identity["source_sha256"] == args.source_sha256
    assert "cleanup.sh" in identity["scripts"]
    args.release = "v1.0"
    with pytest.raises(ValueError, match="release mismatch"):
        build.source_identity(args)
    args.release = "v1.1"
    args.source_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build.source_identity(args)


@pytest.mark.parametrize(
    "info",
    [
        [{"format": "raw"}],
        [{"format": "qcow2", "backing-filename": "base"}],
        [{"format": "qcow2"}, {"format": "qcow2"}],
        [{"format": "qcow2", "dirty-flag": True}],
        [{"format": "qcow2", "snapshots": [{"id": "1"}]}],
        [{"format": "qcow2", "format-specific": {"data": {"data-file": "outside.raw"}}}],
        [{"format": "qcow2", "encrypted": True}],
    ],
)
def test_source_rejects_nonfinal_disks(args, monkeypatch, info):
    monkeypatch.setattr(build, "persistent_mount", lambda *argv, **kwargs: {})
    monkeypatch.setattr(build, "capture", lambda *argv: json.dumps(info))
    with pytest.raises(ValueError):
        build.source_identity(args)


@pytest.mark.parametrize("filesystem", ["tmpfs", "ramfs", "overlay", "nfs", "unknown"])
def test_storage_rejects_unverified_filesystems(tmp_path, monkeypatch, filesystem):
    monkeypatch.setattr(build, "PERSISTENT_ROOT", tmp_path)
    monkeypatch.setattr(
        build,
        "capture",
        lambda *argv: json.dumps(
            {
                "filesystems": [
                    {
                        "fstype": filesystem,
                        "options": "rw",
                        "source": "test",
                        "target": str(tmp_path),
                    }
                ]
            }
        ),
    )
    with pytest.raises(ValueError, match="persistent storage"):
        build.persistent_mount(tmp_path / "not-created-yet")


def test_storage_resolves_symlink_and_checks_containing_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "ram"
    elsewhere.mkdir()
    (home / "link").symlink_to(elsewhere)
    monkeypatch.setattr(build, "PERSISTENT_ROOT", home)
    with pytest.raises(ValueError, match="must be under"):
        build.persistent_mount(home / "link" / "export")
    calls = []

    def capture(*argv):
        calls.append(argv)
        if argv[0] == "df":
            return "Avail\n999999"
        return json.dumps(
            {
                "filesystems": [
                    {"fstype": "ext4", "options": "rw", "source": "/dev/sda", "target": "/"}
                ]
            }
        )

    monkeypatch.setattr(build, "capture", capture)
    assert build.persistent_mount(home / "new" / "export")["available"] == 999999
    assert calls[0][3] == home


def test_capacity_combines_shared_device_budgets_and_retained_vm_peaks(monkeypatch):
    monkeypatch.setattr(
        build,
        "persistent_mount",
        lambda *args, **kwargs: {"available": 100 * build.GIB, "device": 1, "path": "disk"},
    )
    memory = {"available": 30 * build.GIB, "unresident_vm_reservation": 10 * build.GIB}
    monkeypatch.setattr(build, "host_memory", lambda: memory)
    options = dict(
        work_bytes=50 * build.GIB, docker_bytes=50 * build.GIB, memory_gib=8, headroom_gib=16
    )
    with pytest.raises(ValueError, match="capacity"):
        build.capacity(Path("work"), Path("docker"), **options)
    options.update(work_bytes=10 * build.GIB, docker_bytes=10 * build.GIB)
    with pytest.raises(ValueError, match="RAM"):
        build.capacity(Path("work"), Path("docker"), **options)
    memory["available"] = 40 * build.GIB
    assert build.capacity(Path("work"), Path("docker"), **options)


def test_proc_inventory_includes_paused_live_qemu_processes(tmp_path):
    (tmp_path / "meminfo").write_text("MemAvailable: 90000000 kB\nShmem: 1234 kB\n")
    process = tmp_path / "42"
    process.mkdir()
    (process / "comm").write_text("qemu-system-x86\n")
    (process / "cmdline").write_bytes(b"qemu-system-x86_64\0-m\0size=16G\0-S\0")
    (process / "status").write_text("State: T (stopped)\nVmRSS: 1048576 kB\n")
    result = build.host_memory(tmp_path)
    assert result["unresident_vm_reservation"] == 15 * build.GIB
    assert result["vms"][0]["pid"] == 42
    (process / "cmdline").write_bytes(b"qemu-system-x86_64\0-m\0unknown\0")
    with pytest.raises(ValueError, match="cannot inspect"):
        build.host_memory(tmp_path)


@pytest.mark.parametrize(
    "status,body",
    [
        (500, b'{"status":"ok"}'),
        (200, b"error"),
        (200, b"{}"),
        (200, b"[]"),
        (200, b'"ok"'),
        (200, b'{"status":"error"}'),
        (200, b'{"status":true}'),
        (200, b"x" * 65537),
    ],
)
def test_health_fails_closed(status, body):
    assert not build.healthy_status(status, body)


def test_health_accepts_documented_json():
    assert build.healthy_status(200, b'{"status":"ok"}')


def cli_args(args):
    return [
        "--source-qcow2",
        str(args.source_qcow2),
        "--source-sha256",
        args.source_sha256,
        "--release",
        args.release,
        "--data-manifest",
        str(args.data_manifest),
        "--runtime-manifest",
        str(args.runtime_manifest),
        "--workdir",
        str(args.workdir),
        "--image",
        args.image,
        "--ssh-key",
        str(args.ssh_key),
    ]


def test_cli_requires_explicit_inputs_and_candidate(args, monkeypatch):
    with pytest.raises(SystemExit):
        build.parse_args([])
    assert build.parse_args(cli_args(args)).source_qcow2 == args.source_qcow2
    for tag in ("ale:latest", "ale", "ale:v1.1", "ale:base"):
        args.image = tag
        with pytest.raises(SystemExit):
            build.parse_args(cli_args(args))
    args.image = "ale:candidate-unit"
    monkeypatch.setenv("ALE_PUSH_IMAGE", "1")
    with pytest.raises(SystemExit):
        build.parse_args(cli_args(args))


def test_wrapper_help_needs_neither_cloud_nor_docker():
    result = subprocess.run(
        ["bash", str(SCRIPTS / "build.sh"), "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    assert "--source-qcow2" in result.stdout
    assert "dev-ubuntu22" not in result.stdout


class FakeProcess:
    def __init__(self, stdout=b"", code=None):
        self.stdout = io.BytesIO(stdout)
        self.code = code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.code is None:
            self.code = 0
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 0

    def kill(self):
        self.killed = True
        self.code = -9


def test_builder_uses_only_disposable_overlay_and_loopback_ssh(args, tmp_path, monkeypatch):
    calls = []
    process = FakeProcess()
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(
        build.subprocess, "Popen", lambda argv, **kwargs: calls.append(tuple(argv)) or process
    )
    builder = build.Builder(args, tmp_path)
    monkeypatch.setattr(builder, "cua_command", lambda *argv, **kwargs: "ALE_BUILDER_READY")
    monkeypatch.setattr(builder, "bootstrap_ssh", lambda **kwargs: None)
    builder.start()
    assert calls[0][:6] == ("qemu-img", "create", "-f", "qcow2", "-F", "qcow2")
    assert calls[0][-1] == tmp_path / "builder.qcow2"
    qemu = calls[1]
    block = json.loads(qemu[qemu.index("-blockdev") + 1])
    assert block["file"]["filename"] == str(tmp_path / "builder.qcow2")
    assert "restrict=on,hostfwd=tcp:127.0.0.1:" in qemu[qemu.index("-netdev") + 1]
    assert f"hostfwd=tcp:127.0.0.1:{builder.cua_port}-:5000" in qemu[qemu.index("-netdev") + 1]
    assert builder.cua_port != builder.port
    assert "StrictHostKeyChecking=yes" in builder.ssh
    assert calls[-1][0] == "ssh"
    assert all("gcloud" not in command for command in calls)
    builder.close()
    assert process.terminated
    assert args.source_qcow2.read_bytes() == b"small synthetic qcow2"


def test_failed_builder_boot_can_always_be_closed(args, tmp_path, monkeypatch):
    process = FakeProcess(code=1)
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: None)
    monkeypatch.setattr(build.subprocess, "Popen", lambda *argv, **kwargs: process)
    builder = build.Builder(args, tmp_path)
    with pytest.raises(RuntimeError, match="exited"):
        builder.start()
    builder.close()
    assert builder.log.closed


@pytest.mark.parametrize("failure", ["cleanup", "runtime", "import", None])
def test_finalize_checks_failures_and_never_prunes_or_pushes(
    args, contract, tmp_path, monkeypatch, failure
):
    calls = []

    def fake_run(*argv, **kwargs):
        calls.append(argv)
        if "dpkg-query" in argv:
            kwargs["stdout"].write(b"python\t1\n")
        if (
            (failure == "cleanup" and argv[-1] == "/root/cleanup.sh")
            or (failure == "runtime" and "exec" in argv and "--user" in argv)
            or (failure == "import" and argv[:2] == ("docker", "import"))
        ):
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(build, "run", fake_run)
    monkeypatch.setattr(build, "capture", lambda *argv: "sha256:fake")
    monkeypatch.setattr(build, "smoke", lambda *argv: calls.append(("smoke",)))
    process = FakeProcess()
    monkeypatch.setattr(build.subprocess, "Popen", lambda *argv, **kwargs: process)
    before = tmp_path / "packages.before.txt"
    before.write_text("python\t1\n")
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            build.finalize(tmp_path / "fake.zst", args.image, tmp_path, contract, args, before)
        assert not any(command[:2] == ("docker", "commit") for command in calls)
        assert ("smoke",) not in calls
    else:
        assert build.finalize(tmp_path / "fake.zst", args.image, tmp_path, contract, args, before)
        assert ("smoke",) in calls
    if failure != "import":
        assert any(command[:3] == ("docker", "rm", "-f") for command in calls)
    assert not any("push" in command or "rmi" in command or "prune" in command for command in calls)


@pytest.mark.parametrize("corrupt", [False, True])
def test_streaming_archive_audit_checks_compression_and_tar_termination(
    root, contract, monkeypatch, corrupt
):
    data = make_archive(root, contract).getvalue()
    if corrupt:
        data = data[:-1]
    process = FakeProcess(data)
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: None)
    monkeypatch.setattr(build.subprocess, "Popen", lambda *argv, **kwargs: process)
    if corrupt:
        with pytest.raises(ValueError, match="truncated"):
            build.audit_export(Path("tiny.zst"), contract)
    else:
        assert build.audit_export(Path("tiny.zst"), contract)["members"] > 0
    assert process.stdout.closed


def test_preflight_does_not_allocate_builder_or_image(args, monkeypatch):
    args.preflight_only = True
    monkeypatch.setattr(build.shutil, "which", lambda *argv: "/fake/tool")
    monkeypatch.setattr(build.os, "access", lambda *argv: True)
    monkeypatch.setattr(build, "persistent_mount", lambda *argv, **kwargs: {})
    monkeypatch.setattr(build, "source_identity", lambda *argv: ({"source": "test"}, {}, 1024))
    monkeypatch.setattr(build, "docker_storage", lambda: Path("/disk/docker"))
    monkeypatch.setattr(build, "capacity", lambda *argv, **kwargs: {"checked": True})
    monkeypatch.setattr(build, "Builder", lambda *argv: pytest.fail("must not create VM"))
    monkeypatch.setattr(build, "finalize", lambda *argv: pytest.fail("must not import"))
    build.build(args)
    assert (args.workdir / "preflight.json").is_file()
    assert not (args.workdir / "rootfs.tar.zst").exists()


def test_cleanup_sanity_block_returns_nonzero_for_missing_runtime():
    script = (SCRIPTS / "cleanup.sh").read_text()
    sanity = script[script.index('echo "--- verify image-promised paths ---"') :]
    result = subprocess.run(["bash", "-c", sanity], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "CLEANUP_FAILED" in result.stderr
    assert "rm -rf /home/user/.cache" not in script
    assert 'echo "FATAL: XFCE install failed"' in script


@pytest.mark.parametrize("script", ["build.sh", "export_rootfs.sh", "cleanup.sh", "entrypoint.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(SCRIPTS / script)], check=True, timeout=10)


@pytest.mark.parametrize("invalid", [[], [["true"]], [["/bin/true"]], [["/usr/bin/python3", "-V"]]])
def test_runtime_contract_requires_python_and_r(contract, invalid):
    contract["probes"] = invalid
    with pytest.raises(ValueError):
        policy.validate_contract(contract)


def test_explicit_source_audit_exclusions_cannot_hide_required_runtime(root, contract):
    contract["exclude_paths"] = ["/opt/runtime"]
    with pytest.raises(ValueError, match="required runtime excluded"):
        policy.export_manifest(root, contract, io.BytesIO())


@pytest.mark.parametrize("after", ["python\t2\n", "R\t1\n", ""])
def test_package_inventory_rejects_upgrades_removals_and_empty(tmp_path, after):
    before_path = tmp_path / "before"
    before_path.write_text("python\t1\n")
    after_path = tmp_path / "after"
    after_path.write_text(after)
    with pytest.raises(ValueError):
        build.compare_packages(before_path, after_path)


def test_package_inventory_records_added_desktop_packages(tmp_path):
    before = tmp_path / "before"
    before.write_text("python\t1\nR\t1\n")
    after = tmp_path / "after"
    after.write_text("python\t1\nR\t1\nxfwm4\t4.18\n")
    assert build.compare_packages(before, after) == {"xfwm4": "4.18"}


@pytest.mark.parametrize("valid", [False, True])
def test_smoke_checks_http_and_cleans_own_container(args, tmp_path, monkeypatch, valid):
    calls = []
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(build, "capture", lambda *argv: "12345")
    monkeypatch.setattr(build.subprocess, "run", lambda *argv, **kwargs: None)
    monkeypatch.setattr(build.time, "sleep", lambda *argv: None)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *argv):
            pass

        def read(self, size):
            return b'{"status":"ok"}' if valid else b"error page"

    monkeypatch.setattr(build.urllib.request, "urlopen", lambda *argv, **kwargs: Response())
    if valid:
        build.smoke(args.image, "owned-smoke", tmp_path, args)
        assert any(command[:3] == ("docker", "exec", "owned-smoke") for command in calls)
    else:
        with pytest.raises(RuntimeError, match="valid HTTP"):
            build.smoke(args.image, "owned-smoke", tmp_path, args)
    assert calls[-1] == ("docker", "rm", "-f", "owned-smoke")
    assert "--privileged" not in calls[0]
    assert "127.0.0.1::5000" in calls[0]


def test_uefi_uses_disposable_variable_store(args, tmp_path, monkeypatch):
    args.uefi_code = tmp_path / "code.fd"
    args.uefi_code.write_bytes(b"code")
    args.uefi_vars = tmp_path / "vars.fd"
    args.uefi_vars.write_bytes(b"template")
    directory = tmp_path / "builder"
    directory.mkdir()
    commands = []
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: None)
    process = FakeProcess()
    monkeypatch.setattr(
        build.subprocess, "Popen", lambda argv, **kwargs: commands.append(argv) or process
    )
    builder = build.Builder(args, directory)
    monkeypatch.setattr(builder, "cua_command", lambda *argv, **kwargs: "ALE_BUILDER_READY")
    monkeypatch.setattr(builder, "bootstrap_ssh", lambda **kwargs: None)
    builder.start()
    builder.close()
    assert (directory / "uefi-vars.fd").read_bytes() == b"template"
    assert args.uefi_vars.read_bytes() == b"template"
    assert f"if=pflash,format=raw,readonly=on,file={args.uefi_code}" in commands[0]
    assert commands[0][commands[0].index("-machine") + 1] == "q35,smm=off"


@pytest.mark.parametrize("failure", [None, "capacity", "export", "source-mutated", "archive"])
def test_build_orders_checks_and_always_stops_builder(args, contract, monkeypatch, failure):
    calls = []
    identity = {"source_sha256": args.source_sha256, "release": "v1.1"}
    monkeypatch.setattr(build.shutil, "which", lambda *argv: "/fake/tool")
    monkeypatch.setattr(build.os, "access", lambda *argv: True)
    monkeypatch.setattr(build, "persistent_mount", lambda *argv, **kwargs: {})
    monkeypatch.setattr(build, "source_identity", lambda *argv: (identity, contract, 1024))
    monkeypatch.setattr(build, "docker_storage", lambda: Path("/fake/docker"))
    monkeypatch.setattr(build, "capture", lambda *argv: "")

    def capacity(*argv, **kwargs):
        calls.append("capacity")
        if failure == "capacity" and calls.count("capacity") == 3:
            raise ValueError("capacity changed")
        return {"safe": True}

    def audit(*argv):
        calls.append("audit")
        if failure == "archive":
            raise ValueError("archive failed")
        return {"logical_bytes": 1024}

    class FakeBuilder:
        def __init__(self, args, directory):
            self.directory = directory

        def start(self):
            calls.append("start")

        def export(self, path):
            calls.append("export")
            if failure == "export":
                raise RuntimeError("export failed")
            path.write_bytes(b"tiny fixture")
            (self.directory / "packages.before.txt").write_text("python\t1\n")
            if failure == "source-mutated":
                args.source_qcow2.write_bytes(b"changed fixture")

        def close(self):
            calls.append("close")

    def finalize(*argv):
        calls.append("finalize")
        assert calls[-2] == "capacity"
        return "sha256:unit"

    monkeypatch.setattr(build, "capacity", capacity)
    monkeypatch.setattr(build, "audit_export", audit)
    monkeypatch.setattr(build, "Builder", FakeBuilder)
    monkeypatch.setattr(build, "finalize", finalize)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            build.build(args)
        assert "finalize" not in calls
        assert not (args.workdir / "export.json").exists()
    else:
        build.build(args)
        assert calls.index("audit") < calls.index("finalize")
        assert calls[calls.index("start") - 1] == "capacity"
        assert (args.workdir / "export.json").exists()
        calls.clear()
        build.build(args)
        assert "start" not in calls
        assert "export" not in calls
        assert "audit" in calls
        assert "finalize" in calls
    assert calls[-1] == ("close" if failure else "finalize")


def test_docker_remote_daemon_rejected(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(
        build,
        "capture",
        lambda *argv: json.dumps([{"Endpoints": {"docker": {"Host": "tcp://remote:2375"}}}]),
    )
    with pytest.raises(ValueError, match="local unix"):
        build.docker_storage()


def test_tar_pipeline_propagates_failure_without_rc_side_channel(tmp_path):
    script = (SCRIPTS / "export_rootfs.sh").read_text()
    assert "set -euo pipefail" in script
    assert "--no-recursion --null --verbatim-files-from" in script
    assert "--one-file-system" not in script
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "exit 2 | cat"], capture_output=True, timeout=10
    )
    assert result.returncode == 2


def test_real_gnu_tar_and_zstd_round_trip_is_small_and_filtered(root, contract, tmp_path):
    manifest = io.BytesIO()
    policy.export_manifest(root, contract, manifest)
    archive = subprocess.run(
        [
            "tar",
            "--numeric-owner",
            "--no-recursion",
            "--null",
            "--verbatim-files-from",
            "-C",
            str(root),
            "-T",
            "-",
            "-cf",
            "-",
        ],
        input=manifest.getvalue(),
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout
    assert len(archive) < 100_000
    path = tmp_path / "rootfs.tar.zst"
    with path.open("wb") as output:
        subprocess.run(["zstd", "-c"], input=archive, stdout=output, check=True, timeout=10)
    assert build.audit_export(path, contract)["members"] > 0
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(subprocess.CalledProcessError):
        build.audit_export(path, contract)


def test_no_source_ssh_identity_is_required(args):
    parsed = build.parse_args(cli_args(args)[:-2])
    assert parsed.ssh_key is None


@pytest.mark.parametrize("provided", [False, True])
def test_builder_public_key_only_bootstrap(args, tmp_path, monkeypatch, provided):
    if not provided:
        args.ssh_key = None
    builder = build.Builder(args, tmp_path)
    calls = []
    monkeypatch.setattr(build, "persistent_mount", lambda *argv: calls.append(("mount", *argv)))
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(build, "capture", lambda *argv: calls.append(argv) or PUBLIC_KEY)

    def cua(command, **kwargs):
        calls.append(("cua", command))
        assert PUBLIC_KEY in command
        assert "fake key, never used" not in command
        assert "PRIVATE KEY" not in command
        assert str(builder.ssh_key) not in command
        assert "sudo -n python3 -c" in command
        return PUBLIC_KEY

    monkeypatch.setattr(builder, "cua_command", cua)
    builder.bootstrap_ssh(timeout=5)
    assert calls[0] == ("mount", tmp_path)
    generation = [command for command in calls if command[:2] == ("ssh-keygen", "-q")]
    assert bool(generation) is not provided
    assert builder.ssh_key == (args.ssh_key if provided else tmp_path / "id_ed25519")
    assert (tmp_path / "known_hosts").read_text() == f"[127.0.0.1]:{builder.port} {PUBLIC_KEY}\n"
    assert (tmp_path / "known_hosts").stat().st_mode & 0o777 == 0o600
    assert json.loads((tmp_path / "ssh-bootstrap.json").read_text())["public_key_sha256"]


def test_builder_key_generation_checks_persistent_storage_first(args, tmp_path, monkeypatch):
    args.ssh_key = None
    builder = build.Builder(args, tmp_path)

    def unsafe(*argv):
        raise ValueError("tmpfs")

    monkeypatch.setattr(build, "persistent_mount", unsafe)
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: pytest.fail("must not create key"))
    with pytest.raises(ValueError, match="tmpfs"):
        builder.bootstrap_ssh(timeout=5)


class CuaResponse(io.BytesIO):
    def __init__(self, body, status=200, url=""):
        super().__init__(body)
        self.status = status
        self.url = url

    def geturl(self):
        return self.url


@pytest.mark.parametrize(
    "body,status",
    [
        (b'data: {"success":true,"return_code":0,"stdout":"ok"}\n', 503),
        (b'data: {"success":false,"return_code":0,"stdout":"ok"}\n', 200),
        (b'data: {"success":true,"return_code":1,"stdout":"ok"}\n', 200),
        (b'data: {"success":true,"stdout":"ok"}\n', 200),
        (b'data: {"success":true,"return_code":false,"stdout":"ok"}\n', 200),
        (b'data: {"success":true,"return_code":"0","stdout":"ok"}\n', 200),
        (b'data: {"success":true,"return_code":0,"stdout":{}}\n', 200),
        (b"data: [1,2]\n", 200),
        (b"data: not-json\n", 200),
        (b"not SSE", 200),
        (b"x" * 65537, 200),
    ],
)
def test_cua_bootstrap_response_fails_closed(args, tmp_path, body, status):
    builder = build.Builder(args, tmp_path)
    builder.process = FakeProcess()
    url = f"http://127.0.0.1:{builder.cua_port}/cmd"
    response = CuaResponse(body, status, url)
    builder.http = SimpleNamespace(open=lambda *argv, **kwargs: response)
    with pytest.raises(ValueError):
        builder.cua_command("true", timeout=1)
    assert response.closed


def test_cua_bootstrap_uses_existing_sse_wire_contract(args, tmp_path):
    builder = build.Builder(args, tmp_path)
    builder.process = FakeProcess()

    def open_request(request, timeout):
        assert request.full_url == f"http://127.0.0.1:{builder.cua_port}/cmd"
        assert request.get_method() == "POST"
        assert timeout == 1
        assert json.loads(request.data) == {
            "command": "run_command",
            "params": {"command": "printf ready"},
        }
        return CuaResponse(
            b': keepalive\n\ndata: {"success":true,"return_code":0,"stdout":"ready"}\n',
            url=request.full_url,
        )

    builder.http = SimpleNamespace(open=open_request)
    assert builder.cua_command("printf ready", timeout=1) == "ready"


@pytest.mark.parametrize("process", [None, FakeProcess(code=1)])
def test_cua_will_not_bootstrap_without_own_live_builder(args, tmp_path, process):
    builder = build.Builder(args, tmp_path)
    builder.process = process
    builder.http = SimpleNamespace(open=lambda *argv, **kwargs: pytest.fail("no CUA request"))
    with pytest.raises(RuntimeError, match="live disposable builder"):
        builder.cua_command("true", timeout=1)


def test_failed_cua_bootstrap_prevents_ssh(args, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: calls.append(argv))
    process = FakeProcess()
    monkeypatch.setattr(build.subprocess, "Popen", lambda *argv, **kwargs: process)
    builder = build.Builder(args, tmp_path)
    monkeypatch.setattr(builder, "cua_command", lambda *argv, **kwargs: "ALE_BUILDER_READY")

    def rejected(**kwargs):
        raise ValueError("guest public key installation failed")

    monkeypatch.setattr(builder, "bootstrap_ssh", rejected)
    with pytest.raises(ValueError, match="installation failed"):
        builder.start()
    builder.close()
    assert process.terminated
    assert not any(command[0] == "ssh" for command in calls)


def test_guest_provision_creates_only_public_authorization(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    host_public = tmp_path / "host.pub"
    host_public.write_text(PUBLIC_KEY)
    monkeypatch.setattr(bootstrap, "HOST_PUBLIC_KEY", host_public)
    monkeypatch.setattr(
        bootstrap.pwd,
        "getpwnam",
        lambda user: SimpleNamespace(
            pw_dir=str(home),
            pw_uid=1000,
            pw_gid=1000,
        ),
    )
    ownership = []
    monkeypatch.setattr(bootstrap.os, "chown", lambda *argv: ownership.append(argv))
    commands = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *argv, **kwargs: commands.append(argv))
    assert not (home / ".ssh").exists()
    assert bootstrap.provision("user", PUBLIC_KEY) == PUBLIC_KEY
    authorized = home / ".ssh/authorized_keys"
    assert authorized.read_text() == "restrict " + PUBLIC_KEY + "\n"
    assert authorized.stat().st_mode & 0o777 == 0o600
    assert authorized.parent.stat().st_mode & 0o777 == 0o700
    assert list(authorized.parent.iterdir()) == [authorized]
    assert ownership[-1] == (authorized, 1000, 1000)
    assert commands[-1] == (["/usr/bin/systemctl", "start", "ssh"],)


@pytest.mark.parametrize("key", ["PRIVATE KEY", "ssh-ed25519 invalid!", "ssh-ed25519 YQ=="])
def test_guest_bootstrap_rejects_nonpublic_or_malformed_keys(key):
    with pytest.raises(ValueError, match="SSH public key"):
        bootstrap.public_key(key)


def test_guest_bootstrap_refuses_symlinked_ssh_directory(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (home / ".ssh").symlink_to(elsewhere)
    monkeypatch.setattr(bootstrap.pwd, "getpwnam", lambda user: SimpleNamespace(pw_dir=str(home)))
    with pytest.raises(ValueError, match="symlinked"):
        bootstrap.provision("user", PUBLIC_KEY)
    assert not list(elsewhere.iterdir())


def test_stopped_containers_do_not_reserve_guest_ram(tmp_path, monkeypatch):
    (tmp_path / "meminfo").write_text("MemAvailable: 90000000 kB\nShmem: 1234 kB\n")
    (tmp_path / "stopped-containers.json").write_text(
        json.dumps([{"State": "exited", "Memory": 16 * build.GIB} for _ in range(90)])
    )
    zombie = tmp_path / "42"
    zombie.mkdir()
    (zombie / "comm").write_text("qemu-system-x86\n")
    (zombie / "status").write_text("State: Z (zombie)\n")
    monkeypatch.setattr(build, "capture", lambda *argv: pytest.fail("no Docker metadata budget"))
    memory = build.host_memory(tmp_path)
    assert memory["vms"] == []
    assert memory["unresident_vm_reservation"] == 0
    monkeypatch.setattr(build, "host_memory", lambda: memory)
    monkeypatch.setattr(
        build,
        "persistent_mount",
        lambda *argv, **kwargs: {
            "available": 500 * build.GIB,
            "device": 1,
            "path": "disk",
        },
    )
    assert build.capacity(
        Path("work"),
        Path("docker"),
        work_bytes=build.GIB,
        docker_bytes=build.GIB,
        memory_gib=10,
        headroom_gib=16,
    )


@pytest.mark.parametrize("state", ["R (running)", "S (sleeping)", "T (stopped)"])
def test_live_running_or_paused_guests_reserve_peak_minus_rss(tmp_path, state):
    (tmp_path / "meminfo").write_text("MemAvailable: 90000000 kB\nShmem: 1234 kB\n")
    process = tmp_path / "42"
    process.mkdir()
    (process / "comm").write_text("qemu-system-x86\n")
    (process / "cmdline").write_bytes(b"qemu-system-x86_64\0-m\0size=8G\0")
    (process / "status").write_text(f"State: {state}\nVmRSS: 1048576 kB\n")
    assert build.host_memory(tmp_path)["unresident_vm_reservation"] == 7 * build.GIB


def test_export_preserves_grok_install_and_openclaw_plugin_dependencies(root, contract):
    retained = [
        "home/user/.grok/install.json",
        "home/user/.grok/bin/grok",
        "home/user/.openclaw/extensions/plugin/node_modules/dependency/index.js",
    ]
    for name in [*retained, "home/user/.ssh/authorized_keys"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PUBLIC_KEY if name.endswith("authorized_keys") else "runtime")
    with tarfile.open(fileobj=make_archive(root, contract)) as archive:
        names = set(archive.getnames())
    assert all("./" + name in names for name in retained)
    assert not any("/.ssh" in name for name in names)


@pytest.mark.parametrize("state", ["R (running)", "S (sleeping)", "T (stopped)"])
def test_runner_renamed_qemu_is_counted(tmp_path, state):
    (tmp_path / "meminfo").write_text("MemAvailable: 90000000 kB\nShmem: 1234 kB\n")
    process = tmp_path / "42"
    process.mkdir()
    (process / "comm").write_text("windows\n")
    (process / "cmdline").write_bytes(
        b"\0".join(
            [
                b"/usr/bin/qemu-system-x86_64",
                b"-m",
                b"16G",
                b"-name",
                b"windows,process=windows",
                b"",
            ]
        )
    )
    (process / "status").write_text(f"State: {state}\nVmRSS: 1048576 kB\n")
    assert build.host_memory(tmp_path)["unresident_vm_reservation"] == 15 * build.GIB


def test_non_qemu_process_is_not_budgeted(tmp_path):
    (tmp_path / "meminfo").write_text("MemAvailable: 90000000 kB\nShmem: 1234 kB\n")
    process = tmp_path / "42"
    process.mkdir()
    (process / "comm").write_text("python\n")
    (process / "cmdline").write_bytes(b"python\0-m\0pytest\0")
    (process / "status").write_text("State: S (sleeping)\nVmRSS: 1048576 kB\n")
    assert build.host_memory(tmp_path)["vms"] == []


def test_readonly_audit_matches_export_and_never_normalizes_owners(root, contract, monkeypatch):
    monkeypatch.setattr(policy.os, "chown", lambda *args, **kwargs: pytest.fail("read-only"))
    result = audit.inspect_tree(root, contract)
    manifest = io.BytesIO()
    assert result["retained"] == policy.export_manifest(root, contract, manifest)
    assert result["complete"] and result["export_ready"]
    assert result["logical_bytes_upper_bound"] > 0
    assert result["manifest_sha256"]


def test_readonly_audit_does_not_chown_large_owners(root, contract, monkeypatch):
    original = Path.lstat

    def oversized(path):
        metadata = original(path)
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_uid=100000,
            st_gid=100000,
            st_size=metadata.st_size,
            st_blocks=metadata.st_blocks,
        )

    monkeypatch.setattr(Path, "lstat", oversized)
    monkeypatch.setattr(policy.os, "chown", lambda *args, **kwargs: pytest.fail("read-only"))
    result = audit.inspect_tree(root, contract)
    assert result["export_ready"]
    assert result["oversized_owners"] == result["retained"]


@pytest.mark.parametrize("bounds", [{"max_entries": 1}, {"max_seconds": 0.000001}])
def test_readonly_audit_limit_is_not_success(root, contract, bounds):
    result = audit.inspect_tree(root, contract, **bounds)
    assert not result["complete"] and not result["export_ready"]
    assert result["limit_reached"]
    assert "manifest_sha256" not in result


def test_readonly_audit_bounds_findings_and_detects_data_links(root, contract):
    (root / "opt/runtime").unlink()
    for index in range(5):
        (root / f"opt/tool-{index}").symlink_to("/media/user/data/opt/tool")
    result = audit.inspect_tree(root, contract, max_findings=2)
    assert result["complete"] and not result["export_ready"]
    assert result["findings_count"] == 6
    assert len(result["findings"]) == 2


def test_readonly_audit_prunes_data_and_preserves_runtime_cache(root, contract):
    data = root / "media/user/data/task/reference"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"not-exported" * 100)
    target = root / "root/.cache/uv/python/bin/python"
    target.parent.mkdir(parents=True)
    target.write_text("runtime")
    contract["preserve_cache_paths"] = ["/root/.cache/uv/python"]
    result = audit.inspect_tree(root, contract)
    assert result["export_ready"]
    assert result["excluded_roots_or_files"] > 0
    assert "/media/user" not in result["groups"]
    assert result["groups"]["/root/.cache"] == len("runtime")


def test_readonly_audit_unreadable_tree_is_not_success(root, contract, monkeypatch):
    def unreadable(*args, **kwargs):
        kwargs["onerror"](PermissionError(13, "denied", "/opt/runtime"))
        return iter(())

    monkeypatch.setattr(audit.os, "walk", unreadable)
    result = audit.inspect_tree(root, contract)
    assert not result["export_ready"]
    assert result["findings_count"] == 1


@pytest.mark.parametrize("bounds", [{"max_entries": 0}, {"max_seconds": 0}, {"max_findings": 0}])
def test_readonly_audit_rejects_invalid_bounds(root, contract, bounds):
    with pytest.raises(ValueError, match="positive"):
        audit.inspect_tree(root, contract, **bounds)


@pytest.mark.parametrize(
    "prefix", ["/media/user/data/opt/bwa-mem2-2.2.1", "/media/user/data/toolchains/r-biostatistics"]
)
def test_audited_data_runtime_allowlist_preserves_links_but_not_task_data(root, contract, prefix):
    contract["preserve_runtime_paths"] = [prefix]
    contract["required_paths"].append(prefix)
    target = root / prefix.lstrip("/") / "bin/tool"
    target.parent.mkdir(parents=True)
    target.write_text("runtime")
    (target.parent / "alias").symlink_to("tool")
    link = root / "opt/runtime"
    link.unlink()
    link.symlink_to(prefix)
    forbidden = [
        prefix + "/.ssh/id_ed25519",
        prefix + "/.env",
        prefix + "/task-data-secret/reference.7z",
        "/media/user/data/agenthle/input/read.fastq",
        "/media/user/data/agenthle/reference.7z",
        str(Path(prefix).parent) + "/unreviewed/asset",
    ]
    for path in forbidden:
        item = root / path.lstrip("/")
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("must not enter any layer")
    with tarfile.open(fileobj=make_archive(root, contract)) as archive:
        names = archive.getnames()
    assert "." + prefix + "/bin/tool" in names
    assert "." + prefix + "/bin/alias" in names
    assert all("." + path not in names for path in forbidden)
    assert policy.inspect_archive(make_archive(root, contract), contract)["members"] > 0
    assert audit.inspect_tree(root, contract)["export_ready"]


@pytest.mark.parametrize(
    "path",
    [
        "/media/user/data",
        "/media/user/data/opt",
        "/media/user/data/toolchains",
        "/media/user/data/agenthle",
        "/media/user/data/opt/*",
        "/root/.ssh",
        "/media/user/data/opt/tool/../other",
    ],
)
def test_data_runtime_allowlist_rejects_broad_or_unsafe_roots(contract, path):
    contract["preserve_runtime_paths"] = [path]
    contract["required_paths"].append(path)
    with pytest.raises(ValueError):
        policy.validate_contract(contract)


def test_data_runtime_allowlist_cannot_silently_retain_missing_tool(root, contract):
    path = "/media/user/data/opt/missing-tool"
    contract["preserve_runtime_paths"] = [path]
    with pytest.raises(ValueError, match="required path"):
        policy.validate_contract(contract)
    contract["required_paths"].append(path)
    with pytest.raises(ValueError, match="missing runtime"):
        policy.export_manifest(root, contract, io.BytesIO())
    assert not audit.inspect_tree(root, contract)["export_ready"]


def test_preserved_data_runtime_cannot_link_to_task_data(root, contract):
    path = "/media/user/data/opt/tool"
    contract["preserve_runtime_paths"] = [path]
    contract["required_paths"].append(path)
    target = root / path.lstrip("/")
    target.parent.mkdir(parents=True)
    target.symlink_to("/media/user/data/agenthle/task")
    with pytest.raises(ValueError, match="excluded"):
        policy.export_manifest(root, contract, io.BytesIO())


def test_cleanup_does_not_recursively_change_preserved_data_runtime_ownership():
    script = (SCRIPTS / "cleanup.sh").read_text()
    assert "chown -R user:user /media/user/data" not in script
    assert "chown user:user /media/user/data /media/user/data/agenthle" in script


def test_explicit_exclusions_override_data_runtime_preservation(contract):
    path = "/media/user/data/toolchains/r-biostatistics"
    contract["preserve_runtime_paths"] = [path]
    contract["required_paths"].append(path)
    contract["exclude_paths"] = [path + "/unreviewed"]
    policy.validate_contract(contract)
    assert not policy.forbidden(path + "/bin/R", contract)
    assert policy.forbidden(path + "/unreviewed/asset", contract)


@pytest.mark.parametrize(
    "field", ["preserve_runtime_paths", "preserve_cache_paths", "exclude_paths"]
)
def test_contract_path_lists_reject_strings(contract, field):
    contract[field] = "/media/user/data/opt/tool"
    with pytest.raises(ValueError, match="path list"):
        policy.validate_contract(contract)


def test_audited_baseline_stale_link_exclusion_preserves_system_tool(root, contract):
    stale = root / "opt/fastqc-0.12.1"
    stale.symlink_to("/media/user/data/opt/fastqc-0.12.1")
    tool = root / "usr/bin/fastqc"
    tool.parent.mkdir(parents=True)
    tool.write_text("installed system tool")
    contract["required_paths"].append("/usr/bin/fastqc")
    with pytest.raises(ValueError, match="symlink target"):
        policy.export_manifest(root, contract, io.BytesIO())
    contract["exclude_paths"] = ["/opt/fastqc-0.12.1"]
    with tarfile.open(fileobj=make_archive(root, contract)) as archive:
        names = archive.getnames()
    assert "./usr/bin/fastqc" in names
    assert "./opt/fastqc-0.12.1" not in names
    assert stale.is_symlink()


@pytest.mark.parametrize("missing_active", [False, True])
def test_active_plugin_contract_does_not_require_obsolete_mirror(root, contract, missing_active):
    active = "/home/user/.openclaw/plugin-runtime-deps/openclaw-2026.4.26"
    obsolete = "/home/user/.openclaw/plugin-runtime-deps/openclaw-2026.4.24"
    package = "/usr/local/lib/node_modules/openclaw"
    for prefix in [package, active + "/node_modules", active + "/dist/extensions", obsolete]:
        (root / prefix.lstrip("/")).mkdir(parents=True)
    target = root / package.lstrip("/") / "active.js"
    if not missing_active:
        target.write_text("installed active dependency")
    (root / (active + "/node_modules/dependency.js").lstrip("/")).symlink_to(package + "/active.js")
    (root / (obsolete + "/old.js").lstrip("/")).symlink_to(package + "/absent-old.js")
    contract["required_paths"].extend(
        [package, active + "/node_modules", active + "/dist/extensions"]
    )
    contract["exclude_paths"] = [obsolete]
    if missing_active:
        with pytest.raises(ValueError, match="symlink target"):
            policy.export_manifest(root, contract, io.BytesIO())
    else:
        archive = make_archive(root, contract)
        assert policy.inspect_archive(archive, contract)["members"] > 0
        archive.seek(0)
        with tarfile.open(fileobj=archive) as contents:
            names = contents.getnames()
        assert "." + active + "/node_modules/dependency.js" in names
        assert not any(name.startswith("." + obsolete) for name in names)


@pytest.fixture
def provider_harness(args, tmp_path, monkeypatch):
    from ale_run.base_interface import SandboxHandle
    from ale_run.environments.providers import qemu

    calls = []
    sandbox = SandboxHandle(
        id="ale-qemu-docker-export-unit",
        endpoint="http://127.0.0.1:12345",
        os="linux",
        work_dir_base="/home/user/.ale",
        task_data_root="/media/user/data/agenthle",
        node="/usr/local/bin/node",
        python="/opt/ale-run/.venv/bin/python",
        mcp_server_dir="/home/user/cua_mcp_server",
        metadata={
            "provider": "qemu",
            "container_name": "ale-qemu-docker-export-unit",
            "slot_root": str(tmp_path / "runtime/slots/owned"),
            "base_qcow2": str(args.source_qcow2),
        },
    )

    class Provider:
        def __init__(self, config):
            parsed = qemu._build_provider_config(config)
            assert parsed.snapshots["docker-export"].disk_source == str(args.source_qcow2)
            calls.append(("config", config))

        async def acquire(self, spec):
            calls.append(("acquire", spec))
            return sandbox

        async def release(self, handle, *, mode):
            calls.append(("release", handle.id, mode))

    monkeypatch.setattr(qemu, "QemuProvider", Provider)
    monkeypatch.setattr(build, "persistent_mount", lambda *argv: calls.append(("mount", *argv)))
    monkeypatch.setattr(build, "run", lambda *argv, **kwargs: calls.append(argv))
    monkeypatch.setattr(
        build, "capture", lambda *argv: PUBLIC_KEY if argv[0] == "ssh-keygen" else "true"
    )
    monkeypatch.setattr(build.subprocess, "run", lambda *argv, **kwargs: None)
    builder = build.ProviderBuilder(args, tmp_path)
    builder.http = SimpleNamespace(open=lambda *argv, **kwargs: CuaResponse(b'{"status":"ok"}'))
    monkeypatch.setattr(builder, "cua_request", lambda *argv, **kwargs: PUBLIC_KEY)
    return builder, calls, sandbox


def test_provider_builder_uses_owned_local_clone_and_pinned_proxy_ssh(
    provider_harness, args, tmp_path
):
    builder, calls, sandbox = provider_harness
    builder.start()
    config = next(call[1] for call in calls if call[0] == "config")
    snapshot = config["snapshots"]["docker-export"]
    assert snapshot["disk_source"] == str(args.source_qcow2)
    assert snapshot["runtime_root"] == str(tmp_path / "runtime")
    assert snapshot["runner_image"] == args.runner_image_id
    assert snapshot["runner_pull_policy"] == "never"
    assert snapshot["bind_address"] == "127.0.0.1"
    assert not config.get("gcs_sa_key")
    assert "-p" not in builder.ssh
    assert f"ProxyCommand=docker exec -i {sandbox.id} nc 172.30.0.2 22" in builder.ssh
    assert "StrictHostKeyChecking=yes" in builder.ssh
    assert f"HostKeyAlias={sandbox.id}" in builder.ssh
    assert (tmp_path / "known_hosts").read_text() == f"{sandbox.id} {PUBLIC_KEY}\n"
    assert (tmp_path / "sandbox.json").is_file()
    builder.close()
    builder.close()
    assert [call for call in calls if call[0] == "release"] == [("release", sandbox.id, "stop")]


def test_provider_rejects_foreign_handle_without_touching_it(provider_harness):
    builder, calls, sandbox = provider_harness
    sandbox.metadata["slot_root"] = "/parent/retained-builder"
    with pytest.raises(ValueError, match="outside this owned build"):
        builder.start()
    builder.close()
    assert not any(call[0] in {"ssh", "ssh-keygen", "release"} for call in calls)


def test_provider_invalid_health_stops_only_owned_clone(provider_harness):
    builder, calls, sandbox = provider_harness
    builder.http = SimpleNamespace(open=lambda *argv, **kwargs: CuaResponse(b"{}"))
    with pytest.raises(ValueError, match="health"):
        builder.start()
    builder.close()
    assert ("release", sandbox.id, "stop") in calls
    assert not any(call[0] == "ssh-keygen" for call in calls)


def test_stopped_provider_cannot_inject_key(provider_harness):
    builder, _, _ = provider_harness
    with pytest.raises(RuntimeError, match="live disposable"):
        builder.cua_command("true", timeout=1)
    builder.start()
    builder.close()
    with pytest.raises(RuntimeError, match="live disposable"):
        builder.cua_command("true", timeout=1)


def test_provider_bootstrap_failure_keeps_cleanup_available(provider_harness, monkeypatch):
    builder, calls, sandbox = provider_harness

    def failed(**kwargs):
        raise ValueError("bootstrap rejected")

    monkeypatch.setattr(builder, "bootstrap_ssh", failed)
    with pytest.raises(ValueError, match="bootstrap rejected"):
        builder.start()
    builder.close()
    assert ("release", sandbox.id, "stop") in calls


def test_cli_defaults_to_provider_and_supports_separate_stages(args):
    parsed = build.parse_args(cli_args(args) + ["--export-only"])
    assert parsed.builder_backend == "provider"
    assert parsed.stage == "export"
    assert parsed.docker_budget_gib is None
    assert build.parse_args(cli_args(args) + ["--finalize-only"]).stage == "finalize"
    with pytest.raises(SystemExit):
        build.parse_args(cli_args(args) + ["--export-only", "--finalize-only"])


@pytest.fixture
def phased_build(args, contract, monkeypatch):
    calls = []
    budgets = []
    monkeypatch.setattr(build.shutil, "which", lambda *argv: "/existing/tool")
    monkeypatch.setattr(build.os, "access", lambda *argv: True)
    monkeypatch.setattr(build, "persistent_mount", lambda *argv, **kwargs: {})
    monkeypatch.setattr(
        build,
        "source_identity",
        lambda *argv: ({"source_sha256": args.source_sha256}, contract, 1024),
    )
    monkeypatch.setattr(build, "docker_storage", lambda: Path("/persistent/docker"))
    monkeypatch.setattr(
        build,
        "capture",
        lambda *argv: "sha256:" + "a" * 64 if argv[:3] == ("docker", "image", "inspect") else "",
    )

    def capacity(*argv, **kwargs):
        budgets.append(kwargs.copy())
        return {"checked": True}

    class Builder:
        def __init__(self, args, directory):
            self.directory = directory

        def start(self):
            calls.append("start")

        def export(self, destination):
            calls.append("export")
            destination.write_bytes(b"small archive fixture")
            (self.directory / "packages.before.txt").write_text("python\t1\n")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(build, "capacity", capacity)
    monkeypatch.setattr(build, "Builder", Builder)
    monkeypatch.setattr(build, "ProviderBuilder", Builder)
    monkeypatch.setattr(build, "audit_export", lambda *argv: {"logical_bytes": 80 * build.GIB})
    monkeypatch.setattr(build, "finalize", lambda *argv: calls.append("finalize") or "sha256:unit")
    return calls, budgets


def test_export_then_finalize_reserves_only_current_phase(args, phased_build):
    calls, budgets = phased_build
    args.stage = "export"
    args.docker_budget_gib = None
    build.build(args)
    assert "export" in calls and "finalize" not in calls
    assert all(item["docker_bytes"] == 0 for item in budgets)
    measurement = next(args.workdir.glob("*/export-measurement.json"))
    report = json.loads(measurement.read_text())
    assert report["compressed_bytes"] == len(b"small archive fixture")
    assert report["suggested_docker_budget_gib"] == 180
    calls.clear()
    budgets.clear()
    args.stage = "finalize"
    build.build(args)
    assert calls == ["finalize"]
    assert all(item["work_bytes"] == 0 for item in budgets)
    assert budgets[-1]["docker_bytes"] == 180 * build.GIB


def test_finalize_preflight_audits_archive_without_allocating(args, phased_build):
    calls, budgets = phased_build
    args.stage = "export"
    build.build(args)
    calls.clear()
    budgets.clear()
    args.stage = "finalize"
    args.preflight_only = True
    build.build(args)
    assert calls == []
    assert budgets[-1]["work_bytes"] == 0
    assert budgets[-1]["docker_bytes"] == args.docker_budget_gib * build.GIB


def test_finalize_missing_cache_cannot_start_builder(args, phased_build):
    calls, _ = phased_build
    args.stage = "finalize"
    with pytest.raises(ValueError, match="verified source-bound export"):
        build.build(args)
    assert calls == []


def test_export_preflight_does_not_need_native_kvm_rights(args, phased_build, monkeypatch):
    calls, budgets = phased_build
    args.builder_backend = "provider"
    args.stage = "export"
    args.preflight_only = True
    monkeypatch.setattr(build.os, "access", lambda *argv: pytest.fail("no native KVM access test"))
    build.build(args)
    assert calls == []
    assert budgets[0]["docker_bytes"] == build.GIB
    assert args.runner_image_id == "sha256:" + "a" * 64


def test_finalize_capacity_failure_leaves_export_intact(args, phased_build, monkeypatch):
    calls, _ = phased_build
    args.stage = "export"
    build.build(args)
    archive = args.workdir / "rootfs.tar.zst"
    expected = archive.read_bytes()
    calls.clear()
    args.stage = "finalize"

    def insufficient(*argv, **kwargs):
        if kwargs["docker_bytes"]:
            raise ValueError("insufficient persistent capacity")
        return {}

    monkeypatch.setattr(build, "capacity", insufficient)
    with pytest.raises(ValueError, match="capacity"):
        build.build(args)
    assert calls == []
    assert archive.read_bytes() == expected


def test_provider_unresolved_runner_id_fails_before_allocation(args, phased_build, monkeypatch):
    calls, budgets = phased_build
    args.builder_backend = "provider"
    monkeypatch.setattr(build, "capture", lambda *argv: "not-an-image-id")
    with pytest.raises(ValueError, match="immutable ID"):
        build.build(args)
    assert calls == [] and budgets == []


def test_finalize_rejects_tampered_export_before_any_builder(args, phased_build):
    calls, _ = phased_build
    args.stage = "export"
    build.build(args)
    calls.clear()
    (args.workdir / "rootfs.tar.zst").write_bytes(b"changed")
    args.stage = "finalize"
    with pytest.raises(ValueError, match="integrity"):
        build.build(args)
    assert calls == []


@pytest.mark.parametrize("invalid", ["command", None, [["relative"]], [["/bin/true", 1]]])
def test_source_probe_contract_rejects_invalid_commands(contract, invalid):
    contract["source_probes"] = invalid
    with pytest.raises(ValueError):
        policy.validate_contract(contract)


@pytest.mark.parametrize("failure", [False, True])
def test_source_probe_runs_before_runtime_and_archive(
    args, contract, tmp_path, monkeypatch, failure
):
    contract["source_probes"] = [["/usr/bin/python3", "-c", "print('source link check')"]]
    args.runtime_manifest.write_text(json.dumps(contract))
    builder = build.ProviderBuilder(args, tmp_path)
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if failure and argv == contract["source_probes"][0]:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(builder, "command", command)
    destination = tmp_path / "archive.partial"
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            builder.export(destination)
        assert not destination.exists()
        assert contract["probes"][0] not in calls
    else:
        builder.export(destination)
        assert calls.index(contract["source_probes"][0]) < calls.index(contract["probes"][0])
        assert calls[-1][0:2] == ["sudo", "env"]
    assert "source link check" in (tmp_path / "runtime.before.log").read_text()


@pytest.mark.parametrize("missing", [False, True])
def test_uv_minor_alias_resolves_to_retained_patch_interpreter(root, contract, missing):
    parent = root / "home/user/.local/share/uv/python"
    parent.mkdir(parents=True)
    concrete = parent / "cpython-3.14.3-linux-x86_64-gnu"
    (concrete / "bin").mkdir(parents=True)
    if not missing:
        (concrete / "bin/python3.14").write_text("interpreter")
    alias = parent / "cpython-3.14-linux-x86_64-gnu"
    alias.symlink_to(concrete.name)
    retained = "/home/user/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14"
    contract["required_paths"].append(retained)
    if missing:
        with pytest.raises(ValueError, match="missing runtime"):
            policy.export_manifest(root, contract, io.BytesIO())
    else:
        contents = make_archive(root, contract)
        assert policy.inspect_archive(contents, contract)["members"] > 0
