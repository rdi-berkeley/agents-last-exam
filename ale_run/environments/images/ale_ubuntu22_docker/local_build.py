"""Build a data-less Docker candidate from an explicitly sealed local QCOW2."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from bootstrap_ssh import public_key
import rootfs_policy

HERE = Path(__file__).resolve().parent
PERSISTENT_ROOT = Path(
    os.environ.get("ALE_IMAGE_BUILD_ROOT", str(Path.home() / "ale-overall"))
).expanduser().resolve()
DISK_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs"}
GIB = 1024**3


def run(*argv, **kwargs):
    return subprocess.run(
        [str(arg) for arg in argv],
        check=True,
        timeout=kwargs.pop("timeout", 120),
        **kwargs,
    )


def capture(*argv) -> str:
    return run(*argv, stdout=subprocess.PIPE, text=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def persistent_mount(path: Path, *, workspace: bool = True) -> dict:
    resolved = path.expanduser().resolve()
    if workspace and not resolved.is_relative_to(PERSISTENT_ROOT):
        raise ValueError(f"build storage must be under {PERSISTENT_ROOT}: {resolved}")
    existing = resolved
    while not existing.exists():
        existing = existing.parent
    mount = json.loads(
        capture(
            "findmnt", "--json", "--target", existing, "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"
        )
    )["filesystems"][0]
    if mount["fstype"] not in DISK_FILESYSTEMS or "ro" in mount["options"].split(","):
        raise ValueError(f"not verified writable persistent storage: {resolved}")
    usage = capture("df", "-B1", "--output=avail", existing).splitlines()
    mount.update(path=str(resolved), available=int(usage[-1]), device=existing.stat().st_dev)
    return mount


def host_memory(proc_root: Path = Path("/proc")) -> dict:
    """Budget live QEMU processes, including paused guests, never stopped containers."""
    memory = {}
    for line in (proc_root / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    vms = []
    reserved = 0
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / "comm").read_text().strip()
            status = (directory / "status").read_text()
            state = re.search(r"^State:\s+(\w)", status, re.MULTILINE)
            if state and state[1] in {"Z", "X", "x"}:
                continue
            argv = (directory / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
            executable = Path(argv[0]).name
            if not (
                command.startswith("qemu-system")
                or command == "kvm"
                or executable.startswith("qemu-system")
                or executable == "kvm"
            ):
                continue
            raw = argv[argv.index("-m") + 1]
            raw = raw.split(",")[0].removeprefix("size=")
            match = re.fullmatch(r"(\d+)([MG]?)", raw, re.IGNORECASE)
            if not match:
                raise ValueError("cannot budget an existing QEMU -m argument")
            peak = int(match[1]) * (GIB if match[2].upper() == "G" else 1024**2)
            rss = int(re.search(r"VmRSS:\s+(\d+)", status)[1]) * 1024
        except FileNotFoundError:
            continue
        except (PermissionError, ValueError, IndexError, TypeError) as error:
            raise ValueError(f"cannot inspect running VM {directory.name}") from error
        reserved += max(peak - rss, 0)
        vms.append({"pid": int(directory.name), "peak_bytes": peak, "rss_bytes": rss})
    return {
        "available": memory["MemAvailable"],
        "shared": memory["Shmem"],
        "unresident_vm_reservation": reserved,
        "vms": vms,
    }


def capacity(
    workdir: Path,
    docker_root: Path,
    *,
    work_bytes: int,
    docker_bytes: int,
    memory_gib: int,
    headroom_gib: int,
) -> dict:
    work = persistent_mount(workdir)
    docker = persistent_mount(docker_root, workspace=False)
    for mount, needed in ((work, work_bytes), (docker, docker_bytes)):
        if work["device"] == docker["device"]:
            needed = work_bytes + docker_bytes
        if mount["available"] < needed + 10 * GIB:
            raise ValueError(f"insufficient persistent capacity: {mount['path']}")
    memory = host_memory()
    needed = (memory_gib + headroom_gib) * GIB + memory["unresident_vm_reservation"]
    if memory["available"] < needed:
        raise ValueError("insufficient host RAM including retained VM peaks and headroom")
    return {"work": work, "docker": docker, "memory": memory}


def docker_storage() -> Path:
    context = json.loads(capture("docker", "context", "inspect"))[0]
    endpoint = os.environ.get("DOCKER_HOST") or context["Endpoints"]["docker"]["Host"]
    if not endpoint.startswith("unix://"):
        raise ValueError("Docker must use a local unix socket for capacity verification")
    info = json.loads(capture("docker", "info", "--format", "{{json .}}"))
    if info.get("OSType") != "linux":
        raise ValueError("Linux Docker daemon required")
    return Path(info["DockerRootDir"]).resolve(strict=True)


def source_identity(args) -> tuple[dict, dict, int]:
    source = args.source_qcow2.resolve(strict=True)
    persistent_mount(source, workspace=False)
    info = json.loads(capture("qemu-img", "info", "--output=json", "--backing-chain", source))
    if len(info) != 1 or info[0].get("format") != "qcow2" or info[0].get("backing-filename"):
        raise ValueError("source must be a standalone final QCOW2 without a backing chain")
    if info[0].get("format-specific", {}).get("data", {}).get("data-file"):
        raise ValueError("source QCOW2 must not depend on an external data file")
    if info[0].get("snapshots") or info[0].get("dirty-flag") or info[0].get("encrypted"):
        raise ValueError("source has snapshots or an unclean QCOW2 state")
    actual = sha256(source)
    if actual != args.source_sha256:
        raise ValueError("source QCOW2 SHA256 mismatch")
    contract = json.loads(args.runtime_manifest.read_text())
    rootfs_policy.validate_contract(contract)
    data_hash = sha256(args.data_manifest)
    for key, value in {
        "release": args.release,
        "source_sha256": actual,
        "data_manifest_sha256": data_hash,
    }.items():
        if contract.get(key) != value:
            raise ValueError(f"runtime manifest {key} mismatch")
    scripts = {
        path.name: sha256(path) for path in sorted(HERE.iterdir()) if path.suffix in {".py", ".sh"}
    }
    return (
        {
            "schema": 1,
            "source": str(source),
            "source_sha256": actual,
            "release": args.release,
            "data_manifest_sha256": data_hash,
            "runtime_manifest_sha256": sha256(args.runtime_manifest),
            "scripts": scripts,
        },
        contract,
        int(info[0]["virtual-size"]),
    )


def cached_export(archive: Path, receipt: Path, identity: dict) -> bool:
    if not archive.exists() and not receipt.exists():
        return False
    if not archive.is_file() or not receipt.is_file():
        raise ValueError("incomplete export cache; use a new workdir or review the failed export")
    saved = json.loads(receipt.read_text())
    if saved.get("identity") != identity:
        raise ValueError("cached export source/release/script identity mismatch")
    if saved.get("size") != archive.stat().st_size or saved.get("sha256") != sha256(archive):
        raise ValueError("cached export integrity mismatch")
    inventory = Path(saved["builder_evidence"]) / "packages.before.txt"
    if not inventory.resolve().is_relative_to(archive.parent.resolve()) or not inventory.is_file():
        raise ValueError("cached source package inventory missing or outside workdir")
    if saved.get("package_inventory_sha256") != sha256(inventory):
        raise ValueError("cached source package inventory integrity mismatch")
    return True


class TarReader:
    def __init__(self, stream):
        self.stream = stream
        self.tail = b""
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.size += len(data)
        self.tail = (self.tail + data)[-1024:]
        return data


def audit_export(archive: Path, contract: dict) -> dict:
    run("zstd", "--test", archive, timeout=7200)
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        reader = TarReader(process.stdout)
        result = rootfs_policy.inspect_archive(reader, contract)
        while reader.read(1024 * 1024):
            pass
        if reader.size % 512 or reader.tail != b"\0" * 1024:
            raise ValueError("truncated tar export (missing end blocks)")
        if process.wait(timeout=30):
            raise ValueError("export decompression failed")
        return result
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()


class Builder:
    def __init__(self, args, directory: Path):
        self.args = args
        self.directory = directory
        self.process = None
        self.log = None
        with socket.socket() as listener, socket.socket() as cua_listener:
            listener.bind(("127.0.0.1", 0))
            cua_listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
            self.cua_port = cua_listener.getsockname()[1]
        self.ssh_key = args.ssh_key or directory / "id_ed25519"
        self.known_host = f"[127.0.0.1]:{self.port}"
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.ssh = [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            str(self.ssh_key.resolve()),
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={directory / 'known_hosts'}",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            f"{args.ssh_user}@127.0.0.1",
        ]

    def command(self, argv, **kwargs):
        return run(*self.ssh, shlex.join(argv), **kwargs)

    def cua_command(self, command: str, *, timeout: float) -> str:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("CUA bootstrap requires this live disposable builder")
        url = f"http://127.0.0.1:{self.cua_port}/cmd"
        return self.cua_request(url, command, timeout=timeout)

    def cua_request(self, url: str, command: str, *, timeout: float) -> str:
        request = urllib.request.Request(
            url,
            data=json.dumps({"command": "run_command", "params": {"command": command}}).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        deadline = time.monotonic() + timeout
        with self.http.open(request, timeout=timeout) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("invalid CUA HTTP response")
            received = 0
            while time.monotonic() < deadline:
                line = response.readline(65537)
                received += len(line)
                if not line or received > 65536:
                    break
                if not line.startswith(b"data:"):
                    continue
                data = json.loads(line[5:].strip().decode("utf-8-sig"))
                if not isinstance(data, dict) or data.get("success") is not True:
                    raise ValueError("CUA command did not succeed")
                result = data.get("return_code", data.get("returncode"))
                if (
                    type(result) is not int
                    or result != 0
                    or not isinstance(data.get("stdout"), str)
                ):
                    raise ValueError("CUA command returned an invalid or nonzero exit status")
                return data["stdout"]
        raise ValueError("missing, oversized or timed-out CUA command response")

    def bootstrap_ssh(self, *, timeout: float):
        persistent_mount(self.directory)
        if self.args.ssh_key is None:
            if self.ssh_key.exists() or self.ssh_key.is_symlink():
                raise ValueError("refusing to overwrite a builder SSH identity")
            run(
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"ale-docker-build-{self.directory.name}",
                "-f",
                self.ssh_key,
            )
        key = public_key(capture("ssh-keygen", "-y", "-P", "", "-f", self.ssh_key))
        script = (HERE / "bootstrap_ssh.py").read_text()
        host_key = public_key(
            self.cua_command(
                shlex.join(["sudo", "-n", "python3", "-c", script, self.args.ssh_user, key]),
                timeout=timeout,
            )
        )
        known_hosts = self.directory / "known_hosts"
        known_hosts.write_text(f"{self.known_host} {host_key}\n")
        known_hosts.chmod(0o600)
        write_json(
            self.directory / "ssh-bootstrap.json",
            {
                "public_key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "host_key_sha256": hashlib.sha256(host_key.encode()).hexdigest(),
                "transport": "disposable-builder-loopback-CUA",
            },
        )

    def start(self):
        overlay = self.directory / "builder.qcow2"
        run(
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            self.args.source_qcow2.resolve(),
            overlay,
        )
        self.log = (self.directory / "qemu.log").open("wb")
        block = {
            "driver": "qcow2",
            "node-name": "build-disk",
            "file": {"driver": "file", "filename": str(overlay)},
        }
        firmware = []
        if self.args.uefi_code:
            writable_vars = self.directory / "uefi-vars.fd"
            shutil.copyfile(self.args.uefi_vars, writable_vars)
            firmware = [
                "-drive",
                f"if=pflash,format=raw,readonly=on,file={self.args.uefi_code}",
                "-drive",
                f"if=pflash,format=raw,file={writable_vars}",
            ]
        self.process = subprocess.Popen(
            [
                "qemu-system-x86_64",
                "-enable-kvm",
                "-machine",
                "q35,smm=off" if self.args.uefi_code else "pc",
                "-cpu",
                "host",
                "-m",
                f"{self.args.memory_gib}G",
                "-smp",
                str(self.args.cpus),
                "-blockdev",
                json.dumps(block),
                "-device",
                "virtio-blk-pci,drive=build-disk",
                "-netdev",
                f"user,id=net0,restrict=on,hostfwd=tcp:127.0.0.1:{self.port}-:22,"
                f"hostfwd=tcp:127.0.0.1:{self.cua_port}-:5000",
                "-device",
                "virtio-net-pci,netdev=net0",
                "-display",
                "none",
                "-serial",
                "stdio",
                "-monitor",
                "none",
                "-no-reboot",
                *firmware,
            ],
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.args.boot_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("QEMU builder exited; inspect qemu.log")
            try:
                remaining = max(0.1, deadline - time.monotonic())
                if self.cua_command("printf ALE_BUILDER_READY", timeout=min(5, remaining)) != (
                    "ALE_BUILDER_READY"
                ):
                    raise ValueError("unexpected builder CUA readiness response")
                break
            except (OSError, ValueError):
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        else:
            raise TimeoutError("builder CUA did not become ready")
        self.bootstrap_ssh(timeout=min(60, max(0.1, deadline - time.monotonic())))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("QEMU builder exited; inspect qemu.log")
            try:
                self.command(
                    ["sudo", "-n", "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=min(15, max(0.1, deadline - time.monotonic())),
                )
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                time.sleep(2)
        raise TimeoutError("builder SSH/sudo did not become ready")

    def export(self, destination: Path):
        self.command(["sudo", "mkdir", "-p", "/tmp/ale-docker-build"])
        for path in [
            HERE / "export_rootfs.sh",
            HERE / "rootfs_policy.py",
            self.args.runtime_manifest,
        ]:
            name = "runtime.json" if path == self.args.runtime_manifest else path.name
            with path.open("rb") as source:
                self.command(
                    ["sudo", "tee", f"/tmp/ale-docker-build/{name}"],
                    stdin=source,
                    stdout=subprocess.DEVNULL,
                )
        for name, argv in (
            ("mounts.json", ["findmnt", "--json"]),
            ("packages.before.txt", ["dpkg-query", "-W", "-f=${Package}\t${Version}\n"]),
        ):
            with (self.directory / name).open("wb") as output:
                self.command(argv, stdout=output)
        with (self.directory / "runtime.before.log").open("wb") as output:
            contract = json.loads(self.args.runtime_manifest.read_text())
            for argv in contract.get("source_probes", []) + contract["probes"]:
                output.write(("PROBE " + json.dumps(argv) + "\n").encode())
                output.flush()
                self.command(argv, stdout=output, stderr=subprocess.STDOUT, timeout=600)
        with destination.open("xb") as output, (self.directory / "export.log").open("wb") as log:
            self.command(
                [
                    "sudo",
                    "env",
                    "ALE_RUNTIME_MANIFEST=/tmp/ale-docker-build/runtime.json",
                    "bash",
                    "/tmp/ale-docker-build/export_rootfs.sh",
                ],
                stdout=output,
                stderr=log,
                timeout=21600,
            )

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self.log is not None:
            self.log.close()


class ProviderBuilder(Builder):
    def __init__(self, args, directory: Path):
        self.args = args
        self.directory = directory
        self.sandbox = None
        self.provider = None
        self.ssh_key = args.ssh_key or directory / "id_ed25519"
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.runtime_root = directory / "runtime"
        self.stopped = False
        self.owned = False

    def start(self):
        sys.path.insert(0, str(HERE.parents[3]))
        from ale_run.base_interface import SandboxSpec
        from ale_run.environments.providers.qemu import QemuProvider

        persistent_mount(self.runtime_root)
        self.provider = QemuProvider(
            {
                "snapshots": {
                    "docker-export": {
                        "image": "ale-ubuntu22",
                        "disk_source": str(self.args.source_qcow2),
                        "runtime_root": str(self.runtime_root),
                        "image_cache_dir": str(self.directory / "unused-image-cache"),
                        "runner_image": self.args.runner_image_id,
                        "runner_pull_policy": "never",
                        "bind_address": "127.0.0.1",
                        "vcpus": self.args.cpus,
                        "memory_gb": self.args.memory_gib,
                        "ready_timeout_s": self.args.boot_timeout,
                    }
                }
            }
        )
        self.sandbox = asyncio.run(
            self.provider.acquire(
                SandboxSpec(
                    snapshot="docker-export", task_id=f"docker-export-{self.directory.name}"
                )
            )
        )
        metadata = self.sandbox.metadata
        if (
            metadata.get("provider") != "qemu"
            or metadata.get("container_name") != self.sandbox.id
            or not Path(metadata["slot_root"]).resolve().is_relative_to(self.runtime_root.resolve())
            or Path(metadata["base_qcow2"]).resolve() != self.args.source_qcow2.resolve()
            or not re.fullmatch(r"ale-qemu-[a-z0-9-]+", self.sandbox.id)
        ):
            raise ValueError("provider returned a sandbox outside this owned build")
        self.owned = True
        write_json(self.directory / "sandbox.json", asdict(self.sandbox))
        self.known_host = self.sandbox.id
        proxy = shlex.join(["docker", "exec", "-i", self.sandbox.id, "nc", "172.30.0.2", "22"])
        self.ssh = [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            str(self.ssh_key),
            "-o",
            f"ProxyCommand={proxy}",
            "-o",
            f"HostKeyAlias={self.known_host}",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.directory / 'known_hosts'}",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            f"{self.args.ssh_user}@172.30.0.2",
        ]
        with self.http.open(self.sandbox.endpoint + "/status", timeout=10) as response:
            if not healthy_status(response.status, response.read(65537)):
                raise ValueError("provider builder returned invalid health response")
        run("docker", "exec", self.sandbox.id, "sh", "-c", "command -v nc", timeout=15)
        self.bootstrap_ssh(timeout=60)
        self.command(["sudo", "-n", "true"], timeout=20, stdout=subprocess.DEVNULL)

    def cua_command(self, command: str, *, timeout: float) -> str:
        if self.sandbox is None or self.stopped or not self.owned:
            raise RuntimeError("CUA bootstrap requires this live disposable provider builder")
        if (
            capture("docker", "inspect", "--format", "{{.State.Running}}", self.sandbox.id)
            != "true"
        ):
            raise RuntimeError("owned provider builder is not running")
        return self.cua_request(self.sandbox.endpoint + "/cmd", command, timeout=timeout)

    def close(self):
        if self.sandbox is None or self.stopped or not self.owned:
            return
        try:
            with (self.directory / "runner.log").open("wb") as output:
                subprocess.run(
                    ["docker", "logs", "--tail", "1000", self.sandbox.id],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                    check=False,
                )
        finally:
            asyncio.run(self.provider.release(self.sandbox, mode="stop"))
            self.stopped = True


def healthy_status(status: int, body: bytes) -> bool:
    if status != 200 or len(body) > 65536:
        return False
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


def smoke(image: str, name: str, directory: Path, args) -> None:
    try:
        run(
            "docker",
            "run",
            "-d",
            "--pull=never",
            "--name",
            name,
            "--memory",
            f"{args.memory_gib}g",
            "--cpus",
            args.cpus,
            "--shm-size=2g",
            "-p",
            "127.0.0.1::5000",
            "-e",
            "ALE_ENABLE_DIND=0",
            "--entrypoint",
            "/dockerstartup/entrypoint.sh",
            image,
            "--wait",
        )
        port = int(
            capture(
                "docker",
                "inspect",
                "--format",
                '{{(index (index .NetworkSettings.Ports "5000/tcp") 0).HostPort}}',
                name,
            )
        )
        successes = 0
        for _ in range(40):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/status", timeout=4
                ) as response:
                    valid = healthy_status(response.status, response.read(65537))
            except (OSError, urllib.error.URLError):
                valid = False
            successes = successes + 1 if valid else 0
            if successes == 2:
                run(
                    "docker",
                    "exec",
                    name,
                    "env",
                    "DISPLAY=:0",
                    "XAUTHORITY=/home/user/.Xauthority",
                    "bash",
                    "-ec",
                    "test -S /tmp/.X11-unix/X0; pgrep -x xfwm4; pgrep -x xfce4-panel; "
                    "xdotool getdisplaygeometry",
                    timeout=30,
                )
                return
            time.sleep(3)
        raise RuntimeError("CUA did not return two valid HTTP 200 status=ok responses")
    finally:
        with (directory / "smoke.log").open("wb") as output:
            subprocess.run(
                ["docker", "logs", name],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        run("docker", "rm", "-f", name)


def compare_packages(before: Path, after: Path) -> dict:
    inventories = []
    for path in (before, after):
        packages = dict(line.split("\t", 1) for line in path.read_text().splitlines())
        if not packages:
            raise ValueError("empty package inventory")
        inventories.append(packages)
    source, container = inventories
    changed = [name for name, version in source.items() if container.get(name) != version]
    if changed:
        raise ValueError("source packages removed or upgraded: " + ", ".join(sorted(changed)))
    return {name: version for name, version in container.items() if name not in source}


def finalize(
    archive: Path, image: str, directory: Path, contract: dict, args, source_packages: Path
) -> str:
    base = "ale-docker-build:base-" + directory.name
    container = "ale-docker-finalize-" + directory.name
    producer = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        run("docker", "import", "-", base, stdin=producer.stdout, timeout=21600)
        producer.stdout.close()
        if producer.wait(timeout=30):
            raise RuntimeError("decompression failed during import")
    finally:
        producer.stdout.close()
        if producer.poll() is None:
            producer.kill()
        producer.wait()
    try:
        run(
            "docker",
            "run",
            "-d",
            "--pull=never",
            "--name",
            container,
            "--user",
            "0",
            "--memory",
            f"{args.memory_gib}g",
            "--cpus",
            args.cpus,
            base,
            "sleep",
            "infinity",
        )
        run("docker", "exec", container, "mkdir", "-p", "/dockerstartup")
        for script, destination in (
            ("entrypoint.sh", "/dockerstartup/entrypoint.sh"),
            ("cleanup.sh", "/root/cleanup.sh"),
        ):
            run("docker", "cp", HERE / script, f"{container}:{destination}")
        with (directory / "cleanup.log").open("wb") as output:
            run(
                "docker",
                "exec",
                container,
                "bash",
                "/root/cleanup.sh",
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        with (directory / "runtime.after.log").open("wb") as output:
            for argv in contract["probes"]:
                run(
                    "docker",
                    "exec",
                    "--user",
                    "user",
                    container,
                    *argv,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
        with (directory / "packages.after.txt").open("wb") as output:
            run(
                "docker",
                "exec",
                container,
                "dpkg-query",
                "-W",
                "-f=${Package}\t${Version}\n",
                stdout=output,
            )
        added = compare_packages(source_packages, directory / "packages.after.txt")
        write_json(directory / "packages.added.json", added)
        run(
            "docker",
            "commit",
            "--change",
            "USER user",
            "--change",
            "WORKDIR /home/user",
            "--change",
            "ENV HOME=/home/user",
            "--change",
            "ENV PATH=/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--change",
            'CMD ["/bin/bash"]',
            container,
            image,
            timeout=21600,
        )
    finally:
        run("docker", "rm", "-f", container)
    smoke(image, "ale-docker-smoke-" + directory.name, directory, args)
    return capture("docker", "image", "inspect", "--format", "{{.Id}}", image)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-qcow2", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--ssh-key", type=Path, help="optional unencrypted private key; otherwise generated"
    )
    parser.add_argument("--ssh-user", default="user")
    parser.add_argument("--uefi-code", type=Path)
    parser.add_argument("--uefi-vars", type=Path)
    parser.add_argument("--builder-backend", choices=("provider", "native"), default="provider")
    parser.add_argument("--runner-image", default="agentslastexam/ale-qemu:0.2.0")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--stage", choices=("all", "export", "finalize"), default="all")
    stages.add_argument("--export-only", dest="stage", action="store_const", const="export")
    stages.add_argument("--finalize-only", dest="stage", action="store_const", const="finalize")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=8)
    parser.add_argument("--headroom-gib", type=int, default=16)
    parser.add_argument("--boot-timeout", type=int, default=600)
    parser.add_argument("--overlay-budget-gib", type=int, default=16)
    parser.add_argument("--export-budget-gib", type=int, default=120)
    parser.add_argument("--docker-budget-gib", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    for field in (
        "source_qcow2",
        "data_manifest",
        "runtime_manifest",
        "workdir",
        "ssh_key",
        "uefi_code",
        "uefi_vars",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("--source-sha256 must be lowercase SHA256")
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*:candidate-[a-zA-Z0-9_.-]+", args.image):
        parser.error("--image must have an explicit candidate-* tag, never a release/default alias")
    if min(args.cpus, args.memory_gib, args.boot_timeout) <= 0 or args.headroom_gib < 8:
        parser.error("positive CPU/memory/timeout and at least 8 GiB headroom required")
    if min(args.overlay_budget_gib, args.export_budget_gib) <= 0 or (
        args.docker_budget_gib is not None and args.docker_budget_gib <= 0
    ):
        parser.error("storage budgets must be positive")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", args.ssh_user):
        parser.error("invalid SSH user")
    if args.ssh_key is not None and not args.ssh_key.is_file():
        parser.error("--ssh-key must be an existing private key (public key is installed via CUA)")
    if bool(args.uefi_code) != bool(args.uefi_vars):
        parser.error("UEFI needs both --uefi-code and a --uefi-vars template")
    if args.builder_backend == "provider" and args.uefi_code:
        parser.error("provider mode uses its pinned runner firmware, not native --uefi-* files")
    if args.uefi_code and "," in str(args.workdir):
        parser.error("UEFI workdir must not contain commas")
    for path in (args.uefi_code, args.uefi_vars):
        if path and (not path.is_file() or "," in str(path.resolve())):
            parser.error("UEFI inputs must be existing files without commas in their paths")
    if any(os.environ.get(key, "0") != "0" for key in ("ALE_PUSH_IMAGE", "ALE_BUILD_ON_VM")):
        parser.error("local builds do not accept push or build-on-VM switches")
    return args


def build(args) -> None:
    os.umask(0o077)
    for executable in (
        "qemu-img",
        "ssh",
        "ssh-keygen",
        "docker",
        "zstd",
        "findmnt",
        "df",
    ):
        if shutil.which(executable) is None:
            raise ValueError(f"missing build prerequisite: {executable}")
    if args.builder_backend == "native" and args.stage != "finalize":
        if shutil.which("qemu-system-x86_64") is None:
            raise ValueError("native backend requires qemu-system-x86_64")
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise ValueError("native backend requires read/write access to /dev/kvm")
    persistent_mount(args.workdir)
    identity, contract, virtual_size = source_identity(args)
    if args.builder_backend == "provider":
        runner_id = capture("docker", "image", "inspect", "--format", "{{.Id}}", args.runner_image)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", runner_id):
            raise ValueError(
                "provider runner must already be installed locally with an immutable ID"
            )
        args.runner_image_id = runner_id
        identity["provider"] = {
            "runner_image_id": runner_id,
            "qemu_provider_sha256": sha256(HERE.parents[1] / "providers/qemu.py"),
        }
    if args.uefi_code:
        for path in (args.uefi_code, args.uefi_vars):
            persistent_mount(path, workspace=False)
        identity["firmware"] = {"code": sha256(args.uefi_code), "vars": sha256(args.uefi_vars)}
    docker_root = docker_storage()
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = workdir / "rootfs.tar.zst"
        receipt = workdir / "export.json"
        reuse = cached_export(archive, receipt, identity)
        if args.stage == "finalize" and not reuse:
            raise ValueError(
                "finalize requires a verified source-bound export; run --export-only first"
            )
        budget = dict(
            work_bytes=0 if reuse else (args.overlay_budget_gib + args.export_budget_gib) * GIB,
            docker_bytes=GIB if args.builder_backend == "provider" and not reuse else 0,
            memory_gib=args.memory_gib + 2,
            headroom_gib=args.headroom_gib,
        )
        check = capacity(workdir, docker_root, **budget)
        write_json(
            workdir / "preflight.json",
            {
                "identity": identity,
                "capacity": check,
                "source_virtual_bytes": virtual_size,
                "budgets": budget,
                "stage": args.stage,
                "reused_export": reuse,
            },
        )
        if args.preflight_only and args.stage != "finalize":
            print(f"Preflight passed; no allocations. Receipt: {workdir / 'preflight.json'}")
            return
        if args.stage != "export":
            images = capture("docker", "image", "ls", "--quiet", args.image)
            if images:
                raise ValueError("candidate tag already exists; use a new candidate tag")
        directory = workdir / uuid.uuid4().hex
        directory.mkdir()
        write_json(directory / "identity.json", identity)
        builder = None
        try:
            if not reuse:
                write_json(
                    directory / "capacity.before-vm.json", capacity(workdir, docker_root, **budget)
                )
                builder_class = ProviderBuilder if args.builder_backend == "provider" else Builder
                builder = builder_class(args, directory)
                builder.start()
                partial = directory / "rootfs.tar.zst.partial"
                write_json(
                    directory / "capacity.before-export.json",
                    capacity(workdir, docker_root, **budget),
                )
                builder.export(partial)
                builder.close()
                if sha256(args.source_qcow2) != identity["source_sha256"]:
                    raise ValueError("sealed source changed while exporting")
                inspection = audit_export(partial, contract)
                partial.replace(archive)
                write_json(
                    receipt,
                    {
                        "identity": identity,
                        "size": archive.stat().st_size,
                        "sha256": sha256(archive),
                        "inspection": inspection,
                        "package_inventory_sha256": sha256(directory / "packages.before.txt"),
                        "builder_evidence": str(directory),
                    },
                )
            inspection = audit_export(archive, contract)
            suggested_docker_gib = (inspection["logical_bytes"] * 2 + GIB - 1) // GIB + 20
            docker_gib = args.docker_budget_gib or suggested_docker_gib
            write_json(
                directory / "export-measurement.json",
                {
                    "compressed_bytes": archive.stat().st_size,
                    "inspection": inspection,
                    "suggested_docker_budget_gib": suggested_docker_gib,
                    "docker_budget_gib": docker_gib,
                    "docker_allocated": False,
                    "next_stage": "--finalize-only --preflight-only",
                },
            )
            if args.stage == "export":
                print(f"Export audited; no Docker image allocated. Measurements: {directory}")
                return
            if inspection["logical_bytes"] * 2 > docker_gib * GIB:
                raise ValueError("Docker budget too small for the inspected rootfs")
            write_json(directory / "archive-audit.json", inspection)
            budget.update(work_bytes=0, docker_bytes=docker_gib * GIB)
            write_json(
                directory / "capacity.before-import.json", capacity(workdir, docker_root, **budget)
            )
            if args.preflight_only:
                print(f"Finalize preflight passed; no image allocated. Measurements: {directory}")
                return
            export_receipt = json.loads(receipt.read_text())
            source_packages = Path(export_receipt["builder_evidence"]) / "packages.before.txt"
            image_id = finalize(archive, args.image, directory, contract, args, source_packages)
            write_json(
                directory / "candidate.json",
                {
                    "identity": identity,
                    "image": args.image,
                    "image_id": image_id,
                    "export_sha256": sha256(archive),
                    "build_smoke": "passed",
                    "release_validated": False,
                    "pending": [
                        "all-layer/content audit",
                        "review added desktop packages",
                        "official provider GUI/tool/agent smokes",
                        "all 99 task runs",
                    ],
                },
            )
            print(f"Candidate built, not release-validated: {args.image}; evidence: {directory}")
        finally:
            if builder is not None:
                builder.close()


def main() -> None:
    args = parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        build(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
