import ast
from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ale_run/environments/images/ale_ubuntu22_docker/cleanup.sh"
)


def test_temporary_directories_are_restored_before_package_commands():
    script = SCRIPT.read_text()
    bootstrap, desktop = script.split("# --- desktop:", 1)
    assert "apt-get" in desktop
    assert "mkdir -p /tmp/.X11-unix /var/tmp" in bootstrap
    assert "chmod 1777 /tmp /tmp/.X11-unix /var/tmp" in bootstrap


def test_bootstrap_creates_directories_before_setting_sticky_permissions():
    bootstrap = SCRIPT.read_text().split("# --- desktop:", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'mkdir() { printf "mkdir %s\\n" "$*"; }; '
            'chmod() { printf "chmod %s\\n" "$*"; };\n' + bootstrap,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "chmod +x /dockerstartup/entrypoint.sh",
        "mkdir -p /tmp/.X11-unix /var/tmp",
        "chmod 1777 /tmp /tmp/.X11-unix /var/tmp",
    ]


def test_desktop_install_uses_official_sources_without_editing_repo_config():
    script = SCRIPT.read_text()
    assert (
        "apt_sources=(-o Dir::Etc::sourcelist=/etc/apt/sources.list -o Dir::Etc::sourceparts=-)"
    ) in script
    assert script.count('apt-get "${apt_sources[@]}" update -qq') == 2
    assert script.count('apt-get "${apt_sources[@]}" install') == 2
    assert "x11-xserver-utils thunar" in script


def test_gui_smoke_sets_its_own_display_for_docker_exec():
    module = ast.parse((SCRIPT.parent / "local_build.py").read_text())
    smoke = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "smoke"
    )
    commands = [
        [argument.value for argument in node.args if isinstance(argument, ast.Constant)]
        for node in ast.walk(smoke)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run"
    ]
    command = next(command for command in commands if command[:2] == ["docker", "exec"])
    assert command[2:5] == [
        "env",
        "DISPLAY=:0",
        "XAUTHORITY=/home/user/.Xauthority",
    ]


def test_final_cleanup_removes_transient_files_without_touching_runtime(tmp_path):
    contents = {
        "var/log/apt/history.log": "build log",
        "var/log/dpkg.log": "package installation",
        "var/cache/apt/archives/lock": "",
        "var/cache/man/index.db": "regenerable index",
        "var/cache/ldconfig/aux-cache": "regenerable cache",
        "var/cache/debconf/passwords.dat": "",
        "var/cache/debconf/config.dat": "installed package settings",
        "var/lib/dpkg/status": "installed package database",
        "opt/scientific-tool/runtime": "installed runtime",
    }
    for name, value in contents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    (tmp_path / "var/log/external-link").symlink_to(tmp_path / "opt")
    script = SCRIPT.read_text().split('echo "--- scrub build-only caches and install logs ---"', 1)[
        1
    ]
    script = script.replace("/var/", str(tmp_path / "var") + "/")
    subprocess.run(["bash", "-euc", script], check=True, capture_output=True, text=True)
    for name in (
        "var/cache/debconf/config.dat",
        "var/lib/dpkg/status",
        "opt/scientific-tool/runtime",
    ):
        assert (tmp_path / name).read_text() == contents.pop(name)
    assert all(not (tmp_path / name).exists() for name in contents)
    assert (tmp_path / "var/cache/apt/archives/partial").is_dir()
    assert (tmp_path / "var/log/apt").is_dir()
    assert not (tmp_path / "var/log/external-link").is_symlink()


def test_final_cleanup_accepts_absent_transient_directories(tmp_path):
    script = SCRIPT.read_text().split('echo "--- scrub build-only caches and install logs ---"', 1)[
        1
    ]
    script = script.replace("/var/", str(tmp_path / "var") + "/")
    subprocess.run(["bash", "-euc", script], check=True, capture_output=True, text=True)
    assert (tmp_path / "var/cache/apt/archives/partial").is_dir()
    assert (tmp_path / "var/log/apt").is_dir()
