"""AntigravityCliDeployer — drives the Google Antigravity CLI (``agy``).

Shape: in-sandbox CLI (``executor=sandbox``), same as gemini_cli/claude_code.

The default ALE integration forwards an OAuth credential file the operator
produced by logging in once on the host:

  1. host (one-time):   ``agy``  → browser login → writes
     ``~/.gemini/antigravity-cli/antigravity-oauth-token`` (contains a refresh_token).
  2. env passthrough:   the lifecycle materialises that file's content into
     ``ANTIGRAVITY_OAUTH_TOKEN`` (or passes ``ANTIGRAVITY_OAUTH_TOKEN_PATH``).
  3. install() here:    writes it back to the same path inside the sandbox and
     ``chmod 600`` it, after which ``agy`` silent-auths headlessly.

GUI comes from the cua MCP bridge declared in ``agy``'s native
``~/.gemini/config/mcp_config.json`` (NOT the gemini-cli ``settings.json``).
Modern ``agy`` releases expose an NDJSON event stream, which is captured as
``transcript.jsonl``. ALE normalizes its per-generation usage and per-tool
duration into a task-level metrics summary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import AntigravityCliConfig
from .metrics import build_metrics_summary

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_UPDATER_BASE_URL = "https://antigravity-cli-auto-updater-974169037036.us-central1.run.app"
_PINNED_RELEASES: dict[tuple[str, str], tuple[str, str]] = {
    ("1.1.25", "linux_amd64"): (
        (
            "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
            "1.1.25-6680093607723008/linux-x64/cli_linux_x64.tar.gz"
        ),
        (
            "c5af6d1cfc2faf3d183b3497c56cb4951067bb56e8a1d1e5a4f081875aac2b07"
            "3bd97d0eb9458b772f2ddb934222372935f48ae2c7893d7013d975afd7516e6c"
        ),
    ),
    ("1.1.25", "windows_amd64"): (
        (
            "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
            "1.1.25-6680093607723008/windows-x64/cli_windows_x64.exe"
        ),
        (
            "55bfcfe11ac6196ac7c1ff440a9be96efd1bf8e2567b696d0dd92b64f53a37ae9"
            "83fcca571ff93d7b81c347cd124ba64d0769d212295504a4acd43a65f8924f2"
        ),
    ),
}

# agy stores its OAuth credential here (shared ~/.gemini home, agy-specific dir).
_TOKEN_RELPATH = (".gemini", "antigravity-cli", "antigravity-oauth-token")
_ACCOUNTS_RELPATH = (".gemini", "google_accounts.json")
_CREDENTIAL_TRANSPORT_VARS = (
    "ANTIGRAVITY_OAUTH_TOKEN",
    "ANTIGRAVITY_OAUTH_TOKEN_PATH",
    "ANTIGRAVITY_GOOGLE_ACCOUNTS",
)


def _installed_version(agy_path: str) -> str | None:
    try:
        # stdin=DEVNULL: a bare `agy --version` can otherwise wait on a TTY /
        # trip Defender on Windows.
        probe = subprocess.run(
            [agy_path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    m = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    return m.group(1) if m else None


def _win_appdata_bin(home: str) -> str:
    """``%LOCALAPPDATA%\\agy\\bin`` — where install.ps1 drops ``agy.exe``."""
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    return os.path.join(local_appdata, "agy", "bin")


def _find_agy(home: str, *, is_windows: bool) -> str | None:
    """Resolve the agy binary, preferring the installer's drop location.

    The sandbox entry runs without a login shell, so the install dir may not be
    on PATH and ``shutil.which`` misses the installer-dropped binary. Linux:
    ``~/.local/bin/agy``; Windows: ``%LOCALAPPDATA%\\agy\\bin\\agy.exe``."""
    cand = _install_target(home, is_windows=is_windows)
    if cand.is_file():
        return str(cand)
    return shutil.which("agy.exe" if is_windows else "agy") or shutil.which("agy")


def _install_target(home: str, *, is_windows: bool) -> Path:
    if is_windows:
        return Path(_win_appdata_bin(home)) / "agy.exe"
    return Path(home) / ".local" / "bin" / "agy"


def _official_release(version: str, *, is_windows: bool) -> tuple[str, str]:
    """Resolve an exact official release URL and checksum.

    The updater endpoint only exposes the latest release. Keep metadata for the
    pinned default locally so a later upstream release does not make an older,
    reproducible ALE config impossible to install.
    """
    platform_name = "windows_amd64" if is_windows else "linux_amd64"
    pinned = _PINNED_RELEASES.get((version, platform_name))
    if pinned is not None:
        return pinned
    manifest_url = f"{_UPDATER_BASE_URL}/manifests/{platform_name}.json"
    with urllib.request.urlopen(manifest_url, timeout=60) as response:
        manifest = json.load(response)
    actual = str(manifest.get("version", ""))
    if actual != version:
        raise RuntimeError(
            "antigravity_cli: configured version "
            f"{version} is not the current official manifest version {actual}; "
            "set download_url to an archived release or update cli_version"
        )
    url = str(manifest.get("url", ""))
    sha512 = str(manifest.get("sha512", ""))
    if not url or not re.fullmatch(r"[0-9a-fA-F]{128}", sha512):
        raise RuntimeError("antigravity_cli: updater manifest is missing url/sha512")
    return url, sha512.lower()


def _sha512(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_release(cfg: AntigravityCliConfig, home: str, *, is_windows: bool) -> None:
    """Download and install the configured release, verifying official hashes."""
    target = _install_target(home, is_windows=is_windows)
    target.parent.mkdir(parents=True, exist_ok=True)

    if cfg.download_url:
        url = cfg.download_url
        expected_sha512 = cfg.download_sha512.lower()
        if not re.fullmatch(r"[0-9a-f]{128}", expected_sha512):
            raise RuntimeError(
                "antigravity_cli: download_sha512 must be a SHA-512 hex digest "
                "when download_url is set"
            )
    else:
        url, expected_sha512 = _official_release(cfg.cli_version, is_windows=is_windows)

    with tempfile.TemporaryDirectory(prefix="ale-agy-install-") as tmp:
        payload = Path(tmp) / ("agy.exe" if is_windows else "agy.tar.gz")
        urllib.request.urlretrieve(url, payload)
        if expected_sha512:
            actual_sha512 = _sha512(payload)
            if actual_sha512 != expected_sha512:
                raise RuntimeError(
                    "antigravity_cli: release checksum mismatch "
                    f"({actual_sha512} != {expected_sha512})"
                )

        source = payload
        if not is_windows:
            with tarfile.open(payload, "r:gz") as archive:
                member = next(
                    (
                        item
                        for item in archive.getmembers()
                        if item.isfile() and Path(item.name).name in {"antigravity", "agy"}
                    ),
                    None,
                )
                if member is None:
                    raise RuntimeError("antigravity_cli: release archive has no CLI binary")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError("antigravity_cli: could not extract CLI binary")
                source = Path(tmp) / "agy"
                source.write_bytes(extracted.read())

        staged = target.with_suffix(target.suffix + ".new")
        shutil.copyfile(source, staged)
        staged.chmod(0o755)
        os.replace(staged, target)


def _spawn_agy(
    argv: list[str],
    *,
    transcript_file: Path,
    stderr_log: Path,
    env: dict[str, str],
    cwd: Path,
) -> subprocess.Popen:
    """Open launch files and spawn agy outside the asyncio event loop."""
    with transcript_file.open("wb") as tout, stderr_log.open("wb") as terr:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=tout,
            stderr=terr,
            env=env,
            cwd=str(cwd),
            start_new_session=bool(hasattr(os, "setsid")),
        )


class AntigravityCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the Google ``agy`` CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "transcript.jsonl",
        "stderr.log",
        "agy_cli.log",
    )

    @property
    def version(self) -> str | None:
        cfg: AntigravityCliConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: AntigravityCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox
        self._is_windows = not sandbox.is_linux

        home = os.path.expanduser("~")

        # 1. Locate/install agy and enforce the exact configured version. The
        # official bootstrapper installs "latest" and refuses to replace an
        # existing binary, so use the signed updater manifest directly.
        agy = _find_agy(home, is_windows=self._is_windows)
        installed = await asyncio.to_thread(_installed_version, agy) if agy else None
        stale = bool(agy and cfg.cli_version and installed and installed != cfg.cli_version)
        if not agy or stale:
            if stale:
                logger.info(
                    "antigravity_cli: %s != pinned %s — reinstalling", installed, cfg.cli_version
                )
            await self._install_agy(cfg, home)
            agy = _find_agy(home, is_windows=self._is_windows)
            if not agy:
                raise RuntimeError("antigravity_cli: 'agy' not found after install")
            installed = await asyncio.to_thread(_installed_version, agy)
        if cfg.cli_version and installed != cfg.cli_version:
            raise RuntimeError(
                f"antigravity_cli: installed version {installed!r} does not match "
                f"configured {cfg.cli_version!r}"
            )
        self._agy_path = agy
        # Put the install dir on PATH so launch() and any self-update find it.
        bin_dir = (
            _win_appdata_bin(home) if self._is_windows else os.path.join(home, ".local", "bin")
        )
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("antigravity_cli: CLI ok — agy %s at %s", installed or "?", agy)

        # 2. clean work dir
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 3. inject the OAuth credential the operator produced on the host.
        self._write_oauth_token()

        # 4. cua GUI bridge + agy config. Idempotent bridge install.
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server

        await ensure_cua_mcp_server(sandbox)

        gemini_home = Path(home) / ".gemini"
        gemini_home.mkdir(parents=True, exist_ok=True)
        self._gemini_dir = str(gemini_home)
        cua_server = {
            "cua": {
                "command": sandbox.node,
                "args": [
                    self._join(sandbox.mcp_server_dir, "src", "index.js", is_linux=sandbox.is_linux)
                ],
                "env": cua_bridge_env(self.executor),
            },
        }
        # agy reads its MCP servers from ~/.gemini/config/mcp_config.json (its
        # NATIVE config), NOT the gemini-cli settings.json — that is where the
        # cua GUI tools (screenshot/click/type/…) must be declared to load.
        config_dir = gemini_home / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "mcp_config.json").write_text(
            json.dumps({"mcpServers": cua_server}, indent=2),
            encoding="utf-8",
        )
        # agy >= 1.1 reads CLI policy from its own app-data directory. Keep the
        # permission mode explicit even though launch also passes the one-run
        # override; this prevents artifact/tool review prompts in headless mode.
        settings = {
            "toolPermission": (
                "always-proceed" if cfg.dangerously_skip_permissions else "request-review"
            ),
            "artifactReviewPolicy": (
                "always-proceed" if cfg.dangerously_skip_permissions else "asks-for-review"
            ),
        }
        cli_app_dir = gemini_home / "antigravity-cli"
        cli_app_dir.mkdir(parents=True, exist_ok=True)
        (cli_app_dir / "settings.json").write_text(
            json.dumps(settings, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "antigravity_cli: config staged at %s (cua -> config/mcp_config.json)", gemini_home
        )

        # 5. Windows: pre-warm node + the cua bridge modules. A COLD node start
        #    on Windows (Defender scan + ESM module load of the MCP SDK) is slow
        #    enough to intermittently miss agy's MCP tool-discovery window at
        #    launch, so the cua GUI tools register only ~1 run in 4. One warm-up
        #    run loads the modules into the OS file cache and lets Defender scan
        #    them once, so agy's spawn of the bridge at launch is fast and wins
        #    the race. (Linux node start is fast; no warm-up needed there.)
        if self._is_windows:
            self._ensure_grep_windows()
            await self._prewarm_bridge(sandbox)
            await self._prewarm_agy()

    def _ensure_grep_windows(self) -> None:
        """Put ``grep`` on PATH for agy's ``grep_search`` tool.

        agy shells out to ``grep``, which isn't on the Windows sandbox PATH by
        default — so ``grep_search`` fails with ``exec: 'grep': ... not found``.
        Git for Windows (baked into ale-win10) already ships a real GNU grep at
        ``…\\Git\\usr\\bin\\grep.exe`` (with its DLLs co-located); it's just not
        on PATH. Prepend that dir. As a fallback for an image WITHOUT Git, fetch
        a single-file busybox-w32 and run it as ``grep.exe``. Best-effort: a
        failure only loses ``grep_search``, not the run.
        """
        if shutil.which("grep"):
            return
        home = os.path.expanduser("~")
        for d in (
            r"C:\Program Files\Git\usr\bin",
            r"C:\Program Files (x86)\Git\usr\bin",
            os.path.join(home, "AppData", "Local", "Programs", "Git", "usr", "bin"),
        ):
            if os.path.isfile(os.path.join(d, "grep.exe")):
                self._prepend_path(d)
                logger.info("antigravity_cli: grep via Git for Windows (%s on PATH)", d)
                return
        # No Git grep — fetch busybox-w32 (single exe; run as grep.exe → grep applet).
        bin_dir = _win_appdata_bin(home)
        try:
            import urllib.request

            os.makedirs(bin_dir, exist_ok=True)
            grep_exe = os.path.join(bin_dir, "grep.exe")
            urllib.request.urlretrieve("https://frippery.org/files/busybox/busybox.exe", grep_exe)
            self._prepend_path(bin_dir)
            logger.info("antigravity_cli: installed busybox grep at %s", grep_exe)
        except Exception as e:  # noqa: BLE001 — best-effort, non-fatal
            logger.warning(
                "antigravity_cli: could not provision grep (grep_search "
                "will be unavailable on Windows): %s",
                e,
            )

    @staticmethod
    def _prepend_path(directory: str) -> None:
        if directory not in os.environ.get("PATH", ""):
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")

    async def _prewarm_agy(self) -> None:
        """Run ``agy models`` once so the FIRST-run side effects (config
        migration, the background auto-updater) are done before the real launch.
        On a fresh Windows VM those first-run steps otherwise race with MCP tool
        discovery, so the cua tools register only intermittently; priming makes
        the real launch a reliable 'subsequent' run.

        Uses ``models`` (a metadata call), NOT a ``-p`` generation turn, so it
        consumes **no model quota** — a per-task generation warm-up would double
        quota usage across a benchmark and exhaust the account.
        """
        # gemini_dir is a global flag. agy 1.1.25 rejects it when it appears
        # after the subcommand (``agy models --gemini_dir=...``).
        argv = [self._agy_path, f"--gemini_dir={self._gemini_dir}", "models"]
        env = self._without_credential_transport(os.environ.copy())
        try:
            await asyncio.to_thread(
                subprocess.run,
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                env=env,
            )
            logger.info(
                "antigravity_cli: warmed up agy (first-run migration primed, no quota used)"
            )
        except subprocess.TimeoutExpired:
            logger.info("antigravity_cli: agy warm-up timed out (continuing)")
        except (OSError, subprocess.SubprocessError) as e:
            logger.info("antigravity_cli: agy warm-up skipped: %s", e)

    async def _prewarm_bridge(self, sandbox) -> None:
        """Spawn the cua bridge once so node + its modules are warm/cached."""
        from ale_run.agents._bootstrap import cua_bridge_env

        index_js = self._join(sandbox.mcp_server_dir, "src", "index.js", is_linux=sandbox.is_linux)
        env = self._without_credential_transport({**os.environ, **cua_bridge_env(self.executor)})
        try:
            # stdin=DEVNULL → the stdio MCP server gets EOF and idles/exits once
            # its modules are loaded; the timeout caps the (slow, one-time) cold
            # start. We only care that the modules end up cached, not the output.
            await asyncio.to_thread(
                subprocess.run,
                [sandbox.node, index_js],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=45,
                env=env,
            )
        except subprocess.TimeoutExpired:
            pass
        except (OSError, subprocess.SubprocessError) as e:
            logger.info("antigravity_cli: bridge pre-warm skipped: %s", e)
            return
        logger.info("antigravity_cli: pre-warmed cua bridge (node modules cached)")

    async def _install_agy(self, cfg: AntigravityCliConfig, home: str) -> None:
        """Install the exact configured agy release on Linux or Windows."""
        await asyncio.to_thread(_install_release, cfg, home, is_windows=self._is_windows)
        logger.info("antigravity_cli: installed agy %s", cfg.cli_version)

    def _write_oauth_token(self) -> None:
        """Write agy's OAuth credential into place from env passthrough.

        Resolution order (mirrors cursor_cli's auth.json handling):
        1. ``ANTIGRAVITY_OAUTH_TOKEN``       — raw token-file JSON content.
        2. ``ANTIGRAVITY_OAUTH_TOKEN_PATH``  — path to the token file.
        Optional ``ANTIGRAVITY_GOOGLE_ACCOUNTS`` writes google_accounts.json.
        Security: never log the content, only byte counts.
        """
        home = Path(os.path.expanduser("~"))
        token_file = home.joinpath(*_TOKEN_RELPATH)
        token_file.parent.mkdir(parents=True, exist_ok=True)

        content = os.environ.get("ANTIGRAVITY_OAUTH_TOKEN", "").strip()
        if not content:
            path = os.environ.get("ANTIGRAVITY_OAUTH_TOKEN_PATH", "").strip()
            if path and Path(path).expanduser().is_file():
                content = Path(path).expanduser().read_text(encoding="utf-8")
        if not content:
            raise RuntimeError(
                "antigravity_cli: no OAuth credential. Log in once on the host "
                "(`agy`) then set ANTIGRAVITY_OAUTH_TOKEN_PATH to "
                "~/.gemini/antigravity-cli/antigravity-oauth-token (or "
                "ANTIGRAVITY_OAUTH_TOKEN to its content)."
            )
        token_file.write_text(content, encoding="utf-8")
        token_file.chmod(0o600)
        logger.info("antigravity_cli: wrote OAuth token (%d B)", len(content))

        accounts = os.environ.get("ANTIGRAVITY_GOOGLE_ACCOUNTS", "").strip()
        if accounts:
            af = home.joinpath(*_ACCOUNTS_RELPATH)
            af.write_text(accounts, encoding="utf-8")
            logger.info("antigravity_cli: wrote google_accounts.json (%d B)", len(accounts))

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: AntigravityCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        agy_log = wd / "agy_cli.log"
        pid_file = wd / "agy.pid"
        for f in (transcript_file, stderr_log, agy_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass
        prompt_file.write_text(prompt, encoding="utf-8")

        # agy's print flag consumes its argument; unlike Gemini CLI, a lone
        # hyphen means the literal prompt "-" rather than "read stdin".
        argv = [self._agy_path, f"--print={prompt}"]
        # Pin the gemini dir to an ABSOLUTE path: agy resolves a relative
        # ".gemini" against CWD on Windows and falls back to a default, which
        # makes config discovery (incl. the cua MCP server) non-deterministic.
        gemini_dir = getattr(self, "_gemini_dir", "")
        if gemini_dir:
            argv.append(f"--gemini_dir={gemini_dir}")
        # Give every task its own native agy log. Copying agy's global cli.log
        # symlink after exit races under concurrency and can capture a different
        # task's log. Direct logging also lets the lifecycle incrementally pull
        # this file while the task is still running.
        argv.append(f"--log-file={agy_log}")
        if cfg.model:
            argv += ["--model", cfg.model]
        argv += ["--output-format", "stream-json"]
        # Raise agy's print-mode timeout well above the wall budget — its 5m
        # default silently truncates longer tasks (no output written).
        if getattr(cfg, "print_timeout", ""):
            argv.append(f"--print-timeout={cfg.print_timeout}")
        if cfg.dangerously_skip_permissions:
            argv.append("--dangerously-skip-permissions")
        # agy file tools reject paths outside the workspace; add the task data
        # root (outside cwd) as an extra workspace dir.
        task_data_root = getattr(self.executor.sandbox, "task_data_root", "")
        if task_data_root:
            argv += ["--add-dir", task_data_root]

        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        # The OAuth credential reaches agy via the file written in install(), NOT
        # the env. Strip the transport vars so the long-lived refresh token never
        # enters agy's process env (and thus any shell command the agent runs).
        self._without_credential_transport(env)
        logger.info(
            "antigravity_cli: launching agy (model=%r, prompt_chars=%d, stream_json=true)",
            cfg.model,
            len(prompt),
        )

        t0 = time.monotonic()
        proc = await asyncio.to_thread(
            _spawn_agy,
            argv,
            transcript_file=transcript_file,
            stderr_log=stderr_log,
            env=env,
            cwd=wd,
        )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("antigravity_cli: spawned pid=%s", proc.pid)

        try:
            while proc.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            self._terminate_proc_group(proc, force=False)
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S)
            except (TimeoutError, asyncio.CancelledError):
                self._terminate_proc_group(proc, force=True)
            raise

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        native_result = _read_result_envelope(transcript_file)
        native_status = str((native_result or {}).get("status") or "").upper()
        status = "completed" if exit_code == 0 and native_status == "SUCCESS" else "failed"
        error: str | None = None
        if status == "failed":
            native_error = (native_result or {}).get("error")
            if native_status and native_status != "SUCCESS" and native_error:
                error = f"agy result status={native_status}: {_json_text(native_error)}"
            elif exit_code == 0 and native_status != "SUCCESS":
                error = "agy exited without a terminal SUCCESS result"
            else:
                error = self._diagnose_failure(stderr_log, transcript_file, exit_code)
        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(transcript_file),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    @staticmethod
    def _terminate_proc_group(proc: subprocess.Popen, *, force: bool) -> None:
        """Terminate agy and its MCP/subagent child processes together."""
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(
                    os.getpgid(proc.pid),
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            elif force:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    @staticmethod
    def _without_credential_transport(env: dict[str, str]) -> dict[str, str]:
        """Remove host-to-sandbox credential carriers from a child env."""
        for name in _CREDENTIAL_TRANSPORT_VARS:
            env.pop(name, None)
        return env

    # =========================================================================
    # internals
    # =========================================================================

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        sep = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    def _diagnose_failure(self, stderr_log: Path, transcript: Path, exit_code: int | None) -> str:
        parts = [f"agent failed (rc={exit_code})"]
        st = _read_text_tolerant(stderr_log)
        tx = _read_text_tolerant(transcript)
        if st.strip():
            parts.append(f"stderr tail: ...{st[-800:]}")
        if tx.strip():
            parts.append(f"transcript tail: ...{tx[-800:]}")
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: AntigravityCliConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Convert agy's native event stream into an ATIF trajectory."""
        transcript_file = work_dir / "transcript.jsonl"
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"antigravity-cli: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            try:
                metrics = build_metrics_summary(work_dir, model=config.model or None)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("antigravity_cli: could not build metrics summary: %s", exc)
            else:
                builder.trajectory.extra.setdefault("antigravity_cli", {}).update(
                    {
                        "exit_code": run_result.exit_code,
                        "transcript_path": str(transcript_file),
                        "agy_log_path": str(work_dir / "agy_cli.log"),
                        "metrics_summary_path": str(work_dir / "metrics_summary.json"),
                        "metrics_availability": metrics.get("availability"),
                    }
                )
            return
        raw = _strip_ansi(transcript_file.read_text(encoding="utf-8", errors="replace"))
        native_events: list[dict] = []
        for line in raw.splitlines():
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                native_events.append(decoded)

        envelope: dict | None = None
        for event in native_events:
            if event.get("event") == "result" and isinstance(event.get("result"), dict):
                envelope = event["result"]
        if (
            envelope is None
            and len(native_events) == 1
            and (
                "response" in native_events[0]
                or "status" in native_events[0]
                or "usage" in native_events[0]
            )
        ):
            envelope = native_events[0]

        metadata: dict = {
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
        }
        before_stream_steps = len(builder.trajectory.steps)
        added_generation = cls._consume_stream_steps(native_events, builder)
        stream_steps_added = len(builder.trajectory.steps) > before_stream_steps
        if envelope is not None and (
            "response" in envelope or "status" in envelope or "usage" in envelope
        ):
            usage = envelope.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            response = envelope.get("response")
            if not added_generation and isinstance(response, str) and response.strip():
                builder.add_step(
                    source="agent",
                    message=response,
                    metrics=StepMetrics(
                        input_tokens=_optional_int(usage.get("input_tokens")),
                        output_tokens=_optional_int(usage.get("output_tokens")),
                        cache_read_tokens=_optional_int(usage.get("cache_read_tokens")),
                        duration_ms=_duration_ms(envelope.get("duration_seconds")),
                    ),
                    extra={"thinking_tokens": _optional_int(usage.get("thinking_tokens"))},
                )
            error = envelope.get("error")
            if isinstance(error, str) and error.strip():
                builder.add_step(
                    source="system",
                    message=error,
                    extra={"reason": "agy_error", "status": envelope.get("status")},
                )
            builder.override_final_metrics(
                total_input_tokens=_optional_int(usage.get("input_tokens")),
                total_output_tokens=_optional_int(usage.get("output_tokens")),
                total_cache_read_tokens=_optional_int(usage.get("cache_read_tokens")),
            )
            metadata.update(
                {
                    "conversation_id": envelope.get("conversation_id"),
                    "status": envelope.get("status"),
                    "num_turns": envelope.get("num_turns"),
                    "usage": usage,
                }
            )
        elif not native_events:
            lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
            if lines:
                builder.add_step(source="agent", message="\n".join(lines))
        elif not stream_steps_added:
            builder.add_step(
                source="system",
                message="antigravity-cli: native stream ended without result or usable steps",
                extra={"reason": "incomplete_native_stream"},
            )

        try:
            metrics = build_metrics_summary(work_dir, model=config.model or None)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("antigravity_cli: could not build metrics summary: %s", exc)
        else:
            metadata.update(
                {
                    "agy_log_path": str(work_dir / "agy_cli.log"),
                    "metrics_summary_path": str(work_dir / "metrics_summary.json"),
                    "metrics_availability": metrics.get("availability"),
                }
            )
        builder.trajectory.extra.setdefault("antigravity_cli", {}).update(metadata)

    @classmethod
    def _consume_stream_steps(
        cls,
        native_events: list[dict],
        builder: TrajectoryBuilder,
    ) -> bool:
        """Map agy's step stream to ordered ATIF generation/tool steps.

        ``agy`` emits multiple ACTIVE ``text_delta`` records followed by a DONE
        record for one agent response. Tool records similarly have ACTIVE and
        terminal copies. Grouping by ``step_index`` preserves the complete text
        while emitting every tool exactly once.
        """
        updates: dict[int, list[dict]] = {}
        order: list[int] = []
        for event in native_events:
            if event.get("event") != "step_update":
                continue
            update = event.get("step_update")
            if not isinstance(update, dict):
                continue
            step_index = _optional_int(update.get("step_index"))
            if step_index is None:
                continue
            if step_index not in updates:
                updates[step_index] = []
                order.append(step_index)
            updates[step_index].append(update)

        added_generation = False
        for step_index in order:
            records = updates[step_index]
            terminal = next(
                (
                    record
                    for record in reversed(records)
                    if record.get("state") in {"DONE", "ERROR"}
                ),
                None,
            )
            if terminal is None:
                latest = records[-1]
                if latest.get("step_type") == "tool":
                    cls._consume_tool_step(latest, step_index, builder, include_result=False)
                elif latest.get("step_type") == "agent_response":
                    chunks = [
                        record["text_delta"]
                        for record in records
                        if isinstance(record.get("text_delta"), str) and record["text_delta"]
                    ]
                    if chunks:
                        builder.add_step(
                            source="agent",
                            message="".join(chunks),
                            extra={
                                "agy_step_index": step_index,
                                "agy_step_type": "agent_response",
                                "incomplete": True,
                            },
                        )
                        added_generation = True
                continue
            step_type = str(terminal.get("step_type") or "")
            if step_type == "agent_response":
                chunks = [
                    record["text_delta"]
                    for record in records
                    if isinstance(record.get("text_delta"), str) and record["text_delta"]
                ]
                step_usage = terminal.get("usage")
                if not isinstance(step_usage, dict):
                    step_usage = {}
                builder.add_step(
                    source="agent",
                    message="".join(chunks) or None,
                    metrics=StepMetrics(
                        input_tokens=_optional_int(step_usage.get("input_tokens")),
                        output_tokens=_optional_int(step_usage.get("output_tokens")),
                        cache_read_tokens=_optional_int(step_usage.get("cache_read_tokens")),
                        duration_ms=_duration_ms(terminal.get("duration_seconds")),
                    ),
                    extra={
                        "thinking_tokens": _optional_int(step_usage.get("thinking_tokens")),
                        "agy_step_index": step_index,
                        "agy_step_type": step_type,
                    },
                )
                added_generation = True
            elif step_type == "tool":
                cls._consume_tool_step(terminal, step_index, builder)
            elif step_type not in {"", "user_input"}:
                builder.add_step(
                    source="system",
                    message=None,
                    extra={
                        "agy_step_index": step_index,
                        "agy_step_type": step_type,
                        "state": terminal.get("state"),
                    },
                )
        return added_generation

    @staticmethod
    def _consume_tool_step(
        update: dict,
        step_index: int,
        builder: TrajectoryBuilder,
        *,
        include_result: bool = True,
    ) -> None:
        tool_info = update.get("tool_info")
        if not isinstance(tool_info, dict):
            tool_info = {}
        tool_name = str(update.get("tool_name") or tool_info.get("name") or "unknown")
        arguments = tool_info.get("parameters")
        if not isinstance(arguments, dict):
            arguments = {"value": arguments} if arguments is not None else {}
        call_id = f"agy_step_{step_index}"
        builder.add_step(
            source="agent",
            tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=arguments)],
            extra={
                "agy_step_index": step_index,
                "state": update.get("state"),
                "duration_ms": _duration_ms(update.get("duration_seconds")),
                "incomplete": not include_result,
            },
        )

        if not include_result:
            return

        output = tool_info.get("output")
        error = tool_info.get("error")
        content: list[ContentPart] = []
        if output is not None:
            content.append(ContentPart(type="text", text=_json_text(output)))
        if error is not None:
            content.append(ContentPart(type="text", text=_json_text(error)))
        builder.add_step(
            source="environment",
            observation=Observation(
                results=[
                    ToolResult(
                        tool_call_id=call_id,
                        content=content,
                        is_error=update.get("state") == "ERROR" or error is not None,
                    )
                ],
            ),
            extra={
                "agy_step_index": step_index,
                "state": update.get("state"),
                "duration_ms": _duration_ms(update.get("duration_seconds")),
            },
        )


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _duration_ms(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(value * 1000)
    return None


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_result_envelope(path: Path) -> dict | None:
    """Return the last native result record, tolerating partial JSONL tails."""
    for line in reversed(_read_text_tolerant(path).splitlines()):
        try:
            record = json.loads(_strip_ansi(line))
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        result = record.get("result")
        if record.get("event") == "result" and isinstance(result, dict):
            return result
        if any(key in record for key in ("response", "status", "usage")):
            return record
    return None
