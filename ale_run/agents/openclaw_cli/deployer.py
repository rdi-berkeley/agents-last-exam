"""OpenClawCliDeployer — drives ``openclaw agent --local``.

Fork tarball + native CUA plugin (not MCP).  The deployer expects both
the tarball and plugin source to be available inside the sandbox at
configurable paths (baked into the image or volume-mounted).

Config files written: ``openclaw.json``, ``auth-profiles.json``,
``exec-approvals.json``, ``workspace-state.json``.

Output: JSON envelope on stderr (``--json`` flag), plus session
trajectory JSONL at ``~/.openclaw/agents/main/sessions/``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import CUA_TOOL_NAMES, OpenClawCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_AGENT_ID = "main"

_STDERR_PREAMBLE_PREFIXES = (
    "[agent/embedded] session file repaired",
    "[agent/embedded] embedded run agent end",
    "[agent/embedded] embedded run failover decision",
    "[diagnostic] lane task error",
    "[model-fallback/decision]",
)


class OpenClawCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for ``openclaw agent --local``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("stderr.log",)

    # =========================================================================
    # install
    # =========================================================================

    async def _install_from_tarball(self, tarball: str) -> None:
        from ale_run.agents._bootstrap import ensure_npm
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            npm = await ensure_npm()
        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        env = {**os.environ, "npm_config_cache": os.path.join(home, ".npm-ale")}
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--prefix", prefix, tarball],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {tarball} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        # npm drops the openclaw shim in <prefix>/bin (Linux) or directly in
        # <prefix> (Windows). Put both on PATH so shutil.which finds it.
        for bin_dir in (prefix, os.path.join(prefix, "bin")):
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("openclaw_cli: installed from tarball — %s",
                     (proc.stdout or "").strip()[-200:])

        # Verify dist integrity. A truncated tarball download or a partial
        # npm extraction silently drops esbuild dynamic-import chunks; the CLI
        # then dies at runtime with ERR_MODULE_NOT_FOUND for the missing
        # provider-runtime / transcript-resolve chunks. Fail fast here with a
        # clear cause rather than letting a broken install reach launch.
        # Global node_modules layout differs per OS: <prefix>/lib/node_modules
        # on Linux, <prefix>/node_modules on Windows.
        for nm in (
            Path(prefix) / "lib" / "node_modules" / "openclaw" / "dist",
            Path(prefix) / "node_modules" / "openclaw" / "dist",
        ):
            if nm.is_dir():
                dist_dir = nm
                break
        else:
            dist_dir = Path(prefix) / "lib" / "node_modules" / "openclaw" / "dist"
        chunk_count = len(list(dist_dir.glob("*.js"))) if dist_dir.is_dir() else 0
        if chunk_count < 1000:
            raise RuntimeError(
                f"openclaw tarball install looks incomplete: only {chunk_count} "
                f"dist/*.js chunks under {dist_dir} (expected >2000). The "
                f"tarball download was likely truncated — re-run or bake "
                f"openclaw into the image."
            )

    async def _download_tarball(self, url: str) -> str:
        """Download fork tarball from a URL, return local path."""
        home = os.path.expanduser("~")
        dest = Path(home) / ".ale-openclaw-fork.tgz"
        # --retry guards against transient drops that produce a truncated
        # tarball (the historic root cause of ERR_MODULE_NOT_FOUND at launch:
        # npm extracts the partial tgz, silently dropping dist chunks).
        proc = await asyncio.to_thread(
            subprocess.run,
            ["curl", "-fSL", "--retry", "3", "--retry-all-errors",
             "-o", str(dest), url],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"openclaw tarball download failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        size = dest.stat().st_size if dest.exists() else 0
        if size < 1_000_000:
            raise RuntimeError(
                f"openclaw tarball download truncated: {size} bytes at {dest} "
                f"(expected >1MB)"
            )
        logger.info("openclaw_cli: tarball downloaded to %s (%d bytes)", dest, size)
        return str(dest)

    async def _clone_cua_plugin(self, repo_url: str, branch: str) -> str:
        """Clone the CUA plugin source from the fork repo."""
        home = os.path.expanduser("~")
        clone_dir = Path(home) / ".ale-openclaw-repo"
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        proc = await asyncio.to_thread(
            subprocess.run,
            ["git", "clone", "--depth", "1", "-b", branch,
             "--filter=blob:none", "--sparse", repo_url, str(clone_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"openclaw repo clone failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        proc2 = await asyncio.to_thread(
            subprocess.run,
            ["git", "sparse-checkout", "set", "cua-plugin"],
            capture_output=True, text=True, timeout=30,
            cwd=str(clone_dir),
        )
        if proc2.returncode != 0:
            raise RuntimeError(
                f"sparse-checkout failed (rc={proc2.returncode}): "
                f"{(proc2.stderr or '')[:500]}"
            )
        plugin_path = str(clone_dir / "cua-plugin")
        logger.info("openclaw_cli: CUA plugin cloned to %s", plugin_path)
        return plugin_path

    async def _build_cua_plugin(self, plugin_src: str) -> None:
        """Build CUA plugin from source and install to ~/.openclaw/extensions/cua/."""
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            from ale_run.agents._bootstrap import ensure_npm
            npm = await ensure_npm()

        home = os.path.expanduser("~")
        build_dir = Path(home) / ".ale-cua-plugin-build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        shutil.copytree(plugin_src, str(build_dir))

        env = {**os.environ, "npm_config_cache": f"{home}/.npm-ale"}
        for step_name, step_cmd in [
            ("npm install", [npm, "install", "--no-audit", "--no-fund"]),
            ("npm run build", [npm, "run", "build"]),
        ]:
            proc = await asyncio.to_thread(
                subprocess.run, step_cmd,
                capture_output=True, text=True, timeout=120,
                cwd=str(build_dir), env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CUA plugin {step_name} failed (rc={proc.returncode}): "
                    f"{(proc.stderr or '')[:500]}"
                )

        install_dir = Path(home) / ".openclaw" / "extensions" / "cua"
        install_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("package.json", "openclaw.plugin.json"):
            src = build_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(install_dir / fname))
        dist_src = build_dir / "dist" / "index.cjs"
        if dist_src.exists():
            (install_dir / "dist").mkdir(exist_ok=True)
            shutil.copy2(str(dist_src), str(install_dir / "dist" / "index.cjs"))
        logger.info("openclaw_cli: CUA plugin installed at %s", install_dir)

    @staticmethod
    def _route_model(model: str, provider: str) -> str:
        """Prefix a model ref with the auth provider so openclaw routes
        it correctly.  Without the prefix, ``anthropic/claude-...`` is sent
        to the ``anthropic`` provider directly, which has no configured key.
        Models already prefixed with a known provider are left untouched.
        """
        if model.split("/", 1)[0] == provider:
            return model
        return f"{provider}/{model}"

    @staticmethod
    def _direct_provider_for_model(model: str) -> str:
        """Map a model ref to its native (direct) provider id.

        Used only when ``provider == "direct"``: openclaw's upstream has
        first-class ``openai`` and ``anthropic`` providers, so a direct run
        routes to whichever vendor owns the model. The vendor is taken from
        an explicit ``<vendor>/`` prefix when present, otherwise inferred
        from the bare model name.
        """
        head, _, _ = model.partition("/")
        if head in ("openai", "anthropic"):
            return head
        name = model.rsplit("/", 1)[-1].lower()
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith(("gpt", "o1", "o3", "o4")):
            return "openai"
        raise RuntimeError(
            f"openclaw_cli: provider=direct cannot infer a native provider "
            f"for model {model!r}. Use an OpenAI (gpt-*) or Anthropic "
            f"(claude-*) model, or prefix it with 'openai/' or 'anthropic/'."
        )

    def _resolve_routing(self, cfg: OpenClawCliConfig) -> tuple[str, str]:
        """Return ``(provider, api_key)`` for the configured routing.

        Explicit, provider-driven (not key-presence inference):
          - ``openrouter`` → OPENROUTER_API_KEY.
          - ``direct`` → openai/anthropic native provider chosen by the
            model's vendor, keyed by OPENAI_API_KEY / ANTHROPIC_API_KEY.
        Missing the required key for the chosen provider is a hard error.
        """
        env = self.executor.env or {}

        def _key(name: str) -> str:
            return env.get(name) or os.environ.get(name, "")

        if cfg.provider == "openrouter":
            api_key = _key("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "openclaw_cli: provider=openrouter but OPENROUTER_API_KEY "
                    "is not set. Export it or pass it via executor env before "
                    "launch()."
                )
            return "openrouter", api_key

        if cfg.provider == "direct":
            provider = self._direct_provider_for_model(cfg.model)
            key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
            api_key = _key(key_var)
            if not api_key:
                raise RuntimeError(
                    f"openclaw_cli: provider=direct resolved to {provider!r} "
                    f"for model {cfg.model!r} but {key_var} is not set. Export "
                    "it or pass it via executor env before launch()."
                )
            return provider, api_key

        raise RuntimeError(
            f"openclaw_cli: unknown provider {cfg.provider!r} "
            "(expected 'openrouter' or 'direct')"
        )

    def _write_config(self, cfg: OpenClawCliConfig) -> None:
        """Write openclaw.json, auth-profiles.json, exec-approvals, workspace-state."""
        home = os.path.expanduser("~")
        oc_home = Path(home) / ".openclaw"
        oc_home.mkdir(parents=True, exist_ok=True)

        # Explicit, provider-driven routing (not key-presence inference).
        # The model ref is prefixed with the resolved provider so openclaw
        # routes the request through it: "openrouter/<model>" for the
        # OpenRouter gateway, or the native "openai/..." / "anthropic/..."
        # provider for a direct run.
        provider, api_key = self._resolve_routing(cfg)

        # --- openclaw.json ---
        primary_model = self._route_model(cfg.model, provider)
        # The resolved provider's plugin must be enabled for its auth
        # profile to load (e.g. "anthropic" is not in the default allow set).
        plugins_allow = list(cfg.plugins_allow)
        if provider not in plugins_allow:
            plugins_allow.append(provider)
        tools_also_allow = list(CUA_TOOL_NAMES)
        agent_defaults: dict = {
            "model": {"primary": primary_model},
            "timeoutSeconds": int(cfg.timeout_s),
            "models": {primary_model: {}},
        }
        # Only add heartbeat config when a valid duration is specified;
        # "never" is not a valid duration string for the openclaw CLI
        # (it expects ms, s, m, h suffixes).  Omitting the key disables it.
        if cfg.heartbeat_every and cfg.heartbeat_every.lower() != "never":
            agent_defaults["heartbeat"] = {"every": cfg.heartbeat_every}
        oc_config = {
            "agents": {
                "defaults": agent_defaults,
            },
            "plugins": {
                "allow": plugins_allow,
                "deny": list(cfg.plugins_deny),
                # Point the native CUA plugin at this image's cua-server. The
                # plugin defaults to localhost:5000 (correct on GCE); ale-kasm
                # runs cua-server on 8000, so set it explicitly from the
                # executor's bridge URL. Key path: plugins.entries.<id>.config.
                "entries": {
                    "cua": {
                        "config": {
                            "serverUrl": self.executor.cua_bridge_url(),
                        },
                    },
                },
            },
            "tools": {
                "alsoAllow": tools_also_allow,
                "deny": list(cfg.tools_deny),
                # yolo exec policy: without this, exec/shell tool calls hit
                # the device-pairing scope-upgrade flow and fail on a
                # headless sandbox with no human to approve. Matches the
                # exec-approvals.json defaults written below.
                "exec": {
                    "host": "gateway",
                    "security": "full",
                    "ask": "off",
                },
            },
            "gateway": {
                "mode": "local",
                "bind": "loopback",
            },
        }
        if cfg.vision_model:
            oc_config["tools"]["media"] = {
                "image": {"models": {"default": self._route_model(cfg.vision_model, provider)}},
            }
        (oc_home / "openclaw.json").write_text(
            json.dumps(oc_config, indent=2), encoding="utf-8",
        )

        # --- exec-approvals.json (yolo) ---
        approvals = {
            "version": 1,
            "defaults": {
                "security": "full",
                "ask": "off",
                "askFallback": "full",
            },
            "socket": {},
            "agents": {},
        }
        (oc_home / "exec-approvals.json").write_text(
            json.dumps(approvals, indent=2), encoding="utf-8",
        )

        # --- auth-profiles.json ---
        agent_dir = oc_home / "agents" / _AGENT_ID / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        auth = {
            "profiles": {
                f"{provider}:default": {
                    "provider": provider,
                    "type": "api_key",
                    "key": api_key,
                },
            },
            "lastGood": {
                provider: f"{provider}:default",
            },
        }
        (agent_dir / "auth-profiles.json").write_text(
            json.dumps(auth, indent=2), encoding="utf-8",
        )

        # --- workspace bootstrap completion (skip the interactive wizard) ---
        # OpenClaw's embedded agent reads its workspace at
        # ``~/.openclaw/workspace/`` and treats bootstrap as *pending* unless
        # ``<workspace>/.openclaw/workspace-state.json`` carries
        # ``setupCompletedAt`` AND ``<workspace>/BOOTSTRAP.md`` is gone. While
        # pending, the agent enters the interactive "who am I?" bootstrap
        # conversation (BOOTSTRAP.md) instead of the task; under ``--json`` it
        # buffers all output to the end, so on a headless VM it produces zero
        # stdout/stderr and hangs until the wall budget — observed as a silent
        # Windows runtime hang. (The runtime materializes this workspace lazily
        # on first run, so we also re-assert these markers in launch().)
        self._complete_workspace_bootstrap(oc_home)

        # --- .env ---
        env_file = oc_home / ".env"
        env_file.write_text("OPENCLAW_RAW_STREAM=0\n", encoding="utf-8")

        logger.info("openclaw_cli: config staged at %s", oc_home)

    def _complete_workspace_bootstrap(self, oc_home: Path | None = None) -> None:
        """Mark OpenClaw's workspace bootstrap complete so ``agent --local``
        skips the interactive wizard.

        Mirrors agenthle's verified Windows/Linux path. OpenClaw resolves its
        agent workspace to ``~/.openclaw/workspace/`` and considers bootstrap
        *pending* until ``<workspace>/.openclaw/workspace-state.json`` carries
        ``setupCompletedAt`` and ``<workspace>/BOOTSTRAP.md`` is removed. We
        also seed ``MEMORY.md`` + ``memory/{today,yesterday}.md`` (touch-only,
        never truncating) because the default workspace ``AGENTS.md`` tells the
        agent to read them at session start and the lazy writers that create
        them never fire in a single short benchmark run.

        Pure stdlib + filesystem (the deployer runs inside the sandbox), so a
        single implementation covers Linux and Windows. Idempotent.
        """
        if oc_home is None:
            oc_home = Path(os.path.expanduser("~")) / ".openclaw"
        workspace = oc_home / "workspace"
        ws_state_dir = workspace / ".openclaw"
        ws_state_dir.mkdir(parents=True, exist_ok=True)

        state_path = ws_state_dir / "workspace-state.json"
        state: dict = {"version": 1}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {"version": 1}
        now = datetime.now(timezone.utc).isoformat()
        state.setdefault("bootstrapSeededAt", now)
        state["setupCompletedAt"] = now
        state_path.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8",
        )

        bootstrap_md = workspace / "BOOTSTRAP.md"
        if bootstrap_md.exists():
            try:
                bootstrap_md.unlink()
            except OSError:
                pass

        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        for f in (
            workspace / "MEMORY.md",
            memory_dir / f"{today}.md",
            memory_dir / f"{yesterday}.md",
        ):
            try:
                f.touch(exist_ok=True)
            except OSError:
                pass
        logger.info(
            "openclaw_cli: workspace bootstrap marked complete at %s", workspace,
        )

    async def install(self) -> None:
        cfg: OpenClawCliConfig = self.config  # type: ignore[assignment]

        # Ensure node/npm reachable (on Windows node ships off PATH).
        from ale_run.agents._bootstrap import ensure_npm
        await ensure_npm()

        # 1. Install openclaw CLI.
        # The sandbox entry runs without the login-shell environment, so a
        # pre-baked openclaw under ~/.npm-global/bin or ~/.local/bin is NOT
        # on PATH and shutil.which would miss it — triggering a needless (and
        # historically flaky) tarball re-install. Augment PATH with the common
        # user-level npm bin dirs first so a complete pre-baked install is
        # preferred over a fresh download. (<prefix> itself for Windows shims,
        # <prefix>/bin for Linux.)
        home = os.path.expanduser("~")
        for bin_dir in (
            os.path.join(home, ".npm-global", "bin"),
            os.path.join(home, ".local", "bin"),
            os.path.join(home, ".local"),
        ):
            if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        openclaw_path = shutil.which("openclaw")
        if not openclaw_path:
            tarball = cfg.tarball_path
            if not Path(tarball).exists() and cfg.tarball_url:
                logger.info("openclaw_cli: tarball not at %s, downloading from %s",
                            tarball, cfg.tarball_url)
                tarball = await self._download_tarball(cfg.tarball_url)
            if not Path(tarball).exists():
                raise RuntimeError(
                    f"OpenClawCliDeployer: 'openclaw' not on PATH and "
                    f"tarball not found at {tarball}. Set tarball_url in config "
                    f"or bake into the sandbox image."
                )
            logger.info("openclaw_cli: installing from tarball %s", tarball)
            await self._install_from_tarball(tarball)
            openclaw_path = shutil.which("openclaw")
            if not openclaw_path:
                raise RuntimeError(
                    "OpenClawCliDeployer: 'openclaw' still not found after install"
                )
        self._openclaw_path = openclaw_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [openclaw_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"openclaw --version timed out: {e}")
        logger.info("openclaw_cli: CLI ok — %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 2. Build CUA plugin (if source available and not already installed)
        home = os.path.expanduser("~")
        plugin_entry = Path(home) / ".openclaw" / "extensions" / "cua" / "dist" / "index.cjs"
        if not plugin_entry.exists():
            plugin_src = cfg.cua_plugin_path
            if not Path(plugin_src).is_dir() and cfg.cua_plugin_repo:
                logger.info("openclaw_cli: CUA plugin not at %s, cloning from %s",
                            plugin_src, cfg.cua_plugin_repo)
                plugin_src = await self._clone_cua_plugin(
                    cfg.cua_plugin_repo, cfg.cua_plugin_branch,
                )
            if Path(plugin_src).is_dir():
                logger.info("openclaw_cli: building CUA plugin from %s", plugin_src)
                await self._build_cua_plugin(plugin_src)
            else:
                logger.warning(
                    "openclaw_cli: CUA plugin source not found at %s — "
                    "computer tools will be unavailable",
                    plugin_src,
                )
        else:
            logger.info("openclaw_cli: CUA plugin already installed")

        # 3. Write config files
        self._write_config(cfg)

        # 4. Pre-warm the bundled-plugin runtime-deps mirror.
        await self._prewarm_plugin_runtime_mirror(cfg)

    async def _prewarm_plugin_runtime_mirror(self, cfg: OpenClawCliConfig) -> None:
        """Force OpenClaw to build its bundled-plugin runtime-deps mirror now,
        during the untimed install, rather than on the first (timed) agent turn.

        When ``openclaw`` is installed to a global npm prefix it cannot write
        runtime deps in place, so on first plugin-load it mirrors the bundled
        plugins' dist tree (~60 plugins, thousands of files) into
        ``~/.openclaw/plugin-runtime-deps/.../dist/extensions/`` under a
        filesystem lock. On Windows this recursive copy takes minutes; if it
        runs inside ``agent --local`` it produces no stdout/stderr until done
        and blows the agent's wall budget — the silent runtime hang. The mirror
        is built once and reused (existing targets are skipped), so a cheap
        ``models list`` here (which loads the same provider plugins) pays the
        cost up front and leaves every subsequent ``launch()`` fast.

        Best-effort: a failure here is non-fatal — the agent turn would just
        pay (or re-pay) the mirror cost itself.
        """
        argv = self._launch_prefix() + ["models", "list"]
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        try:
            t0 = time.monotonic()
            proc = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True, text=True, env=env,
                timeout=600,
                cwd=str(Path(self.executor.work_dir)),
            )
            logger.info(
                "openclaw_cli: plugin runtime mirror pre-warmed in %.0fs (rc=%s)",
                time.monotonic() - t0, proc.returncode,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "openclaw_cli: plugin runtime mirror pre-warm timed out after 600s; "
                "the first agent turn may pay the remaining mirror cost"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "openclaw_cli: plugin runtime mirror pre-warm failed (%s); "
                "the first agent turn will build the mirror itself", exc,
            )

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: OpenClawCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # Re-assert workspace bootstrap completion: OpenClaw's runtime
        # materializes ~/.openclaw/workspace/ (with a fresh BOOTSTRAP.md) lazily
        # on first run, so the markers written during install() may have been
        # superseded before we launch the turn.
        self._complete_workspace_bootstrap()

        prompt_file = wd / "prompt.txt"
        stdout_log = wd / "stdout.log"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "openclaw.pid"

        for f in (stdout_log, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        home = os.path.expanduser("~")
        env_file = Path(home) / ".openclaw" / ".env"
        argv = self._launch_prefix() + [
            "agent", "--local",
            "--agent", _AGENT_ID,
            "--message", prompt,
            "--json",
            "--timeout", str(int(cfg.timeout_s)),
            "--thinking", cfg.thinking,
        ]
        env = self._build_env(cfg, env_file)

        t0 = time.monotonic()
        with open(stdout_log, "wb") as fout, \
             open(stderr_log, "wb") as ferr:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("openclaw_cli: spawned pid=%s", proc.pid)

        deadline = t0 + cfg.timeout_s
        while proc.poll() is None:
            if time.monotonic() > deadline:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S,
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                return AgentRunResult(
                    status="timeout",
                    pid=proc.pid,
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = _diagnose_failure(stderr_log, exit_code)

        # Copy session trajectory to work_dir for gathering
        session_id = self._extract_session_id(stderr_log)
        if session_id:
            src = Path(home) / ".openclaw" / "agents" / _AGENT_ID / "sessions" / f"{session_id}.jsonl"
            dst = wd / "transcript.jsonl"
            if src.exists():
                shutil.copy2(str(src), str(dst))
                logger.info("openclaw_cli: copied session trajectory to %s", dst)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(wd / "transcript.jsonl"),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    # =========================================================================
    # internals
    # =========================================================================

    def _launch_prefix(self) -> list[str]:
        """Argv prefix that invokes openclaw.

        On Linux the npm ``openclaw`` shim is launched directly. On Windows
        the ``openclaw.cmd`` shim mangles arg passing / loses its node lookup
        when spawned headless (see agenthle notes), so invoke
        ``node.exe <…>\\node_modules\\openclaw\\openclaw.mjs`` directly when
        the .mjs entry can be located; otherwise fall back to the shim.
        """
        if self.executor.sandbox.is_linux:
            return [self._openclaw_path]
        node = self.executor.sandbox.node
        if not node or not os.path.isfile(node):
            node = shutil.which("node") or shutil.which("node.exe") or "node"
        mjs = self._openclaw_mjs_entry()
        if mjs:
            return [node, mjs]
        return [self._openclaw_path]

    def _openclaw_mjs_entry(self) -> str | None:
        """Locate the openclaw.mjs bin entry under the global node_modules."""
        home = os.path.expanduser("~")
        candidates = [
            Path(home) / ".local" / "node_modules" / "openclaw" / "openclaw.mjs",
            Path(home) / ".local" / "lib" / "node_modules" / "openclaw" / "openclaw.mjs",
        ]
        # Derive from the resolved shim path's directory too (handles a
        # pre-baked install under a different prefix).
        shim = getattr(self, "_openclaw_path", None)
        if shim:
            shim_dir = Path(shim).parent
            candidates += [
                shim_dir / "node_modules" / "openclaw" / "openclaw.mjs",
                shim_dir / "lib" / "node_modules" / "openclaw" / "openclaw.mjs",
            ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def _build_env(self, cfg: OpenClawCliConfig, env_file: Path) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        # For a direct run, drop the OpenRouter key so its presence (always
        # exported by the secrets sidecar) cannot make openclaw fall back to
        # the openrouter provider behind the explicitly-chosen direct one.
        if cfg.provider == "direct":
            env.pop("OPENROUTER_API_KEY", None)
        # Source .env file values
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
        return env

    @staticmethod
    def _extract_session_id(stderr_log: Path) -> str | None:
        """Extract session ID from the --json stderr envelope."""
        text = _read_text_tolerant(stderr_log)
        if not text:
            return None
        json_obj = _parse_stderr_json(text)
        if json_obj:
            meta = json_obj.get("meta", {})
            agent_meta = meta.get("agentMeta", {})
            return agent_meta.get("sessionId")
        return None

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: BaseAgentConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        # 1. Parse session trajectory JSONL (if available)
        transcript_file = work_dir / "transcript.jsonl"
        if transcript_file.exists():
            raw = transcript_file.read_text(encoding="utf-8", errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cls._consume_session_event(event, builder)

        # 2. Parse stderr JSON envelope
        stderr_file = work_dir / "stderr.log"
        if stderr_file.exists():
            stderr_text = stderr_file.read_text(encoding="utf-8", errors="replace")
            json_obj = _parse_stderr_json(stderr_text)
            if json_obj:
                builder.trajectory.extra.setdefault("openclaw_cli", {})["result_envelope"] = json_obj
                meta = json_obj.get("meta", {})
                agent_meta = meta.get("agentMeta", {})
                usage = agent_meta.get("usage", {})
                if usage:
                    builder.trajectory.extra["openclaw_cli"]["usage"] = usage

        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message="openclaw-cli: no session transcript available",
                extra={"reason": "no_transcript"},
            )

        builder.trajectory.extra.setdefault("openclaw_cli", {}).update({
            "exit_code": run_result.exit_code,
        })

    @classmethod
    def _consume_session_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "message":
            cls._consume_message(event, builder)
        elif etype == "tool_result":
            cls._consume_tool_result(event, builder)

    @staticmethod
    def _consume_message(event: dict, builder: TrajectoryBuilder) -> None:
        message = event.get("message", {})
        role = message.get("role", "")
        content_blocks = message.get("content", [])
        if not isinstance(content_blocks, list):
            content_blocks = []

        usage = message.get("usage", {})
        metrics = StepMetrics(
            input_tokens=usage.get("input"),
            output_tokens=usage.get("output"),
            cache_read_tokens=usage.get("cacheRead"),
            cache_creation_tokens=usage.get("cacheWrite"),
        ) if usage else None

        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    reasoning_parts.append(block.get("thinking", ""))
                elif btype in ("toolCall", "tool_use"):
                    name = block.get("name", "")
                    args = block.get("arguments") or block.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append(ToolCall(
                        id=block.get("id", ""),
                        name=name,
                        arguments=args,
                    ))

            builder.add_step(
                source="agent",
                message="\n".join(p for p in text_parts if p) or None,
                reasoning="\n".join(p for p in reasoning_parts if p) or None,
                tool_calls=tool_calls,
                metrics=metrics,
            )
        elif role == "user":
            text_parts = []
            results: list[ToolResult] = []
            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    content = block.get("content")
                    parts: list[ContentPart] = []
                    if isinstance(content, str):
                        parts.append(ContentPart(type="text", text=content))
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                parts.append(ContentPart(type="text", text=c.get("text", "")))
                    results.append(ToolResult(
                        tool_call_id=block.get("tool_use_id") or block.get("call_id", ""),
                        content=parts,
                        is_error=bool(block.get("is_error")),
                    ))
            if results:
                builder.add_step(
                    source="environment",
                    observation=Observation(results=results),
                )
            elif text_parts:
                builder.add_step(
                    source="user",
                    message="\n".join(p for p in text_parts if p),
                )
        elif role == "toolResult":
            output = event.get("output") or message.get("output", "")
            call_id = event.get("tool_use_id") or event.get("call_id", "")
            builder.add_step(
                source="environment",
                observation=Observation(results=[
                    ToolResult(
                        tool_call_id=call_id,
                        content=[ContentPart(type="text", text=str(output))],
                        is_error=bool(event.get("is_error")),
                    ),
                ]),
            )

    @staticmethod
    def _consume_tool_result(event: dict, builder: TrajectoryBuilder) -> None:
        output = event.get("output", "")
        call_id = event.get("tool_use_id") or event.get("call_id", "")
        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    tool_call_id=call_id,
                    content=[ContentPart(type="text", text=str(output))],
                    is_error=bool(event.get("is_error")),
                ),
            ]),
        )


def _parse_stderr_json(stderr: str) -> dict | None:
    """Find JSON envelope in openclaw --json stderr stream."""
    if not stderr:
        return None

    lines = stderr.splitlines()
    # Strategy 1: line-based — first line that is just "{"
    for i, line in enumerate(lines):
        if line.strip() == "{":
            payload = "\n".join(lines[i:])
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                break

    # Strategy 2: backward brace-balance from last "}"
    text = stderr.rstrip()
    if text.endswith("}"):
        end = len(text) - 1
        depth = 1
        in_str = False
        for j in range(end - 1, -1, -1):
            ch = text[j]
            if in_str:
                if ch == '"' and (j == 0 or text[j - 1] != "\\"):
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[j:end + 1])
                    except json.JSONDecodeError:
                        return None
    return None


_DIAG_SIGNAL_PATTERNS = (
    "FailoverError",
    "No API key found",
    "lane task error",
    "model fallback decision",
    "Error:",
)


def _diagnose_failure(stderr_log: Path, exit_code: int | None) -> str:
    parts = [f"agent failed (rc={exit_code})"]
    text = _read_text_tolerant(stderr_log)
    if text.strip():
        # The model-catalog ESM warning (ERR_MODULE_NOT_FOUND) is emitted
        # transiently while bundled plugins stage their runtime deps and is
        # caught/non-fatal — it tends to occupy the literal tail and mask the
        # real cause. Surface the highest-signal diagnostic lines first.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        signal = [
            ln for ln in lines
            if any(pat in ln for pat in _DIAG_SIGNAL_PATTERNS)
            and "ERR_MODULE_NOT_FOUND" not in ln
        ]
        if signal:
            parts.append("stderr signals: " + " || ".join(signal[-5:]))
        parts.append(f"stderr tail: ...{text[-1200:]}")
    return " | ".join(parts)


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
