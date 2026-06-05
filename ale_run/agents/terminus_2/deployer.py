"""Terminus2Deployer — drives the harbor terminus_2 agent (cua-verse fork).

terminus_2 is harbor's tmux-driven, ReAct-style agent: each turn the LLM
emits ``{analysis, plan, commands[]}`` JSON; the agent feeds keystrokes
into a tmux pane on the machine it runs on and feeds the resulting pane
output back into the next turn. It terminates on a double-confirmed
``task_complete`` signal or on the outer timeout.

ALE runs it from the ``cua-verse/harbor`` fork on branch ``agenthle``
which ships a thin ``harbor-terminus2`` CLI shim plus a
``LocalShellEnvironment`` so the agent's tmux loop drives the same
sandbox it runs inside. There is **no Docker layer and no CUA MCP
bridge** — the single conceptual action is ``tmux send-keys``.

Install: ``uv tool install`` the fork (no submodule, no pre-baked image
requirement). Pre-installs tmux + asciinema via apt if absent.

Launch: ``harbor-terminus2 --prompt-file <f> --model <m> --logs-dir <d>
--temperature <t> [--max-turns N] [--api-base U] [--no-recording]``.
The selected provider's API key is injected into the launched process's
environment ONLY (never written to any gathered file).

Transcript: terminus_2 emits an ATIF ``trajectory.json`` under
``<logs-dir>/agent/``; ``parse_artifacts`` converts it to ALE steps.

Linux-only: terminus_2's TmuxSession requires tmux + asciinema and a
POSIX environment. Windows sandboxes are explicitly unsupported.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

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

from .config import HARBOR_FORK_REF, HARBOR_FORK_URL, Terminus2Config

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 10.0
_TERM_GRACE_S = 3.0


class Terminus2Deployer(BaseAgentDeployer):
    """Stdlib-only deployer for the harbor terminus_2 agent (cua-verse fork)."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("stdout.log",)

    @property
    def version(self) -> str | None:
        return None  # discovered at install time

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        """Install the harbor fork's ``harbor-terminus2`` CLI + ensure tmux.

        Runs inside the sandbox via ``_sandbox_entry`` -- like
        :meth:`launch`, all filesystem/process ops use local I/O and
        ``subprocess`` rather than the cua bridge.
        """
        sandbox = self.executor.sandbox
        if not sandbox.is_linux:
            raise NotImplementedError("terminus_2 is Linux-only")

        home = os.path.expanduser("~")
        local_bin = f"{home}/.local/bin"
        terminus_bin = f"{local_bin}/harbor-terminus2"

        async def _sh(cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
            return await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )

        # 1. tmux + asciinema are required by terminus_2's TmuxSession. Try to
        #    pre-install (best-effort: apt may need sudo / network). A hard
        #    failure here surfaces clearly rather than mid-run.
        have_tmux = shutil.which("tmux") is not None
        have_asciinema = shutil.which("asciinema") is not None
        if not (have_tmux and have_asciinema):
            logger.info("terminus_2: installing tmux/asciinema via apt ...")
            await _sh(
                "sudo apt-get update -y >/dev/null 2>&1 || apt-get update -y >/dev/null 2>&1; "
                "sudo apt-get install -y tmux asciinema git >/dev/null 2>&1 || "
                "apt-get install -y tmux asciinema git >/dev/null 2>&1 || true",
                timeout=300,
            )
        if shutil.which("tmux") is None:
            raise RuntimeError(
                "terminus_2: tmux not available and could not be installed "
                "(needed for the agent's TmuxSession)."
            )

        # 2. Install the harbor-terminus2 CLI from the fork (skip if present).
        already = os.path.isfile(terminus_bin) and os.access(terminus_bin, os.X_OK)
        if not already:
            # Ensure uv is available (uv tool install gives an isolated venv +
            # a `harbor-terminus2` console script on ~/.local/bin).
            if not shutil.which("uv"):
                logger.info("terminus_2: bootstrapping uv ...")
                await _sh("curl -LsSf https://astral.sh/uv/install.sh | sh", timeout=180)
                if local_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
            uv = shutil.which("uv") or f"{local_bin}/uv"

            logger.info(
                "terminus_2: uv tool install harbor (fork=%s@%s) ...",
                HARBOR_FORK_URL, HARBOR_FORK_REF,
            )
            install = await _sh(
                f"export PATH=\"{local_bin}:$PATH\" && "
                f"'{uv}' tool install --python 3.12 --reinstall "
                f"--from 'git+{HARBOR_FORK_URL}@{HARBOR_FORK_REF}' harbor 2>&1",
                timeout=900,
            )
            if install.returncode != 0:
                combined = ((install.stdout or "") + (install.stderr or "")).strip()
                raise RuntimeError(
                    f"terminus_2: harbor install failed (rc={install.returncode}): "
                    f"...{combined[-1500:]}"
                )

        # 3. Verify the CLI is on PATH.
        verify = await _sh(
            f"export PATH=\"{local_bin}:$PATH\" && "
            f"'{terminus_bin}' --version 2>&1 || '{terminus_bin}' --help 2>&1 | head -3",
            timeout=60,
        )
        out = (verify.stdout or "").strip()
        logger.info("terminus_2: CLI check -- %s", out[:200])
        if not os.path.isfile(terminus_bin):
            raise RuntimeError(
                f"terminus_2: '{terminus_bin}' not found after install. "
                f"check: {out[:500]}"
            )

        # 4. Working dirs.
        wd = self.executor.work_dir
        logs_dir = f"{wd}/logs"
        for d in (wd, logs_dir, f"{logs_dir}/agent"):
            os.makedirs(d, exist_ok=True)

        logger.info("terminus_2: install complete")

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        """Spawn harbor-terminus2 and poll until completion or timeout.

        Runs inside the sandbox via ``_sandbox_entry``. The selected
        provider's API key is placed in the child process's environment
        only -- never written to a file under ``work_dir`` (which the
        framework gathers to the host).
        """
        cfg: Terminus2Config = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        terminus_bin = f"{home}/.local/bin/harbor-terminus2"
        logs_dir = wd / "logs"
        agent_dir = logs_dir / "agent"

        prompt_file = wd / "prompt.txt"
        stdout_log = wd / "stdout.log"
        exit_file = wd / "exit_code.txt"

        # Clean previous-run state so a resumed run unit starts fresh.
        for f in (stdout_log, exit_file):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        if agent_dir.exists():
            shutil.rmtree(agent_dir, ignore_errors=True)
        for d in (logs_dir, agent_dir):
            d.mkdir(parents=True, exist_ok=True)

        # The harbor fork's TmuxSession hardcodes the session name
        # ``terminus-2``; a session left alive by a prior/aborted run (or the
        # image-build-time install probe) makes the next launch die with
        # "duplicate session: terminus-2". Kill any stale session before
        # spawning so each run starts from a clean tmux server.
        await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             "tmux kill-session -t terminus-2 2>/dev/null || true"],
            capture_output=True, timeout=10,
        )

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(cfg, terminus_bin, str(prompt_file), str(logs_dir))
        env = self._build_env(cfg)

        logger.info(
            "terminus_2: launching (model=%s, provider=%s)",
            cfg.litellm_model_id, cfg.provider,
        )

        t0 = time.monotonic()
        with open(stdout_log, "wb") as sout:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=sout,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        logger.info("terminus_2: spawned pid=%s", proc.pid)

        # The episode wall budget is orchestration-owned: the executor wraps
        # launch() in asyncio.wait_for(timeout=timeout_s) (derived from the
        # task), so we just wait for the child here. If that budget fires we
        # are cancelled mid-await; kill the process group (tmux server +
        # children) before propagating so it cannot outlive the run.
        try:
            while proc.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            self._kill_process(proc)
            raise

        # Give the CLI a beat to flush the final trajectory.json + recording.
        await asyncio.sleep(_TERM_GRACE_S)
        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        try:
            exit_file.write_text(str(exit_code), encoding="ascii")
        except OSError:
            pass

        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = self._diagnose_failure(str(stdout_log), exit_code)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(agent_dir / "trajectory.json"),
            stderr_path=str(stdout_log),
            duration_s=duration_s,
            error=error,
        )

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> None:
        # Kill the whole process group so terminus_2's tmux server / child
        # shells die with it.
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), 15)
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), 9)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass
        # tmux server may outlive the process group; best-effort cleanup.
        try:
            subprocess.run(
                ["bash", "-c", "tmux kill-server 2>/dev/null || true"],
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _build_argv(
        self, cfg: Terminus2Config, terminus_bin: str, prompt_file: str, logs_dir: str,
    ) -> list[str]:
        argv = [
            terminus_bin,
            "--prompt-file", prompt_file,
            "--model", cfg.litellm_model_id,
            "--logs-dir", logs_dir,
            "--temperature", str(cfg.temperature),
        ]
        if cfg.api_base:
            argv += ["--api-base", cfg.api_base]
        if cfg.max_turns is not None:
            # -1 (or any value < 0) = unlimited; terminus_2 has no native
            # unlimited sentinel, so translate to a large finite cap.
            n = 100_000 if cfg.max_turns < 0 else cfg.max_turns
            argv += ["--max-turns", str(n)]
        if not cfg.record_terminal_session:
            argv.append("--no-recording")
        return argv

    def _build_env(self, cfg: Terminus2Config) -> dict[str, str]:
        """Build the child process env: PATH + exactly the provider key.

        Keys live in the process env only; nothing here is written to a
        gathered file.
        """
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v

        home = os.path.expanduser("~")
        local_bin = f"{home}/.local/bin"
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
        env["NO_COLOR"] = "1"

        env_name, key_value = self._selected_api_key(cfg, env)
        if not key_value:
            raise RuntimeError(
                f"terminus_2: required API key {env_name} is empty "
                f"(provider={cfg.provider}, model={cfg.model}). Export it or "
                "pass it via executor env before launch()."
            )
        env[env_name] = key_value
        if cfg.provider == "openrouter":
            env.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return env

    @staticmethod
    def _selected_api_key(cfg: Terminus2Config, env: dict[str, str]) -> tuple[str, str]:
        """Return ``(env_var_name, value)`` for the provider in use.

        openrouter -> OPENROUTER_API_KEY. direct -> inferred from the model
        prefix (``anthropic/...`` -> ANTHROPIC_API_KEY, ``openai/...`` /
        ``gpt...`` -> OPENAI_API_KEY).
        """
        if cfg.provider == "openrouter":
            return "OPENROUTER_API_KEY", env.get("OPENROUTER_API_KEY", "")
        if cfg.provider != "direct":
            raise RuntimeError(
                f"terminus_2: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'direct')"
            )
        prefix = cfg.model.split("/", 1)[0].lower()
        if prefix == "anthropic":
            return "ANTHROPIC_API_KEY", env.get("ANTHROPIC_API_KEY", "")
        if prefix == "openai" or prefix.startswith("gpt"):
            return "OPENAI_API_KEY", env.get("OPENAI_API_KEY", "")
        # Fallback: pick whichever direct key the caller populated.
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            if env.get(name):
                return name, env[name]
        return "ANTHROPIC_API_KEY", env.get("ANTHROPIC_API_KEY", "")

    @staticmethod
    def _diagnose_failure(stdout_log: str, exit_code: int | None) -> str:
        parts = [f"terminus_2 failed (rc={exit_code})"]
        try:
            text = Path(stdout_log).read_text(encoding="utf-8", errors="replace")
            if text.strip():
                parts.append(f"stdout tail: ...{text.strip()[-1000:]}")
        except (FileNotFoundError, OSError):
            pass
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: Terminus2Config,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse terminus_2's ATIF ``trajectory.json`` into trajectory steps."""
        trajectory_path = work_dir / "logs" / "agent" / "trajectory.json"
        if not trajectory_path.exists():
            builder.add_step(
                source="system",
                message=f"terminus_2: no trajectory at {trajectory_path}",
                extra={"reason": "no_trajectory"},
            )
            return

        try:
            trajectory = json.loads(
                trajectory_path.read_text(encoding="utf-8", errors="replace")
            )
        except (OSError, json.JSONDecodeError) as exc:
            builder.add_step(
                source="system",
                message=f"terminus_2: trajectory.json malformed: {exc}",
                extra={"reason": "parse_error"},
            )
            return

        for step in trajectory.get("steps", []) or []:
            cls._consume_step(step, builder)

        usage = cls._extract_usage(trajectory)
        builder.trajectory.extra.setdefault("terminus_2", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(trajectory_path),
            "usage": usage,
        })

    @classmethod
    def _consume_step(cls, step: dict, builder: TrajectoryBuilder) -> None:
        """Convert one ATIF step into ALE trajectory step(s)."""
        source = step.get("source")
        message = step.get("message")
        tool_calls = step.get("tool_calls") or []
        observation = step.get("observation") or {}
        reasoning = step.get("reasoning_content")
        metrics = step.get("metrics") or {}

        if source == "user":
            builder.add_step(source="user", message=_flatten_message(message))
            return

        if source == "system":
            builder.add_step(source="system", message=_flatten_message(message))
            return

        if source == "agent":
            if reasoning:
                builder.add_step(source="agent", reasoning=str(reasoning))
            tc_list: list[ToolCall] = []
            for tc in tool_calls:
                tc_list.append(ToolCall(
                    id=tc.get("tool_call_id") or "",
                    name=tc.get("function_name") or "",
                    arguments=cls._decode_args(tc.get("arguments")),
                ))
            msg_text = _flatten_message(message)
            if msg_text or tc_list:
                builder.add_step(
                    source="agent",
                    message=msg_text or None,
                    tool_calls=tc_list or None,
                    metrics=cls._step_metrics(metrics),
                )
            results = observation.get("results", []) or []
            if results:
                builder.add_step(
                    source="environment",
                    observation=Observation(results=[
                        ToolResult(
                            tool_call_id=r.get("source_call_id") or "",
                            content=[ContentPart(
                                type="text",
                                text=_flatten_message(r.get("content")),
                            )],
                            is_error=bool(r.get("is_error", False)),
                        )
                        for r in results
                    ]),
                )
            return

        # Unknown source -- keep as a system note so nothing is silently lost.
        builder.add_step(
            source="system",
            message=_flatten_message(message),
            extra={"original_source": source},
        )

    @staticmethod
    def _step_metrics(metrics: dict) -> StepMetrics | None:
        if not metrics:
            return None
        prompt = metrics.get("prompt_tokens")
        cached = metrics.get("cached_tokens")
        completion = metrics.get("completion_tokens")
        cost = metrics.get("cost_usd")
        if prompt is None and completion is None and cost is None:
            return None
        uncached = None
        if prompt is not None:
            uncached = max(prompt - (cached or 0), 0)
        return StepMetrics(
            input_tokens=uncached,
            output_tokens=completion,
            cache_read_tokens=cached,
            cost_usd=cost,
        )

    @staticmethod
    def _decode_args(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw}
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        return {}

    @staticmethod
    def _extract_usage(trajectory: dict) -> dict:
        """Aggregate token + cost usage, preferring ``final_metrics``."""
        final = trajectory.get("final_metrics") or {}
        total_prompt = final.get("total_prompt_tokens")
        total_completion = final.get("total_completion_tokens")
        total_cached = final.get("total_cached_tokens")
        total_cost = final.get("total_cost_usd")

        if total_prompt is not None and total_completion is not None:
            uncached = max(total_prompt - (total_cached or 0), 0)
            usage = {
                "uncached_input_tokens": uncached,
                "cache_read_input_tokens": total_cached or 0,
                "output_tokens": total_completion,
                "overall_input_tokens": total_prompt,
            }
            if total_cost is not None:
                usage["total_cost_usd"] = total_cost
            return usage

        # Fallback: sum per-step metrics.
        uncached = cache_read = output = 0
        cost_total = 0.0
        cost_seen = False
        for step in trajectory.get("steps", []) or []:
            m = step.get("metrics") or {}
            prompt = m.get("prompt_tokens") or 0
            cached = m.get("cached_tokens") or 0
            uncached += max(prompt - cached, 0)
            cache_read += cached
            output += m.get("completion_tokens") or 0
            c = m.get("cost_usd")
            if c is not None:
                cost_total += c
                cost_seen = True
        usage = {
            "uncached_input_tokens": uncached,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output,
            "overall_input_tokens": uncached + cache_read,
        }
        if cost_seen:
            usage["total_cost_usd"] = cost_total
        return usage


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _flatten_message(value: Any) -> str:
    """Flatten an ATIF message/content (str | list[ContentPart] | None)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                kind = block.get("type")
                if kind == "text":
                    parts.append(block.get("text", ""))
                elif kind == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(block)[:200])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(value)
