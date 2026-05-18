"""VmExecutor — ships deployer code to the eval VM + runs it there.

Two-step bootstrap each unit:

  1. **scp ale subtree** to ``/home/user/.ale-src/`` on the VM
     (idempotent; checks file size before write to skip unchanged files).
     Includes only files the deployer + ALE base need to import:

         ale/__init__.py
         ale/runtime/{base,executor,local,local_executor,vm,_vm_entry}.py
         ale/runtime/__init__.py
         ale/agents/__init__.py
         ale/agents/base.py
         ale/agents/trajectory.py
         ale/agents/<chosen_agent>/   (config.py + deployer.py + __init__.py)

  2. **python_exec** :func:`ale.runtime._vm_entry.run_deployer_in_vm`
     against a freshly-built ``cb.DesktopSession``. The function (shipped
     via inspect.getsource) does the construct + lifecycle inside VM and
     returns an :class:`AgentRunResult`-shaped dict.

Gather pulls the VM work_dir back to host via session walking.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ale.runtime._vm_entry import run_deployer_in_vm

from ._env import collect_host_env
from .executor import EXECUTORS, Executor

if TYPE_CHECKING:
    from ale.agents.base import AgentRunResult, BaseAgentDeployer

    from .base import AgentRuntime
    from .vm import VmRuntime

logger = logging.getLogger(__name__)


# Where the scp'd ale subtree lands on the VM
VM_ALE_SRC_ROOT = "/home/user/.ale-src"

# Subset of the ale tree shipped to VM — only what the deployer needs to
# import inside the bootstrap. Missing files are skipped gracefully
# (e.g. agents/__init__.py is PEP 420 namespace — no file on disk).
#
# NOTE: we do NOT ship the host's ``ale/__init__.py`` or
# ``ale/runtime/__init__.py`` because they import heavy stuff (registry,
# vm_executor, docker_executor) that the VM doesn't need + can't import.
# Stripped versions are written by :func:`_ship_ale_subtree`.
_ALE_ROOT_REL_PATHS = [
    "runtime/base.py",
    "runtime/executor.py",
    "runtime/local.py",
    "runtime/local_executor.py",
    "runtime/vm.py",
    "runtime/_vm_entry.py",
    "agents/base.py",
    "agents/trajectory.py",
]


class VmExecutor(Executor):
    """In-VM executor: scp + python_exec bootstrap."""

    kind = "vm"

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        runtime: "AgentRuntime",
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        from ale.agents.base import AgentRunResult

        if runtime.kind != "vm":
            raise TypeError(
                f"VmExecutor.run_deployer needs VmRuntime, got {type(runtime).__name__}"
            )
        vm_runtime: VmRuntime = runtime          # type: ignore[assignment]

        # Build a session for shipping code + invoking python_exec
        session = await vm_runtime.make_vm_session()

        # 1. scp ale + deployer-specific subtree to VM
        agent_name = vm_runtime.config.name
        agent_subdir = _deployer_module_subdir(deployer_cls)
        await _ship_ale_subtree(session, agent_subdir=agent_subdir)
        logger.info("vm: shipped ale subtree to %s on VM", VM_ALE_SRC_ROOT)

        # 2. Make sure VM work_dir exists
        wd = str(vm_runtime.work_dir)
        await session.run_command(f"mkdir -p {wd}")

        # 3. Build spec for the bootstrap (JSON-friendly)
        # host_env propagates the operator's API keys / routing config from
        # host shell → VM's Python os.environ. The bash subprocess that
        # spawns the agent CLI inherits from there.
        spec = {
            "ale_src_root": VM_ALE_SRC_ROOT,
            "deployer_module": deployer_cls.__module__,
            "deployer_class": deployer_cls.__name__,
            "config_module": vm_runtime.config.__class__.__module__,
            "config_class": vm_runtime.config.__class__.__name__,
            "config_kwargs": _config_to_kwargs(vm_runtime.config),
            "work_dir": wd,
            "vm_endpoint": vm_runtime.vm_endpoint,
            "vm_os": vm_runtime.vm_os,
            "vm_runtime_kwargs": _vm_runtime_path_kwargs(vm_runtime),
            "host_env": collect_host_env(),
            "prompt": prompt,
        }

        # 4. Ship + run the bootstrap. python_exec uses inspect.getsource on
        #    `run_deployer_in_vm`, so it must be a module-level function in
        #    a real .py file — which it is (_vm_entry.py).
        logger.info(
            "vm: python_exec run_deployer_in_vm (deployer=%s, work_dir=%s)",
            deployer_cls.__name__, wd,
        )
        result_dict = await session.computer.python_exec(run_deployer_in_vm, spec)

        if not result_dict.get("ok", False):
            logger.error(
                "vm: bootstrap failed — %s\n%s",
                result_dict.get("error"),
                result_dict.get("traceback", "")[:1000],
            )

        return AgentRunResult(
            status=result_dict.get("status", "failed"),
            error=result_dict.get("error"),
            transcript_path=result_dict.get("transcript_path"),
            stderr_path=result_dict.get("stderr_path"),
            pid=result_dict.get("pid"),
            exit_code=result_dict.get("exit_code"),
            duration_s=result_dict.get("duration_s"),
        )

# =============================================================================
# helpers
# =============================================================================

def _deployer_module_subdir(deployer_cls: type) -> str:
    """``ale.agents.claude_code.deployer`` → ``agents/claude_code``."""
    mod = deployer_cls.__module__
    parts = mod.split(".")
    # mod like ['ale', 'agents', 'claude_code', 'deployer']
    if parts[:2] == ["ale", "agents"] and len(parts) >= 3:
        return f"agents/{parts[2]}"
    raise ValueError(f"can't derive subdir for {mod}; expected ale.agents.<x>.*")


async def _ship_ale_subtree(session, *, agent_subdir: str) -> None:
    """scp ALE source files to ``<VM_ALE_SRC_ROOT>/ale/...`` on the VM.

    sys.path on the VM gets prepended with ``VM_ALE_SRC_ROOT`` so
    ``import ale.agents.<x>...`` resolves. The ``ale/`` prefix in the
    VM path matches the host's ale package name.

    Idempotent: skips write when remote size+bytes match.
    """
    from ale.runtime import vm_executor as _self_mod
    repo_root = Path(_self_mod.__file__).resolve().parents[2]   # .../agents-last-exam
    ale_root = repo_root / "ale"

    # Universal subset (relative to ale/)
    files: list[Path] = [ale_root / rel for rel in _ALE_ROOT_REL_PATHS]
    # Per-agent subtree (config.py, deployer.py, __init__.py, etc.)
    agent_root = ale_root / agent_subdir
    if agent_root.is_dir():
        for p in sorted(agent_root.rglob("*.py")):
            files.append(p)

    await session.run_command(f"mkdir -p {VM_ALE_SRC_ROOT}/ale/runtime")
    # Write STRIPPED ale/__init__.py + ale/runtime/__init__.py — VM only
    # needs the package namespaces, not the host's heavy import chains
    # (registry, vm_executor, docker_executor).
    await session.write_bytes(
        f"{VM_ALE_SRC_ROOT}/ale/__init__.py",
        b"# vm-stripped ale package (no registry imports)\n",
    )
    await session.write_bytes(
        f"{VM_ALE_SRC_ROOT}/ale/runtime/__init__.py",
        b"# vm-stripped ale.runtime package (no vm_executor/docker_executor imports)\n",
    )
    for src_path in files:
        if not src_path.is_file():
            continue
        # rel relative to *repo_root*, so vm_path becomes
        # <VM_ALE_SRC_ROOT>/ale/agents/<x>/deployer.py
        rel = src_path.relative_to(repo_root)
        vm_path = f"{VM_ALE_SRC_ROOT}/{rel.as_posix()}"
        data = src_path.read_bytes()
        try:
            existing = await session.read_bytes(vm_path)
            if existing == data:
                continue
        except Exception:                                       # noqa: BLE001
            pass
        parent = vm_path.rsplit("/", 1)[0]
        await session.run_command(f"mkdir -p {parent}")
        await session.write_bytes(vm_path, data)


def _config_to_kwargs(cfg) -> dict:
    """Serialize a config dataclass into kwargs for reconstruction in VM.

    Skip ``ClassVar`` (e.g. ``name``) and any non-JSON-serializable fields.
    """
    import dataclasses
    out = {}
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        # Only ship JSON-friendly scalars / lists / dicts / None
        if isinstance(val, (str, int, float, bool, type(None), list, dict, tuple)):
            out[f.name] = val
    return out


def _vm_runtime_path_kwargs(rt: "VmRuntime") -> dict:
    """Pull the path-convention fields from VmRuntime so the in-VM
    bootstrap can reconstruct an identical VmRuntime."""
    return {
        "node_exe": rt.node_exe,
        "user_home": rt.user_home,
        "mcp_server_dir": rt.mcp_server_dir,
        "agent_bin_dir": rt.agent_bin_dir,
        "work_dir_root": rt.work_dir_root,
        "python_exe": rt.python_exe,
    }


# Register at import time
EXECUTORS["vm"] = VmExecutor()
