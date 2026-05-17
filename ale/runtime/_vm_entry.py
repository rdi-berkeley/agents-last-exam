"""VM-side bootstrap shipped to the cua-server's Python via ``python_exec``.

VmExecutor calls ``session.python_exec(run_deployer_in_vm, spec_dict)``.
``inspect.getsource`` reads THIS function's source from this file and ships
it to the VM. Inside the VM, the function:

  1. Sets ``sys.path`` so freshly-scp'd ``/home/user/.ale-src/`` is importable
  2. Imports the deployer class + config class by dotted path
  3. Constructs ``VmRuntime`` + the deployer
  4. Awaits ``install()`` + ``launch(prompt)``
  5. Serializes :class:`AgentRunResult` to a JSON-friendly dict and returns

``python_exec`` JSON-roundtrips args + return value, so all kwargs in
``spec_dict`` must be JSON-serializable scalars / dicts / lists.

Long-run note: this call blocks while ``launch`` polls (potentially
minutes). cua's RPC has its own timeout; for v1 we accept that and
defer detached-execution + separate-polling to a future iteration.
"""
from __future__ import annotations

from typing import Any


def run_deployer_in_vm(spec: dict) -> dict:  # noqa: D401 — JSON in / JSON out
    """Construct + run a deployer inside the VM. Returns AgentRunResult fields.

    ``spec`` keys (all JSON-friendly):
      - ``ale_src_root``        e.g. "/home/user/.ale-src"
      - ``deployer_module``     e.g. "ale.agents.claude_code.deployer"
      - ``deployer_class``      e.g. "ClaudeCodeDeployer"
      - ``config_module``       e.g. "ale.agents.claude_code.config"
      - ``config_class``        e.g. "ClaudeCodeConfig"
      - ``config_kwargs``       dict — passed to config class as **kwargs
      - ``vm_runtime_kwargs``   dict — extra VmRuntime fields (overrides defaults)
      - ``work_dir``            VM path string
      - ``vm_endpoint``         e.g. "http://34.94.212.100:5000"
      - ``vm_os``               "linux" | "windows"
      - ``prompt``              str
    """
    # ⚠ ALL imports below use importlib.import_module — never `from X import Y`
    # in this function body. cua's python_exec source-generator
    # (computer/helpers.generate_source_code) lifts ``import``-statements out
    # of the function body to the top of the generated script, which would
    # try to import ``ale`` BEFORE the sys.path.insert below runs.
    import asyncio
    import importlib
    import sys
    import traceback

    Path = importlib.import_module("pathlib").Path

    # 1. Make freshly-scp'd ale source importable
    src = spec["ale_src_root"]
    if src not in sys.path:
        sys.path.insert(0, src)

    try:
        # 2. Import deployer + config + VmRuntime (NOW that sys.path is set)
        dep_mod = importlib.import_module(spec["deployer_module"])
        cfg_mod = importlib.import_module(spec["config_module"])
        rt_mod = importlib.import_module("ale.runtime.vm")
        dep_cls = getattr(dep_mod, spec["deployer_class"])
        cfg_cls = getattr(cfg_mod, spec["config_class"])
        VmRuntime = rt_mod.VmRuntime

        # 3. Build config + runtime + deployer
        cfg = cfg_cls(**spec["config_kwargs"])
        runtime_kwargs = {
            "work_dir": Path(spec["work_dir"]),
            "vm_endpoint": spec["vm_endpoint"],
            "vm_os": spec["vm_os"],
            "config": cfg,
            **spec.get("vm_runtime_kwargs", {}),
        }
        runtime = VmRuntime(**runtime_kwargs)
        deployer = dep_cls(runtime)

        # 4. Run install + launch synchronously
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(deployer.install())
            result = loop.run_until_complete(deployer.launch(spec["prompt"]))
        finally:
            loop.close()

        # 5. Serialize AgentRunResult
        return {
            "ok": True,
            "status": result.status,
            "error": result.error,
            "transcript_path": result.transcript_path,
            "stderr_path": result.stderr_path,
            "pid": result.pid,
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "failed",
            "error": f"vm_entry: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
