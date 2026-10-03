"""Linux task: formalize & prove IMO 2005 Problem 6 in Lean 4 / Mathlib.

The agent is given ``Problem.lean`` — the exact theorem statement (with fixed
auxiliary definitions) and a `sorry` — and must produce ``solution.lean`` that
replaces the `sorry` with a complete Lean 4 / Mathlib proof.

Grading is a deterministic, layered anti-cheat gate (see ``scripts/grade_lean.py``):

  G1  SIGNATURE PIN  — the submission keeps the exact `imo2005_p6` signature and
                       the `solvedCount` / `numSolvedExactly` definitions, so a
                       weaker or vacuous restatement cannot pass.
  G2  SOURCE SCAN    — reject `sorry` / `admit` / `native_decide` / new `axiom`s.
  G3  COMPILE        — `lean solution.lean` must exit 0 with no `error:`.
  G4  AXIOM AUDIT    — `#print axioms IMO2005P6.imo2005_p6` must depend only on
                       {propext, Classical.choice, Quot.sound}; this rejects
                       `sorryAx`, solver axioms, and `Lean.ofReduceBool`.

Score is 1.0 iff all gates pass (a genuine, complete, axiom-clean proof of the
exact theorem), else 0.0.

Toolchain: fetched from official upstream at setup — no data-bucket upload.
``start`` installs Lean 4.27.0 via ``elan`` and pulls the prebuilt Mathlib v4.27
``.olean`` cache via ``lake exe cache get`` (the sandbox has outbound network),
then writes ``Problem.lean`` plus ``LEANPATH``/``LEANBIN`` pointer files;
``evaluate`` compiles the agent's ``solution.lean`` read-only against that cache.

Modeled on computing_math/regex_synth_v1 (self-contained config, in-VM grading,
layered reward-hack defense), adapted from differential testing to Lean proof
verification via axiom audit.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import guard
    from grade_lean import grade as grade_lean_async
except Exception:  # pragma: no cover
    grade_lean_async = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DOMAIN_NAME = "computing_math"
TASK_NAME = "imo2005p6_lean_v1"
VARIANT_NAME = "base"

# Local (HOST) path to the problem statement staged into the VM.
_PROBLEM_LEAN = Path(__file__).resolve().parent / "assets" / "Problem.lean"

# The Lean 4.27.0 toolchain + Mathlib v4.27 .olean cache are fetched from OFFICIAL
# UPSTREAM at setup time (the sandbox has outbound network): `elan` installs Lean
# from releases.lean-lang.org, and `lake exe cache get` downloads the prebuilt
# Mathlib cache from Mathlib's official Azure cache — no hosting, no data-bucket
# upload. `start()` sets up a pinned Lake project at LEAN_PROJECT and populates
# .lake/packages/<pkg>/.lake/build/lib/lean; the olean search path is the
# colon-join of those dirs (auto-discovered so the exact dependency set is robust
# to Mathlib's manifest changing between versions).
LEAN_VERSION = "leanprover/lean4:v4.27.0"
MATHLIB_REV = "v4.27.0"
_ELAN_HOME = "/home/kasm-user/.elan"
_LEAN_BIN = f"{_ELAN_HOME}/bin/lean"
_LEAN_PROJECT = "/home/kasm-user/.ale_lean/mlproj"
_LEAN_PKGS = f"{_LEAN_PROJECT}/.lake/packages"

_COMPILE_TIMEOUT_S = 900  # generous: first Mathlib import + a large proof file.


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"
    # Self-contained: no task-data staging. start() fetches the toolchain from
    # upstream and writes the problem file; grading compiles in-sandbox.
    REQUIRES_TASK_DATA: bool = False

    @property
    def problem_file(self) -> str:
        return f"{self.input_dir}/Problem.lean"

    @property
    def lean_bin(self) -> str:
        return _LEAN_BIN

    @property
    def solution_path(self) -> str:
        return f"{self.remote_output_dir}/solution.lean"

    @property
    def task_description(self) -> str:
        return f"""You are a formal-methods engineer. Your job is to produce a
machine-checked Lean 4 (with Mathlib) proof of a specified theorem — the kind of
verification artifact used to certify that a mathematical claim holds with zero
trust in informal argument.

## The statement to prove

The exact theorem, with its fixed auxiliary definitions and a `sorry`
placeholder, is staged at:

```text
{self.problem_file}
```

Read it in full. It formalizes: *in a competition with 6 problems where every
pair of problems was solved by more than 2/5 of the contestants and nobody
solved all 6, at least two contestants each solved exactly 5 problems.*

## Your task

Create exactly one file:

```text
{self.solution_path}
```

It must be `Problem.lean` with the `sorry` in `imo2005_p6` replaced by a
complete proof. You MUST keep the `imo2005_p6` signature and the `solvedCount` /
`numSolvedExactly` definitions **exactly** as given (the grader pins them). You
may add any auxiliary lemmas/definitions before the theorem.

## Toolchain (installed at setup)

- Lean 4.27.0 and the prebuilt Mathlib v4.27 `.olean` cache are already installed
  (fetched at task setup). The `lean` binary path is in `{self.input_dir}/LEANBIN`
  and the olean search path is in `{self.input_dir}/LEANPATH`.
- Compile your file read-only against the cache with:

```bash
env LEAN_PATH="$(cat {self.input_dir}/LEANPATH)" "$(cat {self.input_dir}/LEANBIN)" {self.solution_path}
```

  (`lean` is also on `PATH`, so `env LEAN_PATH=... lean ...` works too.)
  Use the prebuilt cache — do NOT run `lake build` on Mathlib (it would rebuild
  from source and take hours). Iterate by editing `solution.lean` and re-running
  the compile command above.

## Grading (deterministic)

Score is 1.0 only if ALL of the following hold, else 0.0:
1. Your file keeps the exact required `imo2005_p6` signature and definitions.
2. It contains no `sorry`, `admit`, `native_decide`, or new `axiom`.
3. `lean solution.lean` compiles with no errors.
4. `#print axioms IMO2005P6.imo2005_p6` depends only on the standard trusted
   axioms `propext`, `Classical.choice`, `Quot.sound`. Any `sorryAx`, declared
   axiom, or `Lean.ofReduceBool` (from `native_decide`) fails the task.

In other words: only a genuine, complete, axiom-clean proof of the exact theorem
scores. Treat `{self.input_dir}` as read-only; keep your proof in
`{self.solution_path}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "variant_name": VARIANT_NAME,
                "input_dir": self.input_dir,
                "remote_output_dir": self.remote_output_dir,
                "problem_file": self.problem_file,
                "solution_path": self.solution_path,
                "lean_bin": self.lean_bin,
                "lean_version": LEAN_VERSION,
                "mathlib_rev": MATHLIB_REV,
                "elan_home": _ELAN_HOME,
                "lean_project": _LEAN_PROJECT,
                "lean_pkgs": _LEAN_PKGS,
                # lean_path is discovered + written by start() after cache fetch.
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


_ELAN_INIT_URL = "https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh"


def _result_out_code(result) -> tuple[str, int]:
    """Normalize a run_command result (dict {stdout,stderr,return_code,success}
    from RemoteDesktopSession, or a CompletedProcess-like object) to
    (combined_output, exit_code)."""
    if isinstance(result, dict):
        out = str(result.get("stdout", "")) + str(result.get("stderr", ""))
        code = result.get("return_code")
        if code is None:
            code = 0 if result.get("success") else 1
        return out, int(code)
    stdout = getattr(result, "stdout", result)
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    stderr = getattr(result, "stderr", "")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    code = getattr(result, "returncode", getattr(result, "exit_code", 0)) or 0
    return str(stdout) + str(stderr), int(code)


async def _setup_toolchain(session, meta) -> str:
    """Install Lean + fetch the prebuilt Mathlib cache from upstream, idempotently.
    Returns the discovered LEAN_PATH (colon-joined olean dirs). Raises on failure.

    NOTE: this sandbox's ``run_command`` does NOT report reliable exit codes
    (the cua interface defaults return_code to 0), so state is detected via
    STDOUT SENTINELS and the presence of expected files — never exit codes."""
    elan = meta["elan_home"]
    proj = meta["lean_project"]
    lean_bin = meta["lean_bin"]
    mathlib_olean = f"{meta['lean_pkgs']}/mathlib/.lake/build/lib/lean/Mathlib.olean"

    async def _out(cmd: str) -> str:
        r = await session.run_command(cmd, check=False)
        out, _ = _result_out_code(r)
        return out

    # Idempotent: a populated cache is present iff the sentinel prints.
    probe = await _out(f"test -x {lean_bin!r} && test -f {mathlib_olean!r} && echo __READY__")
    already = "__READY__" in probe
    logger.info("[imo2005p6] toolchain probe: already=%s", already)

    if not already:
        # 1. install elan + the pinned Lean toolchain from the official releases.
        o = await _out(
            f"curl -sSf --max-time 120 {_ELAN_INIT_URL} -o /tmp/elan-init.sh && "
            f"ELAN_HOME={elan!r} sh /tmp/elan-init.sh -y --default-toolchain {LEAN_VERSION} "
            f"&& test -x {lean_bin!r} && echo __ELAN_OK__"
        )
        logger.info("[imo2005p6] elan-install tail: %s", o[-400:])
        if "__ELAN_OK__" not in o:
            raise RuntimeError(f"imo2005p6: elan/Lean install failed: {o[-500:]}")

        # 2. create a pinned Lake project requiring Mathlib @ MATHLIB_REV.
        lakefile = (
            'name = "mlproj"\n'
            'defaultTargets = ["MlProj"]\n\n'
            '[[require]]\n'
            'name = "mathlib"\n'
            'scope = "leanprover-community"\n'
            f'rev = "{MATHLIB_REV}"\n'
        )
        await session.run_command(f"mkdir -p {proj}/MlProj", check=False)
        await session.write_file(f"{proj}/lakefile.toml", lakefile)
        await session.write_file(f"{proj}/lean-toolchain", LEAN_VERSION + "\n")
        await session.write_file(f"{proj}/MlProj.lean", "import Mathlib\n")

        # 3. resolve deps + fetch the PREBUILT olean cache from Mathlib's CDN
        #    (minutes; NOT `lake build`, which would compile from source for hours).
        o = await _out(
            f"cd {proj!r} && ELAN_HOME={elan!r} PATH={elan}/bin:$PATH lake update && echo __UPDATE_OK__"
        )
        logger.info("[imo2005p6] lake-update tail: %s", o[-400:])
        o = await _out(
            f"cd {proj!r} && ELAN_HOME={elan!r} PATH={elan}/bin:$PATH lake exe cache get 2>&1 | tail -3"
        )
        logger.info("[imo2005p6] cache-get tail: %s", o[-400:])
        # verify the cache actually landed (sentinel on the real olean file).
        chk = await _out(f"test -f {mathlib_olean!r} && echo __CACHE_OK__")
        if "__CACHE_OK__" not in chk:
            raise RuntimeError(f"imo2005p6: Mathlib cache not present after cache get: {o[-500:]}")

    # Discover the olean search path: every dependency's build/lib/lean dir.
    disc = await _out(
        f"for d in {meta['lean_pkgs']}/*/.lake/build/lib/lean; do "
        f"[ -d \"$d\" ] && printf '%s:' \"$d\"; done"
    )
    lean_path = disc.strip().rstrip(":")
    if not lean_path:
        raise RuntimeError("imo2005p6: no olean dirs found after cache fetch")
    logger.info("[imo2005p6] LEAN_PATH discovered: %d dirs", lean_path.count(":") + 1)
    return lean_path


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Fetch the Lean/Mathlib toolchain from upstream, stage Problem.lean, and
    write the LEANPATH/LEANBIN pointer files. No task-data upload: `elan` installs
    Lean and `lake exe cache get` pulls the prebuilt Mathlib cache over the
    sandbox's network."""
    meta = task_cfg.metadata

    for d in (meta["input_dir"], meta["remote_output_dir"]):
        await session.run_command(f"mkdir -p {d!r}", check=False)
    await session.run_command(f"rm -f {meta['solution_path']!r}", check=False)

    # Fetch/verify the toolchain and discover the olean search path.
    lean_path = await _setup_toolchain(session, meta)
    meta["lean_path"] = lean_path  # record for evaluate()

    # The problem statement (statement + fixed defs + `sorry`). No proof, no hints.
    await session.write_file(meta["problem_file"], _PROBLEM_LEAN.read_text(encoding="utf-8"))

    # Pointer files: LEANPATH = olean search path, LEANBIN = the lean binary.
    await session.write_file(f"{meta['input_dir']}/LEANPATH", lean_path)
    await session.write_file(f"{meta['input_dir']}/LEANBIN", meta["lean_bin"])

    # Expose `lean` on PATH for the agent's convenience.
    await session.run_command(
        f"ln -sf {meta['lean_bin']!r} /usr/local/bin/lean 2>/dev/null "
        f"|| sudo ln -sf {meta['lean_bin']!r} /usr/local/bin/lean 2>/dev/null || true",
        check=False,
    )

    logger.info("[imo2005p6] toolchain ready; staged problem at %s; lean_bin=%s",
                meta["problem_file"], meta["lean_bin"])


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Compile the agent's solution.lean in-sandbox and audit its axioms."""
    meta = task_cfg.metadata

    if grade_lean_async is None:
        logger.error("[imo2005p6] grade_lean did not import")
        return [0.0]

    # The agent must have produced solution.lean.
    try:
        raw = await session.read_bytes(meta["solution_path"])
        src = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    except Exception as exc:
        logger.info("[imo2005p6] solution missing at %s: %s", meta["solution_path"], exc)
        return [0.0]

    # LEAN_PATH: prefer the pointer file start() wrote (durable across hooks);
    # fall back to metadata if present.
    lean_path = meta.get("lean_path")
    try:
        raw_lp = await session.read_bytes(f"{meta['input_dir']}/LEANPATH")
        lp = raw_lp.decode("utf-8", "replace") if isinstance(raw_lp, bytes) else str(raw_lp)
        lp = lp.strip()
        if lp:
            lean_path = lp
    except Exception:
        pass
    if not lean_path:
        logger.error("[imo2005p6] no LEANPATH available for grading")
        return [0.0]

    async def run(cmd: str, timeout: int):
        # RemoteDesktopSession.run_command takes no timeout kwarg; the `timeout`
        # arg from the grader is advisory only (the harness wall clock bounds us).
        try:
            result = await session.run_command(cmd, check=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("[imo2005p6] command failed: %s (%s)", cmd, exc)
            return ("", 1)
        out, code = _result_out_code(result)
        return (out, code)

    result = await grade_lean_async(
        src=src,
        run=run,
        solution_path=meta["solution_path"],
        lean_path_env=lean_path,
        workdir=meta["remote_output_dir"],
        lean_bin=meta.get("lean_bin", "lean"),
        per_compile_timeout=_COMPILE_TIMEOUT_S,
    )
    logger.info("[imo2005p6] score=%.4f gate=%s reason=%s", result.score, result.gate, result.reason)
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
