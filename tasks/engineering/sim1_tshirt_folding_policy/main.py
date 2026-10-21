"""AgentHLE task: engineering/sim1_tshirt_folding_policy."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.engineering.sim1_tshirt_folding_policy.scripts.score_sim1_tshirt_folding_policy import (
    evaluate_submission,
    merge_results,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()
logger = logging.getLogger(__name__)

VARIANTS = [("base", "SIM1 short-sleeve T-shirt fold, transfer to unseen fabric materials")]
SIM1_COMMIT = "973476b7f201780d8d653bd76f61062890693385"
EVAL_TMP_DIR = "/dev/shm/agenthle_eval/sim1_tshirt_folding_policy"
SIM_POLL_INTERVAL = 60
# Must stay under the task card's vm.timeout: the VM is torn down at that point,
# so a longer wait here can only expire after the results are already lost.
# 25 episodes at ~2 min plus per-process Warp compilation runs 60-75 min.
SIM_TIMEOUT = 2 * 3600
# Concurrent run_eval.py processes per GPU. Measured on one L4: 1 -> 13.8 ep/h,
# 3 -> 26.9 ep/h (1.95x), 6 -> 27.4 ep/h (1.99x). Aggregate throughput saturates
# at 3; beyond that per-process latency grows linearly for no extra throughput.
EVAL_PROCS_PER_GPU = 3
# Episodes (seeds of one test material) per run_eval.py process. One shard per
# material: Warp recompiles its kernels in every new process (~4 min), so more
# shards than GPUs buys nothing and pays that cost again.
EPISODES_PER_SHARD = 5


def _bash(cmd: str) -> str:
    return "bash -lc " + shlex.quote(cmd)


@dataclass
class Sim1TshirtFoldingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "sim1_tshirt_folding_policy"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def sim1_root(self) -> str:
        return f"{self.software_dir}/sim1"

    @property
    def python_bin(self) -> str:
        return f"{self.software_dir}/venv/bin/python"

    @property
    def output_policy(self) -> str:
        return f"{self.remote_output_dir}/policy.py"

    @property
    def output_checkpoint(self) -> str:
        return f"{self.remote_output_dir}/checkpoint.pt"

    @property
    def task_description(self) -> str:
        i = self.input_dir
        return f"""\
You are learning a T-shirt folding policy for a dual-arm robot in the SIM1 cloth
simulator (Newton physics: MuJoCo-Warp robot + VBD cloth) that must transfer to
fabric materials it has never seen.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task brief: `{i}/task_brief.md`
- Simulator specification (world, robot, cloth, materials, control, observation/action): `{i}/SIM_SPEC.md`
- Success criteria, metrics, test protocol and exact evaluation command: `{i}/SUCCESS.md`
- Training fabric materials (18, each a `physics.json` plus SIM1 texture): `{i}/assets/materials/train`
- Robot, shirt mesh, cloth topology and landmarks: `{i}/assets`
- Evaluation harness (environment, metrics, `run_eval.py`): `{i}/eval`
- Starter template: `{i}/submission_template`
- Simulator install and Python env: `{self.software_dir}` (run with `{self.python_bin}`)

## What You Must Do
1. Read the task brief, simulator specification, and success criteria.
2. Train, script, or otherwise produce a closed-loop policy, using only the 18 training
   materials, that folds the shirt flat, compact, and neatly rectangular across randomized
   cloth poses AND across fabric materials, within a 10 s policy phase. The final score
   comes only from 5 held-out materials you never see (5 poses each, 25 episodes). Each
   episode earns continuous credit for how compact and neat the fold is, 0 for an
   unfolded shirt; the score is the mean credit over those episodes.
   The policy is not told which material it is folding, so it must adapt from the observed
   cloth state.
3. Save exactly two files in `{self.remote_output_dir}`: `policy.py` and `checkpoint.pt`.

## Output Requirements
- `policy.py` defines `Policy(checkpoint_path: str, device: str)` with `reset()` and
  `inference(obs)` returning exactly `{{"action": tensor}}` of shape `(N, 16)`, float32.
- Do not modify files under `{i}` or `{self.software_dir}`; the grader verifies them.
- Do not write final answers outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "sim1_root": self.sim1_root,
                "python_bin": self.python_bin,
                "output_policy": self.output_policy,
                "output_checkpoint": self.output_checkpoint,
            }
        )
        return metadata


def _configs() -> list[Sim1TshirtFoldingConfig]:
    return [Sim1TshirtFoldingConfig(VARIANT_NAME=n, VARIANT_LABEL=label) for n, label in VARIANTS]


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
        for cfg in _configs()
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await _setup(task_cfg, session)
    out = shlex.quote(meta["remote_output_dir"])
    await session.run_command(_bash(f"rm -rf {out} && mkdir -p {out}"), check=False)

    for path in (f"{meta['input_dir']}/SIM_SPEC.md", f"{meta['input_dir']}/eval/run_eval.py", meta["python_bin"]):
        if not await session.file_exists(path):
            raise RuntimeError(f"staged input missing: {path}")
    if await session.directory_exists(meta["reference_dir"]):
        raise RuntimeError(f"reference must be hidden during the agent run: {meta['reference_dir']}")


async def _check_integrity(session: cb.DesktopSession, meta: dict, manifest: dict[str, str]) -> tuple[bool, str]:
    files = " ".join(shlex.quote(rel) for rel in manifest)
    res = await session.run_command(_bash(f"cd {shlex.quote(meta['input_dir'])} && sha256sum -- {files}"), check=False)
    observed = {}
    for line in (res.get("stdout") or "").splitlines():
        digest, _, rel = line.partition("  ")
        observed[rel.strip()] = digest.strip()
    for rel, digest in manifest.items():
        if observed.get(rel) != digest:
            return False, f"input file missing or modified: {rel}"

    sim1 = shlex.quote(meta["sim1_root"])
    head = await session.run_command(_bash(f"git -C {sim1} rev-parse HEAD"), check=False)
    if (head.get("stdout") or "").strip() != SIM1_COMMIT:
        return False, "software/sim1 is not at the pinned commit"
    dirty = await session.run_command(_bash(f"git -C {sim1} status --porcelain --untracked-files=no"), check=False)
    if (dirty.get("stdout") or "").strip():
        return False, "software/sim1 has local modifications"
    return True, "inputs_intact"


async def _run_rollouts(session: cb.DesktopSession, meta: dict, episodes: dict[str, list[int]]) -> list[str] | None:
    """Run the pristine reference/eval harness on the hidden test materials and poll for results.

    One run_eval.py process per EPISODES_PER_SHARD seeds of one test material,
    EVAL_PROCS_PER_GPU processes per GPU, all in the background. Returns the shard results paths.
    """
    run_dir = f"{EVAL_TMP_DIR}/run"
    ref = meta["reference_dir"]
    shard_cmds, results = [], []
    shards = [
        (material, seeds[i : i + EPISODES_PER_SHARD], i // EPISODES_PER_SHARD)
        for material, seeds in sorted(episodes.items())
        for i in range(0, len(seeds), EPISODES_PER_SHARD)
    ]
    for k, (material, seeds, part) in enumerate(shards):
        res = f"{run_dir}/results_{material}_{part}.json"
        results.append(res)
        cmd = " ".join(
            [
                f"CUDA_VISIBLE_DEVICES=$(( {k} % NGPU ))",
                f"SIM1_ROOT={shlex.quote(meta['sim1_root'])}",
                shlex.quote(meta["python_bin"]),
                shlex.quote(f"{ref}/eval/run_eval.py"),
                "--policy", shlex.quote(meta["output_policy"]),
                "--checkpoint", shlex.quote(meta["output_checkpoint"]),
                "--materials", shlex.quote(f"{ref}/materials/test/{material}"),
                "--seeds", *[str(s) for s in seeds],
                "--device", "cuda",
                "--assets_root", shlex.quote(f"{meta['input_dir']}/assets"),
                "--results_path", shlex.quote(res),
                "--states_dir", shlex.quote(f"{run_dir}/states"),
                ">", shlex.quote(f"{run_dir}/eval_{material}_{part}.log"), "2>&1",
            ]
        )
        shard_cmds.append(cmd)
    driver = "\n".join(
        [
            "NGPU=$(nvidia-smi -L | wc -l); [ \"$NGPU\" -ge 1 ] || NGPU=1",
            f"SLOTS=$(( NGPU * {EVAL_PROCS_PER_GPU} ))",
            *[f"{c} &\nwhile [ $(jobs -rp | wc -l) -ge $SLOTS ]; do sleep 5; done" for c in shard_cmds],
            "wait",
            f"touch {shlex.quote(run_dir + '/DONE')}",
        ]
    )
    driver_path = f"{run_dir}/driver.sh"
    await session.run_command(_bash(f"rm -rf {shlex.quote(run_dir)} && mkdir -p {shlex.quote(run_dir)}"), check=False)
    encoded = base64.b64encode((driver + "\n").encode()).decode()
    await session.run_command(_bash(f"echo {encoded} | base64 -d > {shlex.quote(driver_path)}"), check=False)
    await session.run_command(
        _bash(f"nohup bash {shlex.quote(driver_path)} > {shlex.quote(run_dir + '/driver.log')} 2>&1 &"), check=False
    )

    elapsed = 0
    while elapsed < SIM_TIMEOUT:
        await asyncio.sleep(SIM_POLL_INTERVAL)
        elapsed += SIM_POLL_INTERVAL
        if await session.file_exists(f"{run_dir}/DONE"):
            break
    present = [r for r in results if await session.file_exists(r)]
    if len(present) != len(results):
        tail = await session.run_command(_bash(f"tail -n 20 {shlex.quote(run_dir)}/eval_*.log"), check=False)
        logger.error(
            "%d/%d shards produced results after %ds:\n%s",
            len(present), len(results), elapsed, tail.get("stdout", ""),
        )
        return None
    return results


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    ref = meta["reference_dir"]

    for path in (meta["output_policy"], meta["output_checkpoint"]):
        if not await session.file_exists(path):
            logger.error("[%s] missing output %s", tag, path)
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="sim1_fold_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "output"
        local_ref = tmp / "reference"
        (local_ref / "baseline").mkdir(parents=True)
        local_output.mkdir()
        try:
            for rel in ("eval_config.json", "input_manifest.json", "baseline/policy.py", "baseline/checkpoint.pt"):
                (local_ref / rel).write_bytes(await session.read_bytes(f"{ref}/{rel}"))
            for name, key in (("policy.py", "output_policy"), ("checkpoint.pt", "output_checkpoint")):
                (local_output / name).write_bytes(await session.read_bytes(meta[key]))
            smoke = tmp / "smoke_obs.npz"
            smoke.write_bytes(await session.read_bytes(f"{meta['input_dir']}/assets/smoke_obs.npz"))

            manifest = json.loads((local_ref / "input_manifest.json").read_text())
            integrity = await _check_integrity(session, meta, manifest)

            remote_results = None
            if integrity[0]:
                cfg = json.loads((local_ref / "eval_config.json").read_text())
                remote_results = await _run_rollouts(session, meta, cfg["episodes"])
            local_results = None
            if remote_results is not None:
                shards = []
                for k, remote in enumerate(remote_results):
                    shard = tmp / f"results_{k}.json"
                    shard.write_bytes(await session.read_bytes(remote))
                    shards.append(shard)
                local_results = tmp / "results.json"
                local_results.write_text(json.dumps(merge_results(shards)))

            result = evaluate_submission(
                local_output,
                local_ref,
                smoke,
                input_integrity=integrity,
                results_path=local_results,
                allow_fixture_hash_fallback=False,
            )
        except Exception as exc:
            logger.exception("[%s] evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True, default=str))
    return [float(result.get("score", 0.0))]
