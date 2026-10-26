"""Re-grade finished ALE runs with references staged from the host mirror.

For each run dir (``<root>/**/v0/<ts>/`` with run.json + output/): fresh sandbox, task setup,
the saved agent output restored into the task's output dir, reference staged (localref
backend), ``TaskDriver.evaluate()``; the result is written to ``<run_dir>/regrade.json``.

    DAYTONA_API_KEY=... uv run python env/daytona/regrade.py daytona_windows_exp.yaml \
        --refs <data_root>/tasks [--root .logs/ale] [--only <substr>] [--limit N] [--force]
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); os.chdir(REPO_ROOT)

from ale_run.environments.env import ALEEnv
from ale_run.environments.task_data import join, local_ref, task_subdir
from ale_run.orchestration.config_loader import load_experiment
from ale_run.orchestration.factory import EnvironmentRouter
from ale_run.orchestration.lifecycle import _build_env_spec
from ale_run.tasks.driver import TaskDriver
from ale_run.tasks.loader import TaskLoader

log = logging.getLogger("regrade")
CHUNK_BYTES = 1500 * 1024 * 1024


def latest_runs(root: Path):
    """Latest attempt (by timestamp dir) per (agent/model, task) that has a run.json."""
    best = {}
    for rj in root.rglob("run.json"):
        d = rj.parent
        m = re.search(r"/claude_code/([^/]+)/([^/]+)/v(\d+)/(\d{8}_\d{6})$", str(d))
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3))
        if key not in best or m.group(4) > best[key][0]:
            best[key] = (m.group(4), d)
    return [d for _, d in sorted(best.values(), key=lambda x: str(x[1]))]


async def regrade_one(run_dir: Path, router, refs_root: str, eval_timeout: int) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    task_rel = run["task"]["path"]                      # tasks/<domain>/<task>
    variant = int(run["task"].get("variant_index") or 0)
    out_dir = run_dir / "output"
    task_meta = TaskLoader(task_rel).load(variant)
    env_spec = dataclasses.replace(_build_env_spec(task_meta), task_id=task_rel.removeprefix("tasks/"))  # isolation frees disk
    provider = router.provider_for(env_spec.snapshot)
    env = ALEEnv(provider=provider, spec=env_spec)
    t0 = time.time()
    await env.reset_async()
    try:
        driver = TaskDriver(task_path=task_rel, session=env.session, variant=variant,
                            os_type=env.sandbox.os, session_rebuilder=env.reset_session)
        await driver.setup()
        task_data = task_meta["task_data"]
        base = task_subdir(env.sandbox, task_data)
        n_out = -1
        if out_dir.is_dir() and any(out_dir.iterdir()):
            # Restore in <= CHUNK_BYTES archives so archive + extracted files never exceed the
            # sandbox's free disk (ar_full_1500's 8 GB of PDFs blew a single-archive restore).
            tmpdir = Path(tempfile.gettempdir())
            files = sorted(f for f in out_dir.rglob("*") if f.is_file())
            chunks, cur, size = [], [], 0
            for f in files:
                cur.append(f); size += f.stat().st_size
                if size >= CHUNK_BYTES:
                    chunks.append(cur); cur, size = [], 0
            if cur or not chunks:
                chunks.append(cur)
            staging = join(env.sandbox, env.sandbox.task_data_root, "_stage")
            await env.sandbox.mkdir(staging)
            target = join(env.sandbox, base, "output")
            n_out = 0
            for i, chunk in enumerate(chunks):
                local_zip = tmpdir / f"{run['run_id'][:100]}_output_{i:03d}.zip"
                try:
                    with zipfile.ZipFile(local_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                        for f in chunk:
                            zf.write(f, f.relative_to(out_dir).as_posix())
                        if i == 0:
                            for d in sorted(x for x in out_dir.rglob("*") if x.is_dir() and not any(x.iterdir())):
                                zf.writestr(d.relative_to(out_dir).as_posix() + "/", b"")
                    zp = join(env.sandbox, staging, f"{task_data.task_name}_output_{i:03d}.zip")
                    await local_ref.upload_path(env.sandbox, str(local_zip), zp)
                    n_out = await local_ref.expand_zip(env.sandbox, zp, target, clear=(i == 0))
                finally:
                    local_zip.unlink(missing_ok=True)
            if n_out != len(files):
                raise RuntimeError(f"output restore mismatch: {n_out} files in sandbox vs {len(files)} local")
        ref = await local_ref.stage_reference(env.sandbox, task_data, source=f"localref:{refs_root}")
        result = await asyncio.wait_for(driver.evaluate(), timeout=eval_timeout)
        score = result.get("score") if isinstance(result, dict) else result
        return {"run_id": run["run_id"], "task": task_rel, "model": run["agent"].get("model"),
                "original_status": run.get("status"), "original_score": run.get("score"),
                "regrade_score": score, "eval_result": result, "reference": ref,
                "output_files_restored": n_out, "duration_s": round(time.time() - t0, 1),
                "regraded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    finally:
        try:
            await env.close_async(mode="delete")
        except Exception as e:  # noqa: BLE001
            log.warning("release failed: %s", e)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--refs", required=True)
    ap.add_argument("--root", default=".logs/ale"); ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-timeout", type=int, default=3600); ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    spec = load_experiment(args.spec)
    router = EnvironmentRouter(spec.environment)
    runs = [d for d in latest_runs(Path(args.root)) if args.only in str(d)]
    if not args.force:
        runs = [d for d in runs if not (d / "regrade.json").exists()]
        # Runs graded with a staged reference during the sweep (localref) already carry a real score.
        def graded_with_reference(d: Path) -> bool:
            ev = d / "events.jsonl"
            return ev.exists() and '"reference_stage_completed"' in ev.read_text(errors="replace")
        runs = [d for d in runs if not graded_with_reference(d)]
    if args.limit:
        runs = runs[: args.limit]
    log.info("regrading %d run(s)", len(runs))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(d: Path):
        async with sem:
            try:
                rep = await regrade_one(d, router, os.path.expanduser(args.refs), args.eval_timeout)
            except Exception as e:
                log.exception("regrade failed for %s", d)
                rep = {"error": f"{type(e).__name__}: {e}", "regraded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (d / "regrade.json").write_text(json.dumps(rep, indent=2, default=str))
            log.info("REGRADE %s -> %s", d.name and str(d).split("/claude_code/")[-1][:110], rep.get("regrade_score", rep.get("error")))

    await asyncio.gather(*(one(d) for d in runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
