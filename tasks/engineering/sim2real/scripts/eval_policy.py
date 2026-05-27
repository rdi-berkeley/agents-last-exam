"""VM-side evaluation script for sim2real reach task.

Loads the agent's policy from output/, runs it in XArmPickPlace-v0 for N episodes,
and prints a JSON score report to stdout.

Usage (run from the input/ directory so xarm package is available):
    cd <variant>/input
    MUJOCO_GL=egl uv run python <path>/eval_policy.py \
        --policy-dir <variant>/output \
        --num-episodes 50 \
        --seed-start 1000 \
        --max-steps 1000

Output (stdout, JSON):
    {"score": 0.96, "successes": 48, "episodes": 50, "details": [...]}
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import traceback

# Ensure MuJoCo can run headless
os.environ.setdefault("MUJOCO_GL", "egl")


def _find_checkpoint(policy_dir: str) -> str | None:
    """Find the .pt checkpoint file in policy_dir. Prefers the latest iteration."""
    pts = sorted(glob.glob(os.path.join(policy_dir, "*.pt")))
    return pts[-1] if pts else None


def _load_agent_policy(policy_dir: str):
    """Dynamically import the agent's policy.py and load the checkpoint."""
    policy_py = os.path.join(policy_dir, "policy.py")
    if not os.path.isfile(policy_py):
        raise FileNotFoundError(f"No policy.py found in {policy_dir}")

    spec = importlib.util.spec_from_file_location("agent_policy", policy_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "load_policy"):
        raise AttributeError("policy.py does not implement load_policy()")

    ckpt = _find_checkpoint(policy_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No .pt checkpoint found in {policy_dir}")

    return mod.load_policy(ckpt)


def run_evaluation(
    policy_dir: str,
    num_episodes: int = 50,
    seed_start: int = 1000,
    max_steps: int = 1000,
) -> dict:
    """Run evaluation episodes and return results dict."""
    import gymnasium as gym

    import xarm  # noqa: F401 — triggers env registration

    policy = _load_agent_policy(policy_dir)

    env = gym.make("XArmPickPlace-v0", task_name="reach")

    successes = 0
    details = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        policy.start_new_game()

        episode_success = False
        final_step = 0

        for step in range(max_steps):
            try:
                action = policy.act(obs)
            except Exception:
                print(
                    f"Policy.act() crashed on episode {ep}: {traceback.format_exc()}",
                    file=sys.stderr,
                )
                break

            obs, reward, terminated, truncated, info = env.step(action)
            final_step = step + 1

            if info.get("is_success", False):
                episode_success = True
                break

            if terminated or truncated:
                break

        successes += int(episode_success)
        details.append({
            "seed": seed,
            "success": episode_success,
            "steps": final_step,
        })

    env.close()

    score = successes / num_episodes if num_episodes > 0 else 0.0
    return {
        "score": round(score, 4),
        "successes": successes,
        "episodes": num_episodes,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a reach policy")
    parser.add_argument("--policy-dir", required=True, help="Directory with policy.py and *.pt")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    result = run_evaluation(
        policy_dir=args.policy_dir,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
    )

    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
