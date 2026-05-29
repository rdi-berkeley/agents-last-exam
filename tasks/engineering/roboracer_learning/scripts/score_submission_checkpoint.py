#!/usr/bin/env python
"""VM-side scorer for engineering/roboracer_learning."""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

OBSERV_DIM = 1080
HIDDEN_DIM = 256
ACTION_DIM = 1
LR = 0.001
MAX_LAPS = 100
MAX_STEPS_PER_LAP = 20000
SPEED = 2.75
MAP_CONFIG_PATH = "imitation_learning/map/levine2nd/levine2nd_config.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starter-project", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--work-root", required=True)
    return parser.parse_args()


def _json_safe(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _load_map_config(repo_root: Path):
    config_path = repo_root / MAP_CONFIG_PATH
    il_dir = repo_root / "imitation_learning"
    with config_path.open("r", encoding="utf-8") as handle:
        conf_dict = yaml.load(handle, Loader=yaml.FullLoader)
    if conf_dict.get("map_path", "").startswith("./"):
        conf_dict["map_path"] = str(il_dir / conf_dict["map_path"][2:])
    if conf_dict.get("wpt_path", "").startswith("./"):
        conf_dict["wpt_path"] = str(il_dir / conf_dict["wpt_path"][2:])
    return argparse.Namespace(**conf_dict)


def _prepare_imports(repo_root: Path) -> None:
    import_root = str(repo_root)
    gym_root = str(repo_root / "gym")
    for path in [gym_root, import_root]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_f110env(repo_root: Path):
    import importlib.util

    env_module_path = repo_root / "gym" / "f110_gym" / "envs" / "f110_env.py"
    spec = importlib.util.spec_from_file_location("f110_env_direct", str(env_module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.F110Env


def _load_policy(repo_root: Path, checkpoint_path: Path):
    _prepare_imports(repo_root)
    from imitation_learning.policies.agents.agent_mlp import AgentPolicyMLP

    device = torch.device("cpu")
    agent = AgentPolicyMLP(OBSERV_DIM, HIDDEN_DIM, ACTION_DIM, LR, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    return agent


def _load_expert(repo_root: Path, map_conf):
    _prepare_imports(repo_root)
    from imitation_learning.policies.experts.expert_waypoint_follower import ExpertWaypointFollower

    return ExpertWaypointFollower(map_conf)


def _compute_bhattacharyya_distance(agent, expert, observations) -> float:
    tlad = 0.5
    vgain = 0.4
    agent_actions = []
    expert_actions = []

    for obs_dict in observations:
        scan = obs_dict["scans"][0]
        with torch.no_grad():
            agent_action = agent.get_action(scan)
        if isinstance(agent_action, np.ndarray):
            agent_action = float(agent_action.flatten()[0])
        agent_actions.append(agent_action)

        pose_x = obs_dict["poses_x"][0]
        pose_y = obs_dict["poses_y"][0]
        pose_theta = obs_dict["poses_theta"][0]
        _, expert_steer = expert.plan(pose_x, pose_y, pose_theta, tlad, vgain)
        expert_actions.append(expert_steer)

    agent_actions = np.array(agent_actions)
    expert_actions = np.array(expert_actions)
    all_actions = np.concatenate([agent_actions, expert_actions])
    bin_min = all_actions.min() - 0.01
    bin_max = all_actions.max() + 0.01
    bins = np.linspace(bin_min, bin_max, 51)

    hist_agent, _ = np.histogram(agent_actions, bins=bins, density=True)
    hist_expert, _ = np.histogram(expert_actions, bins=bins, density=True)
    hist_agent = hist_agent / (hist_agent.sum() + 1e-10)
    hist_expert = hist_expert / (hist_expert.sum() + 1e-10)

    bc_coeff = np.sum(np.sqrt(hist_agent * hist_expert))
    bc_coeff = min(float(bc_coeff), 1.0)
    return float(-np.log(bc_coeff + 1e-10))


def _evaluate_checkpoint(repo_root: Path, checkpoint_path: Path) -> dict:
    map_conf = _load_map_config(repo_root)
    agent = _load_policy(repo_root, checkpoint_path)
    expert = _load_expert(repo_root, map_conf)
    _prepare_imports(repo_root)
    F110Env = _load_f110env(repo_root)

    env = F110Env(map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])

    sampled_observations = []
    laps_completed = 0
    lap_times = []
    crashed = False

    obs, _, _, _ = env.reset(start_pose)
    prev_lap_count = 0
    prev_lap_time = 0.0

    try:
        for step in range(MAX_LAPS * MAX_STEPS_PER_LAP):
            if len(sampled_observations) < 1000 and step % 50 == 0:
                sampled_observations.append(
                    {
                        "scans": obs["scans"],
                        "poses_x": obs["poses_x"],
                        "poses_y": obs["poses_y"],
                        "poses_theta": obs["poses_theta"],
                    }
                )

            scan = obs["scans"][0]
            with torch.no_grad():
                action_steer = agent.get_action(scan)
            if isinstance(action_steer, np.ndarray):
                action_steer = float(action_steer.flatten()[0])
            action = np.array([[action_steer, SPEED]])

            obs, _, done, _ = env.step(action)
            current_lap_count = int(obs["lap_counts"][0])
            current_lap_time = float(obs["lap_times"][0])

            if current_lap_count > prev_lap_count:
                new_laps = current_lap_count - prev_lap_count
                lap_time = current_lap_time - prev_lap_time
                lap_times.append(lap_time)
                laps_completed += new_laps
                prev_lap_count = current_lap_count
                prev_lap_time = current_lap_time
                if laps_completed >= MAX_LAPS:
                    break

            if done:
                if obs["collisions"][0]:
                    crashed = True
                    break
                if laps_completed >= MAX_LAPS:
                    break
                obs, _, _, _ = env.reset(start_pose)
                prev_lap_count = 0
                prev_lap_time = 0.0
    finally:
        env.close()

    avg_lap_time = float(np.mean(lap_times)) if lap_times else None
    if crashed:
        distance_before_crash = SPEED * float(obs["lap_times"][0])
    else:
        distance_before_crash = float("inf")

    if sampled_observations:
        bd = _compute_bhattacharyya_distance(agent, expert, sampled_observations)
    else:
        bd = float("inf")

    return {
        "avg_lap_time_sec": avg_lap_time,
        "distance_before_crash_m": distance_before_crash,
        "laps_completed": laps_completed,
        "bhattacharyya_distance_vs_pure_pursuit": bd,
    }


def _score(metrics: dict, expected: dict) -> dict:
    thresholds = expected["thresholds"]
    reasons = []
    passed = {
        "checkpoint_exists": True,
        "laps_completed": False,
        "avg_lap_time_sec": False,
        "bhattacharyya_distance_vs_pure_pursuit": False,
    }

    laps_completed = metrics.get("laps_completed")
    if isinstance(laps_completed, int) and laps_completed >= thresholds["laps_completed_min"]:
        passed["laps_completed"] = True
    else:
        reasons.append(
            f"laps_completed={laps_completed!r} < required {thresholds['laps_completed_min']}"
        )

    avg_lap_time = metrics.get("avg_lap_time_sec")
    if isinstance(avg_lap_time, (int, float)) and avg_lap_time <= thresholds["avg_lap_time_sec_max"]:
        passed["avg_lap_time_sec"] = True
    else:
        reasons.append(
            f"avg_lap_time_sec={avg_lap_time!r} > allowed {thresholds['avg_lap_time_sec_max']}"
        )

    bd = metrics.get("bhattacharyya_distance_vs_pure_pursuit")
    if isinstance(bd, (int, float)) and bd <= thresholds["bhattacharyya_distance_vs_pure_pursuit_max"]:
        passed["bhattacharyya_distance_vs_pure_pursuit"] = True
    else:
        reasons.append(
            "bhattacharyya_distance_vs_pure_pursuit="
            f"{bd!r} > allowed {thresholds['bhattacharyya_distance_vs_pure_pursuit_max']}"
        )

    score = 1.0 if all(passed.values()) else 0.0
    return {
        "score": score,
        "passed": passed,
        "reasons": reasons,
        "metrics": _json_safe(metrics),
        "thresholds": _json_safe(thresholds),
        "reference_metrics": _json_safe(expected.get("reference_metrics", {})),
    }


def main() -> int:
    args = _parse_args()
    starter_project = Path(args.starter_project)
    checkpoint = Path(args.checkpoint)
    reference_json = Path(args.reference_json)
    work_root = Path(args.work_root)

    if not starter_project.exists():
        print(json.dumps({"score": 0.0, "reasons": [f"missing starter project: {starter_project}"]}))
        return 0
    if not reference_json.exists():
        print(json.dumps({"score": 0.0, "reasons": [f"missing reference json: {reference_json}"]}))
        return 0
    if not checkpoint.exists():
        print(json.dumps({"score": 0.0, "reasons": [f"missing checkpoint: {checkpoint}"]}))
        return 0

    expected = json.loads(reference_json.read_text(encoding="utf-8"))
    repo_root = work_root / "starter_project"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(starter_project, repo_root)

    try:
        metrics = _evaluate_checkpoint(repo_root, checkpoint)
        payload = _score(metrics, expected)
    except Exception as exc:  # noqa: BLE001 - return structured hard-fail payload
        payload = {
            "score": 0.0,
            "reasons": [f"evaluation_error: {type(exc).__name__}: {exc}"],
            "checkpoint": str(checkpoint),
        }

    print(json.dumps(_json_safe(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
