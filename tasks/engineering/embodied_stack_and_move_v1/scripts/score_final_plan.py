"""Local scorer for embodied_stack_and_move_v1."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_EPS = 1e-6
GRASP_XY_TOL = 0.02
GRASP_Z_REACH = 0.08
SUPPORT_XY_TOL = 0.02

EVALUATOR_ACTION_SCHEMA_DOC = {
    "allowed_actions": [
        {"name": "home", "parameters": []},
        {"name": "move_to_pose", "parameters": ["arm", "pose"]},
        {"name": "grasp", "parameters": ["arm"]},
        {"name": "release", "parameters": ["arm"]},
    ],
    "valid_arms": ["left", "right"],
    "pose_format": "[x, y, z, roll, pitch, yaw]",
}

EVALUATOR_SCENE_STATE_DOC = {
    "robot_state": {
        "left_arm_pose": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        "right_arm_pose": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
    },
    "objects": [
        {
            "id": "red_cube",
            "type": "cube",
            "size": [0.04, 0.04, 0.04],
            "pose": [0.2, 0.1, 0.02, 0.0, 0.0, 0.0],
        },
        {
            "id": "blue_cube",
            "type": "cube",
            "size": [0.04, 0.04, 0.04],
            "pose": [0.32, 0.1, 0.02, 0.0, 0.0, 0.0],
        },
        {
            "id": "green_cylinder",
            "type": "cylinder",
            "radius": 0.02,
            "height": 0.05,
            "pose": [0.26, -0.08, 0.025, 0.0, 0.0, 0.0],
        },
    ],
}

EVALUATOR_WORKSPACE_CONFIG_DOC = {
    "workspace_bounds": {
        "x_min": 0.0,
        "x_max": 0.5,
        "y_min": -0.2,
        "y_max": 0.2,
        "z_min": 0.0,
        "z_max": 0.5,
    },
    "corner_region": {
        "x_min": 0.42,
        "x_max": 0.5,
        "y_min": 0.12,
        "y_max": 0.2,
    },
    "stack_target": {
        "bottom_object": "blue_cube",
        "top_object": "red_cube",
        "position_tolerance_xy": 0.02,
        "height_tolerance_z": 0.02,
    },
    "stability_tolerance": 0.02,
}

EVALUATOR_REFERENCE_GOAL_DOC = {
    "red_cube": {
        "must_be_stacked_on": "blue_cube",
        "position_tolerance_xy": 0.02,
        "height_tolerance_z": 0.02,
    },
    "green_cylinder": {
        "must_be_inside_region": "corner_region",
    },
    "workspace_constraint": "all_objects_within_workspace_bounds",
    "stability_constraint": "red_blue_stack_stable_after_green_cylinder_move",
}


@dataclass
class ObjectState:
    object_id: str
    raw: dict[str, Any]
    pose: list[float]

    @property
    def obj_type(self) -> str:
        return str(self.raw["type"])

    @property
    def half_height(self) -> float:
        if self.obj_type == "cube":
            return float(self.raw["size"][2]) / 2.0
        if self.obj_type == "cylinder":
            return float(self.raw["height"]) / 2.0
        raise ValueError(f"unsupported object type: {self.obj_type}")

    @property
    def half_extents_xy(self) -> tuple[float, float]:
        if self.obj_type == "cube":
            return (
                float(self.raw["size"][0]) / 2.0,
                float(self.raw["size"][1]) / 2.0,
            )
        if self.obj_type == "cylinder":
            radius = float(self.raw["radius"])
            return (radius, radius)
        raise ValueError(f"unsupported object type: {self.obj_type}")

    @property
    def top_z(self) -> float:
        return float(self.pose[2]) + self.half_height

    def center_xy(self) -> tuple[float, float]:
        return (float(self.pose[0]), float(self.pose[1]))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(reason: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"score": 0.0, "reason": reason}
    if details:
        payload["details"] = details
    return payload


def _success(reason: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"score": 1.0, "reason": reason}
    if details:
        payload["details"] = details
    return payload


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _within_workspace(obj: ObjectState, bounds: dict[str, float], eps: float) -> bool:
    half_x, half_y = obj.half_extents_xy
    x, y, z = map(float, obj.pose[:3])
    return (
        x - half_x >= float(bounds["x_min"]) - eps
        and x + half_x <= float(bounds["x_max"]) + eps
        and y - half_y >= float(bounds["y_min"]) - eps
        and y + half_y <= float(bounds["y_max"]) + eps
        and z - obj.half_height >= float(bounds["z_min"]) - eps
        and z + obj.half_height <= float(bounds["z_max"]) + eps
    )


def _validate_pose(pose: Any) -> list[float] | None:
    if not isinstance(pose, list) or len(pose) != 6:
        return None
    values = [float(v) for v in pose] if all(_is_number(v) for v in pose) else None
    return values


def _required_params_by_action(action_schema: dict[str, Any]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for item in action_schema["allowed_actions"]:
        mapping[str(item["name"])] = set(item["parameters"])
    return mapping


def _validate_plan_structure(plan_doc: Any, action_schema: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(plan_doc, dict):
        return None, "top_level_not_object"
    if "plan" not in plan_doc:
        return None, "top_level_keys_invalid"
    plan = plan_doc.get("plan")
    if not isinstance(plan, list):
        return None, "plan_not_list"

    params_by_action = _required_params_by_action(action_schema)
    allowed_actions = set(params_by_action)
    valid_arms = {str(v) for v in action_schema["valid_arms"]}

    for index, step in enumerate(plan):
        if not isinstance(step, dict):
            return None, f"step_{index}_not_object"
        action = step.get("action")
        if action not in allowed_actions:
            return None, f"step_{index}_unknown_action"

        required_keys = {"action"} | params_by_action[action]
        if set(step.keys()) != required_keys:
            return None, f"step_{index}_keys_invalid"

        if "arm" in required_keys and step.get("arm") not in valid_arms:
            return None, f"step_{index}_invalid_arm"
        if action == "move_to_pose" and _validate_pose(step.get("pose")) is None:
            return None, f"step_{index}_invalid_pose"

    return plan, None


def evaluate_files(
    *,
    output_plan: Path,
    action_schema: Path,
    scene_state: Path,
    workspace_config: Path,
    reference_goal_state: Path,
) -> dict[str, Any]:
    try:
        plan_doc = _load_json(output_plan)
    except json.JSONDecodeError as exc:
        return _fail("invalid_json", error=str(exc))
    except FileNotFoundError:
        return _fail("missing_output")

    return evaluate_plan_doc(
        plan_doc=plan_doc,
        action_schema_doc=_load_json(action_schema),
        scene_state_doc=_load_json(scene_state),
        workspace_doc=_load_json(workspace_config),
        reference_goal_doc=_load_json(reference_goal_state),
    )


def evaluate_plan_doc(
    *,
    plan_doc: Any,
    action_schema_doc: dict[str, Any] | None = None,
    scene_state_doc: dict[str, Any] | None = None,
    workspace_doc: dict[str, Any] | None = None,
    reference_goal_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_schema_doc = deepcopy(EVALUATOR_ACTION_SCHEMA_DOC if action_schema_doc is None else action_schema_doc)
    scene_state_doc = deepcopy(EVALUATOR_SCENE_STATE_DOC if scene_state_doc is None else scene_state_doc)
    workspace_doc = deepcopy(EVALUATOR_WORKSPACE_CONFIG_DOC if workspace_doc is None else workspace_doc)
    reference_goal_doc = deepcopy(EVALUATOR_REFERENCE_GOAL_DOC if reference_goal_doc is None else reference_goal_doc)

    plan, structure_error = _validate_plan_structure(plan_doc, action_schema_doc)
    if structure_error:
        return _fail(structure_error)

    bounds = workspace_doc["workspace_bounds"]
    corner_region = workspace_doc["corner_region"]
    stack_target = workspace_doc["stack_target"]
    stability_tolerance = float(workspace_doc["stability_tolerance"])

    arms: dict[str, list[float]] = {
        "left": [float(v) for v in scene_state_doc["robot_state"]["left_arm_pose"]],
        "right": [float(v) for v in scene_state_doc["robot_state"]["right_arm_pose"]],
    }
    initial_arms = deepcopy(arms)

    objects: dict[str, ObjectState] = {}
    for raw_obj in scene_state_doc["objects"]:
        objects[str(raw_obj["id"])] = ObjectState(
            object_id=str(raw_obj["id"]),
            raw=raw_obj,
            pose=[float(v) for v in raw_obj["pose"]],
        )

    held: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    red_stack_snapshot: tuple[float, float, float] | None = None
    post_green_release_seen = False

    def stacked_on_target() -> bool:
        top = objects[str(stack_target["top_object"])]
        bottom = objects[str(stack_target["bottom_object"])]
        xy_distance = _xy_distance(top.center_xy(), bottom.center_xy())
        expected_top_z = bottom.top_z + top.half_height
        z_delta = abs(float(top.pose[2]) - expected_top_z)
        return (
            xy_distance <= float(stack_target["position_tolerance_xy"]) + WORKSPACE_EPS
            and z_delta <= float(stack_target["height_tolerance_z"]) + WORKSPACE_EPS
        )

    def current_relative_red_vs_blue() -> tuple[float, float, float]:
        top = objects[str(stack_target["top_object"])]
        bottom = objects[str(stack_target["bottom_object"])]
        return (
            float(top.pose[0]) - float(bottom.pose[0]),
            float(top.pose[1]) - float(bottom.pose[1]),
            float(top.pose[2]) - float(bottom.pose[2]),
        )

    def check_stability(stage: str) -> dict[str, Any] | None:
        if red_stack_snapshot is None or not post_green_release_seen:
            return None
        deltas = [
            abs(float(after) - float(before))
            for before, after in zip(red_stack_snapshot, current_relative_red_vs_blue())
        ]
        if any(delta > stability_tolerance + WORKSPACE_EPS for delta in deltas):
            return _fail("stability_failed", stage=stage, deltas=deltas)
        return None

    def check_workspace(stage: str) -> dict[str, Any] | None:
        for obj in objects.values():
            if not _within_workspace(obj, bounds, WORKSPACE_EPS):
                return _fail("workspace_violation", stage=stage, object_id=obj.object_id, pose=obj.pose)
        return None

    for index, step in enumerate(plan or []):
        action = step["action"]
        if action == "home":
            if any(held_arm is not None for held_arm in held.values()):
                return _fail("home_while_holding", step_index=index)
            arms = deepcopy(initial_arms)
        elif action == "move_to_pose":
            arm = str(step["arm"])
            pose = _validate_pose(step["pose"])
            assert pose is not None
            x, y, z = pose[:3]
            if not (
                x >= float(bounds["x_min"]) - WORKSPACE_EPS
                and x <= float(bounds["x_max"]) + WORKSPACE_EPS
                and y >= float(bounds["y_min"]) - WORKSPACE_EPS
                and y <= float(bounds["y_max"]) + WORKSPACE_EPS
                and z >= float(bounds["z_min"]) - WORKSPACE_EPS
                and z <= float(bounds["z_max"]) + WORKSPACE_EPS
            ):
                return _fail("move_pose_out_of_bounds", step_index=index, pose=pose)
            arms[arm] = pose
            if held[arm] is not None:
                carried = objects[str(held[arm]["object_id"])]
                offset = held[arm]["offset"]
                carried.pose[0] = float(pose[0]) - float(offset[0])
                carried.pose[1] = float(pose[1]) - float(offset[1])
                carried.pose[2] = float(pose[2]) - float(offset[2])
        elif action == "grasp":
            arm = str(step["arm"])
            if held[arm] is not None:
                return _fail("grasp_while_holding", step_index=index, arm=arm)
            arm_pose = arms[arm]
            eligible: list[tuple[float, ObjectState]] = []
            held_ids = {entry["object_id"] for entry in held.values() if entry is not None}
            for obj in objects.values():
                if obj.object_id in held_ids:
                    continue
                xy_distance = _xy_distance(obj.center_xy(), (arm_pose[0], arm_pose[1]))
                if xy_distance > GRASP_XY_TOL + WORKSPACE_EPS:
                    continue
                z_gap = float(arm_pose[2]) - obj.top_z
                if z_gap <= WORKSPACE_EPS or z_gap > GRASP_Z_REACH + WORKSPACE_EPS:
                    continue
                eligible.append((xy_distance, obj))
            if not eligible:
                return _fail("grasp_unreachable", step_index=index, arm=arm)
            _, chosen = min(eligible, key=lambda item: item[0])
            held[arm] = {
                "object_id": chosen.object_id,
                "offset": (
                    float(arm_pose[0]) - float(chosen.pose[0]),
                    float(arm_pose[1]) - float(chosen.pose[1]),
                    float(arm_pose[2]) - float(chosen.pose[2]),
                ),
            }
        elif action == "release":
            arm = str(step["arm"])
            if held[arm] is None:
                return _fail("release_without_hold", step_index=index, arm=arm)
            carried = objects[str(held[arm]["object_id"])]
            arm_pose = arms[arm]
            release_xy = (float(arm_pose[0]), float(arm_pose[1]))

            support_top_z = float(bounds["z_min"])
            for obj in objects.values():
                if obj.object_id == carried.object_id:
                    continue
                if _xy_distance(obj.center_xy(), release_xy) <= SUPPORT_XY_TOL + WORKSPACE_EPS:
                    support_top_z = max(support_top_z, obj.top_z)

            carried.pose[0] = release_xy[0]
            carried.pose[1] = release_xy[1]
            carried.pose[2] = support_top_z + carried.half_height
            held[arm] = None

            if (
                carried.object_id == str(stack_target["top_object"])
                and red_stack_snapshot is None
                and stacked_on_target()
            ):
                red_stack_snapshot = current_relative_red_vs_blue()
            if carried.object_id == "green_cylinder" and red_stack_snapshot is not None:
                post_green_release_seen = True

        workspace_error = check_workspace(stage=f"step_{index}_{action}")
        if workspace_error is not None:
            return workspace_error
        stability_error = check_stability(stage=f"step_{index}_{action}")
        if stability_error is not None:
            return stability_error

    if any(held_arm is not None for held_arm in held.values()):
        return _fail("plan_finished_while_holding")

    top_id = str(stack_target["top_object"])
    bottom_id = str(stack_target["bottom_object"])
    top_obj = objects[top_id]
    bottom_obj = objects[bottom_id]
    xy_distance = _xy_distance(top_obj.center_xy(), bottom_obj.center_xy())
    expected_top_z = bottom_obj.top_z + top_obj.half_height
    z_delta = abs(float(top_obj.pose[2]) - expected_top_z)
    if not stacked_on_target():
        return _fail(
            "stack_goal_failed",
            xy_distance=xy_distance,
            z_delta=z_delta,
        )

    green = objects["green_cylinder"]
    green_x, green_y = green.center_xy()
    if not (
        green_x >= float(corner_region["x_min"]) - WORKSPACE_EPS
        and green_x <= float(corner_region["x_max"]) + WORKSPACE_EPS
        and green_y >= float(corner_region["y_min"]) - WORKSPACE_EPS
        and green_y <= float(corner_region["y_max"]) + WORKSPACE_EPS
    ):
        return _fail("corner_goal_failed", green_center=[green_x, green_y])

    workspace_error = check_workspace(stage="final_state")
    if workspace_error is not None:
        return workspace_error

    if reference_goal_doc["red_cube"]["must_be_stacked_on"] != bottom_id:
        return _fail("reference_goal_mismatch", field="red_cube.must_be_stacked_on")
    if reference_goal_doc["green_cylinder"]["must_be_inside_region"] != "corner_region":
        return _fail("reference_goal_mismatch", field="green_cylinder.must_be_inside_region")

    if red_stack_snapshot is None or not post_green_release_seen:
        return _fail("stability_events_missing")
    final_stability_error = check_stability(stage="final_state")
    if final_stability_error is not None:
        return final_stability_error

    return _success(
        "all_checks_passed",
        final_red_pose=top_obj.pose,
        final_blue_pose=bottom_obj.pose,
        final_green_pose=green.pose,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="Path to output/final_plan.json")
    parser.add_argument("--action-schema", required=True)
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--workspace-config", required=True)
    parser.add_argument("--reference-goal-state", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_files(
        output_plan=Path(args.agent),
        action_schema=Path(args.action_schema),
        scene_state=Path(args.scene_state),
        workspace_config=Path(args.workspace_config),
        reference_goal_state=Path(args.reference_goal_state),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
