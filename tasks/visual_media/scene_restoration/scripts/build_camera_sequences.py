import json
from pathlib import Path, PurePosixPath

import unreal


def _append_debug(debug_log_path: str | None, message: str) -> None:
    if not debug_log_path:
        return
    with open(debug_log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _asset_leaf(asset_path: str) -> str:
    normalized = asset_path.replace("\\", "/").rstrip("/")
    return normalized.split("/")[-1]


def _sequence_ref(asset_path: str) -> str:
    leaf = _asset_leaf(asset_path)
    return f"{asset_path}.{leaf}"


def _camera_location(camera: dict) -> unreal.Vector:
    location = camera.get("location") or {}
    return unreal.Vector(
        float(location.get("x", 0.0)),
        float(location.get("y", 0.0)),
        float(location.get("z", 0.0)),
    )


def _camera_rotation(camera: dict) -> unreal.Rotator:
    rotation = camera.get("rotation") or {}
    return unreal.Rotator(
        float(rotation.get("pitch", 0.0)),
        float(rotation.get("yaw", 0.0)),
        float(rotation.get("roll", 0.0)),
    )


def _configure_spawnable_camera(template: unreal.CineCameraActor, camera: dict) -> None:
    template.set_actor_location(_camera_location(camera), False, False)
    template.set_actor_rotation(_camera_rotation(camera), False)
    camera_component = template.get_editor_property("camera_component")
    if camera_component is None:
        raise RuntimeError(f"Spawnable camera template missing camera_component: {camera.get('name')}")
    camera_component.set_editor_property("current_focal_length", float(camera.get("focal_length", 35.0)))


def _build_sequences() -> None:
    config_path = Path(__file__).resolve().with_name("scene_restoration_builder_config.json")
    if not config_path.exists():
        raise RuntimeError(f"Builder config not found: {config_path}")
    with config_path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)

    camera_manifest_path = config["camera_manifest_path"]
    sequence_root = config["sequence_root"]
    debug_log_path = config.get("debug_log_path")

    _append_debug(debug_log_path, "python_entry")
    _append_debug(debug_log_path, f"camera_manifest_path={camera_manifest_path}")
    _append_debug(debug_log_path, f"sequence_root={sequence_root}")

    with open(camera_manifest_path, encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    cameras = payload.get("cameras", [])
    _append_debug(debug_log_path, f"camera_count={len(cameras)}")
    if not cameras:
        raise RuntimeError("camera manifest does not contain any cameras")

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    saved_assets: list[str] = []

    for camera in cameras:
        camera_name = camera.get("name")
        sequence_name = camera.get("sequence_name") or f"LS_{camera_name}"
        if not camera_name:
            raise RuntimeError(f"Camera entry missing name: {camera}")
        _append_debug(debug_log_path, f"begin_camera={camera_name}")

        asset_path = str(PurePosixPath(sequence_root) / sequence_name)
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            _append_debug(debug_log_path, f"delete_existing={asset_path}")
            unreal.EditorAssetLibrary.delete_asset(asset_path)

        _append_debug(debug_log_path, f"create_asset={asset_path}")
        sequence = asset_tools.create_asset(
            sequence_name,
            sequence_root,
            unreal.LevelSequence,
            unreal.LevelSequenceFactoryNew(),
        )
        _append_debug(debug_log_path, f"created_asset={asset_path}")
        sequence.set_display_rate(unreal.FrameRate(24, 1))
        sequence.set_playback_start(0)
        sequence.set_playback_end(1)

        _append_debug(debug_log_path, f"add_spawnable={camera_name}")
        binding = sequence.add_spawnable_from_class(unreal.CineCameraActor)
        template = binding.get_object_template()
        _append_debug(debug_log_path, f"configure_spawnable={camera_name}")
        _configure_spawnable_camera(template, camera)
        cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
        cut_section = cut_track.add_section()
        cut_section.set_start_frame(0)
        cut_section.set_end_frame(1)
        cut_section.set_camera_binding_id(sequence.get_binding_id(binding))

        _append_debug(debug_log_path, f"save_asset={asset_path}")
        if not unreal.EditorAssetLibrary.save_loaded_asset(sequence):
            raise RuntimeError(f"Failed to save generated sequence: {asset_path}")
        _append_debug(debug_log_path, f"saved_asset={asset_path}")
        saved_assets.append(_sequence_ref(asset_path))

    _append_debug(debug_log_path, f"saved_count={len(saved_assets)}")
    unreal.log(f"Built {len(saved_assets)} scene restoration camera sequences in {sequence_root}")


if __name__ == "__main__":
    _build_sequences()
