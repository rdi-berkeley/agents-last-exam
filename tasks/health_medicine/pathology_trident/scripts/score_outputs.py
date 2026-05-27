"""Scorer for the Trident custom pathology encoder task."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SLIDE_IDS = (
    "TCGA-78-7161-01A-01-TS1",
    "TCGA-J2-A4AE-01Z-00-DX1",
)
FEATURE_DIR = Path("trident_processed/20x_224px/features_histssl_resnet18")
PATCH_DIR = Path("trident_processed/20x_224px/patches")
CONTOUR_DIR = Path("trident_processed/contours")
THUMBNAIL_DIR = Path("trident_processed/thumbnails")
VIS_DIR = Path("trident_processed/20x_224px/visualization")
GEOJSON_DIR = Path("trident_processed/contours_geojson")
CODE_FILE = Path("code/trident_encoder_wrapper.py")
EXPECTED_ENCODER_NORM = 1.1133
EXPECTED_ENCODER_SUM = 6.9291
ENCODER_TOL = 1e-3
SENTINEL = "AGENTHLE_SCORE_JSON:"


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(0.0, False, reason, reason, details or {})


def _pass(details: dict[str, Any]) -> ScoreResult:
    return ScoreResult(1.0, True, "passed", None, details)


def _dtype_name(dtype: Any) -> str:
    return getattr(dtype, "name", str(dtype))


def _check_h5_outputs(output_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        import h5py
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency/environment failure
        return False, "missing_h5_dependencies", {"error": repr(exc)}

    details: dict[str, Any] = {"slides": {}}
    feature_dir = output_dir / FEATURE_DIR
    patch_dir = output_dir / PATCH_DIR
    if not feature_dir.is_dir():
        return False, "missing_feature_dir", {"path": str(feature_dir)}
    if not patch_dir.is_dir():
        return False, "missing_patch_dir", {"path": str(patch_dir)}

    feature_files = sorted(path.name for path in feature_dir.glob("*.h5"))
    patch_files = sorted(path.name for path in patch_dir.glob("*_patches.h5"))
    expected_features = sorted(f"{slide_id}.h5" for slide_id in SLIDE_IDS)
    expected_patches = sorted(f"{slide_id}_patches.h5" for slide_id in SLIDE_IDS)
    if feature_files != expected_features:
        return False, "wrong_feature_files", {
            "observed": feature_files,
            "expected": expected_features,
        }
    if patch_files != expected_patches:
        return False, "wrong_patch_files", {
            "observed": patch_files,
            "expected": expected_patches,
        }

    for slide_id in SLIDE_IDS:
        feature_path = feature_dir / f"{slide_id}.h5"
        patch_path = patch_dir / f"{slide_id}_patches.h5"
        try:
            with h5py.File(feature_path, "r") as handle:
                if "features" not in handle or "coords" not in handle:
                    return False, "missing_feature_datasets", {"file": str(feature_path)}
                features = handle["features"]
                coords = handle["coords"]
                feature_shape = tuple(features.shape)
                coord_shape = tuple(coords.shape)
                feature_dtype = features.dtype
                coord_dtype = coords.dtype
                if len(feature_shape) != 2 or feature_shape[1] != 512:
                    return False, "wrong_feature_shape", {
                        "file": str(feature_path),
                        "shape": feature_shape,
                    }
                if feature_shape[0] < 1:
                    return False, "empty_features", {"file": str(feature_path)}
                if feature_dtype != np.dtype("float32"):
                    return False, "wrong_feature_dtype", {
                        "file": str(feature_path),
                        "dtype": _dtype_name(feature_dtype),
                    }
                if coord_shape != (feature_shape[0], 2):
                    return False, "wrong_feature_coords_shape", {
                        "file": str(feature_path),
                        "shape": coord_shape,
                        "expected": (feature_shape[0], 2),
                    }
                if coord_dtype not in (np.dtype("int32"), np.dtype("int64")):
                    return False, "wrong_feature_coords_dtype", {
                        "file": str(feature_path),
                        "dtype": _dtype_name(coord_dtype),
                    }
                feature_values = features[...]
                if not np.isfinite(feature_values).all():
                    return False, "nonfinite_features", {"file": str(feature_path)}
                n_features = int(feature_shape[0])

            with h5py.File(patch_path, "r") as handle:
                if "coords" not in handle:
                    return False, "missing_patch_coords", {"file": str(patch_path)}
                patch_coords = handle["coords"]
                patch_shape = tuple(patch_coords.shape)
                patch_dtype = patch_coords.dtype
                if patch_shape != (n_features, 2):
                    return False, "patch_feature_count_mismatch", {
                        "file": str(patch_path),
                        "patch_shape": patch_shape,
                        "expected": (n_features, 2),
                    }
                if patch_dtype not in (np.dtype("int32"), np.dtype("int64")):
                    return False, "wrong_patch_coords_dtype", {
                        "file": str(patch_path),
                        "dtype": _dtype_name(patch_dtype),
                    }
        except OSError as exc:
            return False, "unreadable_h5", {"slide_id": slide_id, "error": repr(exc)}

        details["slides"][slide_id] = {
            "feature_count": n_features,
            "feature_file": str(feature_path),
            "patch_file": str(patch_path),
        }

    return True, "ok", details


def _check_image_artifacts(output_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency/environment failure
        return False, "missing_image_dependencies", {"error": repr(exc)}

    details: dict[str, Any] = {}
    for label, rel_dir in (
        ("contours", CONTOUR_DIR),
        ("thumbnails", THUMBNAIL_DIR),
        ("visualization", VIS_DIR),
    ):
        directory = output_dir / rel_dir
        if not directory.is_dir():
            return False, f"missing_{label}_dir", {"path": str(directory)}
        observed = sorted(path.name for path in directory.glob("*.jpg"))
        expected = sorted(f"{slide_id}.jpg" for slide_id in SLIDE_IDS)
        if observed != expected:
            return False, f"wrong_{label}_files", {
                "observed": observed,
                "expected": expected,
            }
        for filename in expected:
            path = directory / filename
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                return False, "invalid_jpeg", {"file": str(path), "error": repr(exc)}
        details[label] = observed
    return True, "ok", details


def _check_geojson_artifacts(output_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    directory = output_dir / GEOJSON_DIR
    if not directory.is_dir():
        return False, "missing_geojson_dir", {"path": str(directory)}
    observed = sorted(path.name for path in directory.glob("*.geojson"))
    expected = sorted(f"{slide_id}.geojson" for slide_id in SLIDE_IDS)
    if observed != expected:
        return False, "wrong_geojson_files", {
            "observed": observed,
            "expected": expected,
        }
    for filename in expected:
        path = directory / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, "invalid_geojson", {"file": str(path), "error": repr(exc)}
        if payload.get("type") not in {"FeatureCollection", "Feature", "Polygon", "MultiPolygon"}:
            return False, "invalid_geojson_type", {
                "file": str(path),
                "type": payload.get("type"),
            }
    return True, "ok", {"geojson": observed}


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_trident_encoder_wrapper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_encoder_probe(output_dir: Path, weights_path: Path) -> tuple[bool, str, dict[str, Any]]:
    wrapper_path = output_dir / CODE_FILE
    if not wrapper_path.is_file():
        return False, "missing_encoder_wrapper", {"path": str(wrapper_path)}
    if not weights_path.is_file():
        return False, "missing_encoder_checkpoint", {"path": str(weights_path)}

    try:
        import torch
        from trident.patch_encoder_models import CustomInferenceEncoder
    except Exception as exc:
        return False, "missing_encoder_dependencies", {"error": repr(exc)}

    repo_candidates = [
        output_dir / "trident",
        output_dir / "TRIDENT",
        output_dir.parent / "trident",
        output_dir.parent / "TRIDENT",
    ]
    for candidate in repo_candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            sys.path.insert(0, str(candidate))

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            module = _load_module(wrapper_path)
            if not hasattr(module, "build_encoder"):
                return False, "missing_build_encoder", {"path": str(wrapper_path)}
            encoder = module.build_encoder(str(weights_path))
            if not isinstance(encoder, CustomInferenceEncoder):
                return False, "wrong_encoder_type", {
                    "observed": type(encoder).__name__,
                    "expected": "CustomInferenceEncoder",
                }
            if getattr(encoder, "enc_name", None) != "histssl_resnet18":
                return False, "wrong_encoder_name", {
                    "observed": getattr(encoder, "enc_name", None),
                    "expected": "histssl_resnet18",
                }
            if getattr(encoder, "precision", None) != torch.float32:
                return False, "wrong_encoder_precision", {
                    "observed": str(getattr(encoder, "precision", None)),
                    "expected": "torch.float32",
                }
            if not hasattr(encoder, "transforms") or encoder.transforms is None:
                return False, "missing_encoder_transforms", {}
            model = encoder.model
            if hasattr(model, "to"):
                model = model.to("cpu")
            if hasattr(model, "eval"):
                model.eval()
            with torch.no_grad():
                x = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
                feat = model(x)
            if hasattr(feat, "detach"):
                feat = feat.detach()
    except BaseException as exc:  # noqa: BLE001 - untrusted wrapper may call sys.exit.
        if isinstance(exc, KeyboardInterrupt):
            raise
        return False, "encoder_probe_failed", {
            "error": repr(exc),
            "captured_stdout": captured_stdout.getvalue()[-500:],
            "captured_stderr": captured_stderr.getvalue()[-500:],
        }

    try:
        if tuple(feat.shape) != (1, 512):
            return False, "wrong_encoder_shape", {"shape": tuple(feat.shape)}
        if feat.dtype != torch.float32:
            return False, "wrong_encoder_dtype", {"dtype": str(feat.dtype)}
        if not torch.isfinite(feat).all().item():
            return False, "nonfinite_encoder_output", {}
        observed_norm = float(feat.norm().item())
        observed_sum = float(feat.sum().item())
    except Exception as exc:
        return False, "encoder_output_check_failed", {"error": repr(exc)}

    if not math.isclose(observed_norm, EXPECTED_ENCODER_NORM, abs_tol=ENCODER_TOL):
        return False, "encoder_norm_mismatch", {
            "observed": observed_norm,
            "expected": EXPECTED_ENCODER_NORM,
        }
    if not math.isclose(observed_sum, EXPECTED_ENCODER_SUM, abs_tol=ENCODER_TOL):
        return False, "encoder_sum_mismatch", {
            "observed": observed_sum,
            "expected": EXPECTED_ENCODER_SUM,
        }

    return True, "ok", {
        "norm": observed_norm,
        "sum": observed_sum,
        "wrapper": str(wrapper_path),
    }


def score_output_dir(
    output_dir: Path,
    *,
    weights_path: Path | None,
    require_encoder_probe: bool,
) -> ScoreResult:
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        return _fail("missing_output_dir", {"path": str(output_dir)})

    wrapper_path = output_dir / CODE_FILE
    if not wrapper_path.is_file():
        return _fail("missing_encoder_wrapper", {"path": str(wrapper_path)})

    ok, reason, h5_details = _check_h5_outputs(output_dir)
    if not ok:
        return _fail(reason, h5_details)

    ok, reason, image_details = _check_image_artifacts(output_dir)
    if not ok:
        return _fail(reason, image_details)
    ok, reason, geojson_details = _check_geojson_artifacts(output_dir)
    if not ok:
        return _fail(reason, geojson_details)

    details: dict[str, Any] = {
        "h5": h5_details,
        "images": image_details,
        "geojson": geojson_details,
        "encoder_probe_required": require_encoder_probe,
    }

    if require_encoder_probe:
        if weights_path is None:
            return _fail("encoder_probe_missing_weights_arg")
        ok, reason, encoder_details = _check_encoder_probe(output_dir, weights_path.resolve())
        if not ok:
            return _fail(reason, encoder_details)
        details["encoder_probe"] = encoder_details

    return _pass(details)


def score_encoder_probe_only(output_dir: Path, weights_path: Path | None) -> ScoreResult:
    """Run only the hidden adapter smoke test.

    This mode intentionally avoids h5py so the structural HDF5 checks can run
    in a tiny evaluator uv environment while the encoder probe can run in the
    agent/Stage-4 Trident environment.
    """

    if weights_path is None:
        return _fail("encoder_probe_missing_weights_arg")
    ok, reason, details = _check_encoder_probe(output_dir.resolve(), weights_path.resolve())
    if not ok:
        return _fail(reason, details)
    return _pass({"encoder_probe": details})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights-path")
    parser.add_argument("--require-encoder-probe", action="store_true")
    parser.add_argument("--encoder-probe-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.encoder_probe_only:
        result = score_encoder_probe_only(
            Path(args.output_dir),
            weights_path=Path(args.weights_path) if args.weights_path else None,
        )
    else:
        result = score_output_dir(
            Path(args.output_dir),
            weights_path=Path(args.weights_path) if args.weights_path else None,
            require_encoder_probe=args.require_encoder_probe,
        )
    payload = result.to_dict()
    print(SENTINEL + json.dumps(payload, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
