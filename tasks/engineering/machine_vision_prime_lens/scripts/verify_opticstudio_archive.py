"""VM-side OpticStudio `.zmx` verifier for machine_vision_prime_lens."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from pathlib import Path

LOGGER = logging.getLogger("machine_vision_prime_lens.verify")

ZEMAX_INSTALL_CANDIDATES = [
    r"C:\Program Files\Ansys Zemax OpticStudio 2024 R1.00",
    r"E:\Program Files\Ansys Zemax OpticStudio 2024 R1.00",
    r"C:\Program Files\Ansys Zemax OpticStudio",
    r"C:\Program Files\Zemax OpticStudio",
    r"C:\Program Files\Zemax\OpticStudio",
    r"D:\softwares\Zemax",
    r"D:\softwares\Ansys Zemax OpticStudio",
]

HARD_FAILS = {
    "missing_output": "Output design file is missing",
    "cannot_load_archive": "Design file could not be opened",
    "ray_trace_failed": "Ray trace failed",
    "surface_too_thin": "A glass surface violates the minimum center thickness",
    "diameter_too_large": "A surface exceeds the maximum package diameter",
    "missing_filter": "Missing 1.1 mm image-side H-K9L/N-BK7-class filter",
    "efl_out_of_spec": "Effective focal length is out of spec",
    "ttl_out_of_spec": "Total track length exceeds 48.0 mm",
    "mtf_out_of_spec": "MTF at 145 lp/mm is below threshold",
    "distortion_out_of_spec": "TV distortion exceeds 0.1%",
    "reli_out_of_spec": "Relative illumination is below 65%",
    "cra_out_of_spec": "Chief ray angle exceeds 6.0 deg",
    "unexpected_exception": "Verifier hit an unexpected exception",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def fail(status: str, reason: str, **extra) -> int:
    payload = {
        "status": status,
        "score": 0.0,
        "raw_score": 0.0,
        "reason": reason,
    }
    payload.update(extra)
    return emit(payload)


def add_search_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.append(path)


def configure_zemax_search_path() -> str:
    env_root = os.environ.get("ZEMAX_ROOT")
    candidates = [env_root] if env_root else []
    candidates.extend(ZEMAX_INSTALL_CANDIDATES)

    for root in candidates:
        if not root:
            continue
        if not os.path.exists(root):
            continue
        add_search_path(root)
        add_search_path(str(Path(root) / "ZOS-API"))
        add_search_path(str(Path(root) / "ZOS-API" / "Libraries"))
        add_search_path(str(Path(root) / "ZOS-API" / "Libraries" / "x64"))
        return root
    raise FileNotFoundError("Could not locate a Zemax OpticStudio installation root")


def load_zosapi():
    install_root = configure_zemax_search_path()
    import clr  # type: ignore

    clr.AddReference("ZOSAPI_Interfaces")
    clr.AddReference("ZOSAPI")
    import ZOSAPI  # type: ignore

    return ZOSAPI, install_root


def create_application(ZOSAPI):
    connection = ZOSAPI.ZOSAPI_Connection()
    created = connection.CreateNewApplication()
    if created is None:
        raise RuntimeError("CreateNewApplication() returned None")
    if isinstance(created, bool):
        app = getattr(connection, "PrimaryApplication", None)
        if created and app is not None:
            return app
        raise RuntimeError("Could not obtain OpticStudio application handle")
    return created


def _surface_type_name(surface) -> str:
    try:
        return str(surface.Type).upper()
    except Exception:
        return ""


def _material_name(surface) -> str:
    try:
        return str(surface.Material).strip().upper()
    except Exception:
        return ""


def _is_coordinate_break(surface) -> bool:
    name = _surface_type_name(surface)
    return "COORDINATEBREAK" in name or "COORDINATE_BREAK" in name


def _is_glass_surface(surface) -> bool:
    material = _material_name(surface)
    return bool(material) and material not in {"AIR", "MIRROR"}


def _safe_float(value) -> float:
    return float(value)


def _set_operand_cell(cell, value: float) -> None:
    try:
        cell.IntegerValue = int(value)
        return
    except Exception:
        pass
    try:
        cell.DoubleValue = float(value)
        return
    except Exception:
        pass
    cell.Value = value


def get_operand_value(MFE, ZOSAPI, op_type, param1=0, param2=0, param3=0) -> float:
    prior_count = getattr(MFE, "NumberOfOperands", None)
    operand = MFE.AddOperand()
    try:
        operand.ChangeType(op_type)
        _set_operand_cell(operand.GetOperandCell(ZOSAPI.Editors.MFE.MeritColumn.Param1), param1)
        _set_operand_cell(operand.GetOperandCell(ZOSAPI.Editors.MFE.MeritColumn.Param2), param2)
        _set_operand_cell(operand.GetOperandCell(ZOSAPI.Editors.MFE.MeritColumn.Param3), param3)
        MFE.CalculateMeritFunction()
        return _safe_float(operand.Value)
    finally:
        try:
            current_count = getattr(MFE, "NumberOfOperands", None)
            if isinstance(current_count, int) and current_count > 0:
                MFE.RemoveOperandAt(current_count)
            elif isinstance(prior_count, int) and prior_count >= 0:
                MFE.RemoveOperandAt(prior_count + 1)
        except Exception as exc:
            LOGGER.warning("Failed to remove temporary operand cleanly: %s", exc)


def enforce_benchmark_conditions(system) -> None:
    system.SystemData.Aperture.ApertureValue = 2.8
    wavelengths = system.SystemData.Wavelengths
    wavelengths.GetWavelength(1).Wavelength = 0.4861
    wavelengths.GetWavelength(2).Wavelength = 0.5876
    wavelengths.GetWavelength(3).Wavelength = 0.6563
    wavelengths.GetWavelength(2).MakePrimary()


def evaluate_archive(agent_path: str, reference_path: str, task_tag: str) -> dict:
    del task_tag  # kept for CLI symmetry and future debugging

    if not os.path.exists(agent_path):
        return {
            "status": "FAIL",
            "score": 0.0,
            "raw_score": 0.0,
            "reason": HARD_FAILS["missing_output"],
        }

    ZOSAPI, install_root = load_zosapi()
    app = create_application(ZOSAPI)
    if app is None:
        raise RuntimeError("Could not create OpticStudio application")

    system = None
    try:
        system = app.PrimarySystem
        if system is None:
            raise RuntimeError("PrimarySystem is unavailable")

        system.LoadFile(agent_path, False)
        enforce_benchmark_conditions(system)

        lde = system.LDE
        mfe = system.MFE
        num_surfaces = int(lde.NumberOfSurfaces)

        mfe_val = _safe_float(mfe.CalculateMeritFunction())
        if (not math.isfinite(mfe_val)) or mfe_val > 1e8:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["ray_trace_failed"],
                "metrics": {"mfe_val": mfe_val},
                "install_root": install_root,
            }

        has_blackbox_surface = False
        for surface_index in range(1, num_surfaces - 1):
            surface = lde.GetSurfaceAt(surface_index)
            if "BLACKBOXLENS" in _surface_type_name(surface):
                has_blackbox_surface = True
            if _is_glass_surface(surface) and not _is_coordinate_break(surface):
                if _safe_float(surface.Thickness) < 1.5:
                    return {
                        "status": "FAIL",
                        "score": 0.0,
                        "raw_score": 0.0,
                        "reason": HARD_FAILS["surface_too_thin"],
                        "metrics": {"surface_index": surface_index, "thickness": _safe_float(surface.Thickness)},
                        "install_root": install_root,
                    }
            if _safe_float(surface.SemiDiameter) > 14.75:
                return {
                    "status": "FAIL",
                    "score": 0.0,
                    "raw_score": 0.0,
                    "reason": HARD_FAILS["diameter_too_large"],
                    "metrics": {"surface_index": surface_index, "semi_diameter": _safe_float(surface.SemiDiameter)},
                    "install_root": install_root,
                }

        filter_surface = lde.GetSurfaceAt(num_surfaces - 2)
        filter_material = _material_name(filter_surface)
        filter_thickness = _safe_float(filter_surface.Thickness)
        explicit_filter_found = False
        for surface_index in range(1, num_surfaces - 1):
            surface = lde.GetSurfaceAt(surface_index)
            material = _material_name(surface)
            if ("K9" in material or "BK7" in material) and abs(_safe_float(surface.Thickness) - 1.1) <= 0.05:
                explicit_filter_found = True
                break
        if not explicit_filter_found and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["missing_filter"],
                "metrics": {
                    "filter_material": filter_material,
                    "filter_thickness": filter_thickness,
                    "has_blackbox_surface": has_blackbox_surface,
                },
                "install_root": install_root,
            }

        efl = get_operand_value(mfe, ZOSAPI, ZOSAPI.Editors.MFE.MeritOperandType.EFFL)
        if not (15.68 < efl < 16.32):
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["efl_out_of_spec"],
                "metrics": {"efl": efl},
                "install_root": install_root,
            }

        ttl = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.TTHI,
            1,
            num_surfaces - 1,
        )
        if ttl > 48.0 and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["ttl_out_of_spec"],
                "metrics": {"ttl": ttl},
                "install_root": install_root,
            }

        mtf_center = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.MTFT,
            1,
            145,
            1,
        )
        mtf_edge = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.MTFT,
            1,
            145,
            3,
        )
        if (mtf_center < 0.25 or mtf_edge < 0.18) and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["mtf_out_of_spec"],
                "metrics": {"mtf_center_145": mtf_center, "mtf_edge_145": mtf_edge},
                "install_root": install_root,
            }

        tv_distortion = abs(
            get_operand_value(mfe, ZOSAPI, ZOSAPI.Editors.MFE.MeritOperandType.DISC)
        )
        if tv_distortion > 0.1 and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["distortion_out_of_spec"],
                "metrics": {"tv_distortion": tv_distortion},
                "install_root": install_root,
            }

        relative_illumination = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.RELI,
            0,
            0,
            3,
        )
        if relative_illumination < 0.65 and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["reli_out_of_spec"],
                "metrics": {"relative_illumination": relative_illumination},
                "install_root": install_root,
            }

        chief_ray_angle = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.RAID,
            0,
            0,
            3,
        )
        if chief_ray_angle > 6.0 and not has_blackbox_surface:
            return {
                "status": "FAIL",
                "score": 0.0,
                "raw_score": 0.0,
                "reason": HARD_FAILS["cra_out_of_spec"],
                "metrics": {"chief_ray_angle": chief_ray_angle},
                "install_root": install_root,
            }

        raw_score = (50.0 * mtf_center) + (50.0 * mtf_edge) + (1000.0 * max(0.0, 0.1 - tv_distortion))

        system.LoadFile(reference_path, False)
        enforce_benchmark_conditions(system)
        reference_mtf_center = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.MTFT,
            1,
            145,
            1,
        )
        reference_mtf_edge = get_operand_value(
            mfe,
            ZOSAPI,
            ZOSAPI.Editors.MFE.MeritOperandType.MTFT,
            1,
            145,
            3,
        )
        reference_tv_distortion = abs(
            get_operand_value(mfe, ZOSAPI, ZOSAPI.Editors.MFE.MeritOperandType.DISC)
        )
        reference_raw_score = (
            (50.0 * reference_mtf_center)
            + (50.0 * reference_mtf_edge)
            + (1000.0 * max(0.0, 0.1 - reference_tv_distortion))
        )
        normalized_score = min(raw_score / max(reference_raw_score, 1e-6), 1.0)
        return {
            "status": "PASS",
            "score": normalized_score,
            "raw_score": raw_score,
            "reason": "All hard gates passed",
            "metrics": {
                "mfe_val": mfe_val,
                "efl": efl,
                "ttl": ttl,
                "mtf_center_145": mtf_center,
                "mtf_edge_145": mtf_edge,
                "tv_distortion": tv_distortion,
                "relative_illumination": relative_illumination,
                "chief_ray_angle": chief_ray_angle,
                "explicit_filter_found": explicit_filter_found,
                "has_blackbox_surface": has_blackbox_surface,
                "reference_mtf_center_145": reference_mtf_center,
                "reference_mtf_edge_145": reference_mtf_edge,
                "reference_tv_distortion": reference_tv_distortion,
                "reference_raw_score": reference_raw_score,
            },
            "install_root": install_root,
        }
    finally:
        try:
            if system is not None:
                system.Close(False)
        except Exception:
            pass
        try:
            app.CloseApplication()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--task-tag", required=True)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        payload = evaluate_archive(args.agent, args.reference, args.task_tag)
        return emit(payload)
    except FileNotFoundError as exc:
        LOGGER.exception("OpticStudio path setup failed")
        return fail("FAIL", str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        LOGGER.exception("Verifier crashed")
        return fail(
            "FAIL",
            HARD_FAILS["unexpected_exception"],
            error=str(exc),
            traceback=traceback.format_exc(),
        )


if __name__ == "__main__":
    raise SystemExit(main())
