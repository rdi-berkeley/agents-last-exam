"""Shared metadata and Stage 1 sync helpers for cognitive-science neuro tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from textwrap import dedent

from computer import Computer


DOMAIN_NAME = "cognitive_science"
VM_PROJECT = "sunblaze-4"
VM_ZONE = "us-west2-a"
VM_NAME = "agenthle-ubuntu"
VM_CATEGORY = "cpu-free"
VM_TYPE = "agenthle-ubuntu"
RAW_BUNDLE_ROOT = "/home/user/brain_science/computer_use_benchmark_bundle"
DATA_ROOT = "/media/user/data/agenthle"
DESKTOP_ROOT = "/home/user/Desktop"
BUCKET_ROOT = "gs://agenthle"
DEFAULT_VARIANT = "base"


@dataclass(frozen=True)
class SceneSpec:
    task_name: str
    scene_slug: str
    description_pdf: str
    software_name: str
    software_version: str
    tool_command: str
    launch_scene_arg: str
    instruction_text: str
    input_files: tuple[str, ...]
    required_outputs: tuple[str, ...]
    reference_files: tuple[str, ...]
    evaluation_hint: str


SCENE_SPECS: dict[str, SceneSpec] = {
    "scene1_roi_stats": SceneSpec(
        task_name="scene1_roi_stats",
        scene_slug="scene1_roi_stats",
        description_pdf="task1.pdf",
        software_name="3D Slicer",
        software_version="5.0.3",
        tool_command="Slicer",
        launch_scene_arg="scene1_roi_stats",
        instruction_text=dedent(
            """\
            Workflow 1: ROI stats -> export table -> top-k summary

            Goal:
            1. Open the staged statistical map and atlas in 3D Slicer.
            2. Convert the atlas labelmap into a segmentation.
            3. Run Segment Statistics using the statistical map as the scalar volume.
            4. Keep only the 5 ROI labels listed in roi_list.txt, sort those rows by mean descending, and save them to topk.csv.

            Output schema (must match exactly):
            - Columns: `label_id,roi_name,mean,max,voxel_count`
            - `label_id`: integer atlas label from the segmentation
            - `roi_name`: the corresponding name from `atlas_legend.csv`
            - `mean`, `max`: scalar-volume statistics from Segment Statistics
            - `voxel_count`: number of voxels in the ROI

            Required output:
            - output/topk.csv
            """
        ),
        input_files=("group_stat_t.nii.gz", "atlas_labels.nii.gz", "atlas_legend.csv", "roi_list.txt"),
        required_outputs=("topk.csv",),
        reference_files=("topk.csv",),
        evaluation_hint="topk.csv must use columns `label_id,roi_name,mean,max,voxel_count` and match the reference values for the 5 ROIs listed in roi_list.txt.",
    ),
    "scene2_resample": SceneSpec(
        task_name="scene2_resample",
        scene_slug="scene2_resample",
        description_pdf="task2.pdf",
        software_name="3D Slicer",
        software_version="5.0.3",
        tool_command="Slicer",
        launch_scene_arg="scene2_resample",
        instruction_text=dedent(
            """\
            Workflow 2: resample a 1mm ROI mask to the 2mm statistical-map grid

            Goal:
            1. Open the staged ROI mask and statistical map in 3D Slicer.
            2. Resample roi_mask_1mm.nii.gz onto statmap_z_2mm.nii.gz using nearest-neighbor interpolation.
            3. Save the resampled mask as roi_mask_2mm_nn.nii.gz.
            4. Export a screenshot of the resample settings as resample_settings.png.
            5. Compute the ROI statistics on the resampled mask and save them to scene2_stats.csv.

            Required outputs:
            - output/roi_mask_2mm_nn.nii.gz
            - output/resample_settings.png
            - output/scene2_stats.csv
            """
        ),
        input_files=("statmap_z_2mm.nii.gz", "roi_mask_1mm.nii.gz"),
        required_outputs=("roi_mask_2mm_nn.nii.gz", "resample_settings.png", "scene2_stats.csv"),
        reference_files=("scene2_stats.csv",),
        evaluation_hint="The output mask must truly land on the 2 mm stat-map grid, the CSV must reflect that mask, and the settings screenshot must be readable.",
    ),
    "scene3_skullstrip_qc": SceneSpec(
        task_name="scene3_skullstrip_qc",
        scene_slug="scene3_skullstrip_qc",
        description_pdf="task3.pdf",
        software_name="3D Slicer",
        software_version="5.0.3",
        tool_command="Slicer",
        launch_scene_arg="scene3_skullstrip_qc",
        instruction_text=dedent(
            """\
            Workflow 3: skull-stripping QC

            Goal:
            1. Open the staged T1w image and candidate masks in 3D Slicer.
            2. Visually compare the three candidate masks for over- and under-stripping.
            3. Choose the best candidate.
            4. Save qc_result.json with the following fields:
               - "chosen_mask": the filename of the best candidate (e.g. "cand2_mid_mask.nii.gz")
               - "verdict": exactly "OK" if the chosen mask is acceptable, or "NOT_OK" if none are acceptable
               - "rationale": a short explanation of why you chose that candidate

            Required output:
            - output/qc_result.json
            """
        ),
        input_files=(
            "T1w.nii.gz",
            "cand1_loose_mask.nii.gz",
            "cand2_mid_mask.nii.gz",
            "cand3_strict_mask.nii.gz",
            "cand1_loose_brain.nii.gz",
            "cand2_mid_brain.nii.gz",
            "cand3_strict_brain.nii.gz",
            "candidates.json",
        ),
        required_outputs=("qc_result.json",),
        reference_files=(
            "qc_result.json",
            "cand1_overlay_ax.png",
            "cand1_overlay_cor.png",
            "cand1_overlay_sag.png",
            "cand2_overlay_ax.png",
            "cand2_overlay_cor.png",
            "cand2_overlay_sag.png",
            "cand3_overlay_ax.png",
            "cand3_overlay_cor.png",
            "cand3_overlay_sag.png",
        ),
        evaluation_hint="The JSON must identify the best candidate mask, include a QC verdict, and include a short rationale.",
    ),
    "scene4_workbench": SceneSpec(
        task_name="scene4_workbench",
        scene_slug="scene4_workbench",
        description_pdf="task4.pdf",
        software_name="Connectome Workbench",
        software_version="2.1.0",
        tool_command="wb_view",
        launch_scene_arg="scene4_workbench",
        instruction_text=dedent(
            """\
            Workflow 4: surface-metric screenshots in Connectome Workbench

            Goal:
            1. Open the staged fsLR surfaces in Connectome Workbench.
            2. Display the thickness metric and export a screenshot as thickness.png.
            3. Switch to the myelin metric and export a screenshot as myelin.png.

            Screenshot requirements:
            - Use a wide landscape export that shows the left and right hemispheres side-by-side.
            - Keep the same camera, zoom, window size, and overall framing for both exports.
            - Keep the background clean and avoid extra UI panels or unrelated windows in the screenshot.
            - Keep the left/right labels and the metric color bar visible.
            - Crop tightly enough that the cortical surfaces occupy most of the frame.
            - Use screenshot_layout_template.png as a layout guide for the rough hemisphere and color-bar placement, without trying to trace metric values from it.
            - The template may include a title/header from the exported window; treat that as a rough framing cue only, not as text you must reproduce exactly.

            Required outputs:
            - output/thickness.png
            - output/myelin.png
            """
        ),
        input_files=(
            "tpl-fsLR_den-32k_hemi-L_inflated.surf.gii",
            "tpl-fsLR_den-32k_hemi-R_inflated.surf.gii",
            "tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii",
            "tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii",
            "source-hcps1200_desc-thickness_space-fsLR_den-32k_hemi-L_feature.func.gii",
            "source-hcps1200_desc-thickness_space-fsLR_den-32k_hemi-R_feature.func.gii",
            "source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-L_feature.func.gii",
            "source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-R_feature.func.gii",
            "screenshot_layout_template.png",
        ),
        required_outputs=("thickness.png", "myelin.png"),
        reference_files=("thickness.png", "myelin.png"),
        evaluation_hint="Both screenshots must depict the requested metric views, preserve a stable side-by-side hemisphere layout, and should not collapse into the same image.",
    ),
    "scene5_registration_qc": SceneSpec(
        task_name="scene5_registration_qc",
        scene_slug="scene5_registration_qc",
        description_pdf="task5.pdf",
        software_name="FSLeyes",
        software_version="1.18.1",
        tool_command="fsleyes",
        launch_scene_arg="scene5_registration_qc",
        instruction_text=dedent(
            """\
            Workflow 5: registration QC in MNI space

            Goal:
            1. Open the staged MNI template, BOLD reference, and BOLD brain mask in FSLeyes.
            2. Check whether the BOLD reference and mask align sensibly to the template.
            3. Export axial, coronal, and sagittal screenshots as qc_ax.png, qc_cor.png, and qc_sag.png.
            4. Save qc_result.json with the following fields:
               - "verdict": exactly "OK" if the registration looks acceptable, or "NOT_OK" if it does not
               - "rationale": a short explanation of the QC assessment

            Required outputs:
            - output/qc_result.json
            - output/qc_ax.png
            - output/qc_cor.png
            - output/qc_sag.png
            """
        ),
        input_files=("ref_T1w_MNI152_2mm.nii.gz", "boldref.nii.gz", "bold_brain_mask.nii.gz"),
        required_outputs=("qc_result.json", "qc_ax.png", "qc_cor.png", "qc_sag.png"),
        reference_files=("qc_result.json", "qc_ax.png", "qc_cor.png", "qc_sag.png"),
        evaluation_hint="The screenshots must be readable and the QC JSON must report the correct verdict with a short rationale.",
    ),
}


def scene_spec(task_name: str) -> SceneSpec:
    return SCENE_SPECS[task_name]


def task_root(task_name: str, variant_name: str = DEFAULT_VARIANT) -> str:
    return f"{DATA_ROOT}/{DOMAIN_NAME}/{task_name}/{variant_name}"


def desktop_root(task_name: str, variant_name: str = DEFAULT_VARIANT) -> str:
    return f"{DESKTOP_ROOT}/{DOMAIN_NAME}/{task_name}/{variant_name}"


def gcs_root(task_name: str, variant_name: str = DEFAULT_VARIANT) -> str:
    return f"{BUCKET_ROOT}/{DOMAIN_NAME}/{task_name}/{variant_name}"


async def _run(session, command: str, *, check: bool = True, timeout: float | None = None):
    if timeout is not None:
        try:
            result = await session.run_command(command, timeout=timeout)
        except TypeError:
            result = await session.run_command(command)
    else:
        result = await session.run_command(command)
    if isinstance(result, dict):
        return_code = result.get("return_code", result.get("returncode", 1))
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
    else:
        return_code = getattr(result, "return_code", getattr(result, "returncode", 1))
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
    payload = {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
    }
    if check and return_code != 0:
        raise RuntimeError(
            f"remote command failed ({return_code}): {command}\n"
            f"stdout={stdout[:400]}\n"
            f"stderr={stderr[:400]}"
        )
    return payload


async def _write_text(session, path: str, content: str) -> None:
    parent = path.rsplit("/", 1)[0]
    await _run(session, f'mkdir -p "{parent}"')
    await session.write_text(path, content)


def _launcher_script(spec: SceneSpec) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        exec "{RAW_BUNDLE_ROOT}/run_scene.sh" "{spec.launch_scene_arg}" "$@"
        """
    )


def _software_notes(spec: SceneSpec) -> str:
    return dedent(
        f"""\
        Runtime notes
        - Canonical GUI launcher: ./launch_gui.sh
        - Backing raw bundle: {RAW_BUNDLE_ROOT}
        - Tool command inside container: {spec.tool_command}
        - Software family: {spec.software_name} {spec.software_version}
        - The large container images stay in the raw bundle directory on {VM_NAME}; software/ exposes stable launchers and manifests.
        """
    )


async def _create_negative_fixture(session, spec: SceneSpec, root: str) -> None:
    if spec.task_name == "scene1_roi_stats":
        await _write_text(
            session,
            f"{root}/topk.csv",
            "label_id,roi_name,mean,max,voxel_count\n1,wrong_roi,-99,-99,1\n",
        )
        return

    if spec.task_name == "scene2_resample":
        await _run(
            session,
            dedent(
                f"""\
                python3 - <<'PY'
                import csv
                import gzip
                import shutil
                import struct
                from pathlib import Path

                import numpy as np

                NIFTI_DTYPES = {{
                    2: np.uint8,
                    4: np.int16,
                    8: np.int32,
                    16: np.float32,
                    64: np.float64,
                    256: np.int8,
                    512: np.uint16,
                    768: np.uint32,
                }}


                def _unpack(endian: str, fmt: str, header: bytes, start: int):
                    size = struct.calcsize(fmt)
                    return struct.unpack(f"{{endian}}{{fmt}}", header[start:start + size])


                def load_nifti(path: Path):
                    with gzip.open(path, "rb") as handle:
                        payload = handle.read()

                    if struct.unpack("<I", payload[:4])[0] == 348:
                        endian = "<"
                    elif struct.unpack(">I", payload[:4])[0] == 348:
                        endian = ">"
                    else:
                        raise ValueError("invalid_nifti_header")

                    header = payload[:348]
                    dim = _unpack(endian, "8h", header, 40)
                    ndim = int(dim[0])
                    shape = tuple(int(v) for v in dim[1:ndim + 1])
                    datatype = int(_unpack(endian, "h", header, 70)[0])
                    dtype = NIFTI_DTYPES[datatype]
                    vox_offset = int(round(_unpack(endian, "f", header, 108)[0]))
                    data = np.frombuffer(
                        payload,
                        dtype=np.dtype(f"{{endian}}{{dtype().dtype.str[1:]}}"),
                        offset=vox_offset,
                    )
                    expected_size = int(np.prod(shape, dtype=np.int64))
                    data = data[:expected_size].reshape(shape, order="F").copy()
                    return payload, header, data


                positive_mask = Path("{task_root(spec.task_name)}/output_test_pos/roi_mask_2mm_nn.nii.gz")
                output_mask = Path("{root}/roi_mask_2mm_nn.nii.gz")
                statmap_path = Path("{task_root(spec.task_name)}/input/statmap_z_2mm.nii.gz")
                screenshot_src = Path("{task_root(spec.task_name)}/output_test_pos/resample_settings.png")
                screenshot_dst = Path("{root}/resample_settings.png")
                stats_dst = Path("{root}/scene2_stats.csv")

                payload, header, data = load_nifti(positive_mask)
                shifted = np.roll(data, shift=3, axis=0)
                shifted[:3, :, :] = 0

                with gzip.open(output_mask, "wb") as handle:
                    handle.write(payload[:352])
                    handle.write(np.asarray(shifted, dtype=data.dtype, order="F").tobytes(order="F"))

                shutil.copyfile(screenshot_src, screenshot_dst)

                _, _, stat_data = load_nifti(statmap_path)
                mask = shifted > 0.5
                values = stat_data[mask]
                with stats_dst.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("mean", "max", "voxel_count"))
                    writer.writeheader()
                    writer.writerow(
                        {{
                            "mean": f"{{float(values.mean()):.10f}}",
                            "max": f"{{float(values.max()):.10f}}",
                            "voxel_count": int(mask.sum()),
                        }}
                    )
                PY
                """
            ),
            timeout=300.0,
        )
        return

    if spec.task_name == "scene3_skullstrip_qc":
        await _run(
            session,
            dedent(
                f"""\
                cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}"/cand*.png "{root}/"
                """
            ),
        )
        await _write_text(
            session,
            f"{root}/qc_result.json",
            json.dumps(
                {
                    "candidate_voxel_counts": {"cand1": 1846906, "cand2": 1728426, "cand3": 1563172},
                    "chosen_mask": "cand1_loose_mask.nii.gz",
                    "rationale": "Deliberately wrong negative fixture.",
                    "scene": spec.scene_slug,
                    "verdict": "NOT_OK",
                },
                indent=2,
            )
            + "\n",
        )
        return

    if spec.task_name == "scene4_workbench":
        await _run(
            session,
            dedent(
                f"""\
                cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}/thickness.png" "{root}/thickness.png"
                cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}/thickness.png" "{root}/myelin.png"
                """
            ),
        )
        return

    if spec.task_name == "scene5_registration_qc":
        await _run(
            session,
            dedent(
                f"""\
                cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}"/qc_*.png "{root}/"
                """
            ),
        )
        await _write_text(
            session,
            f"{root}/qc_result.json",
            json.dumps(
                {
                    "files": {
                        "bold_brain_mask": "bold_brain_mask.nii.gz",
                        "boldref": "boldref.nii.gz",
                        "ref_T1w_MNI152_2mm": "ref_T1w_MNI152_2mm.nii.gz",
                    },
                    "rationale": "Deliberately wrong negative fixture.",
                    "scene": spec.scene_slug,
                    "verdict": "NOT_OK",
                },
                indent=2,
            )
            + "\n",
        )
        return

    raise KeyError(spec.task_name)


async def _create_positive_fixture(session, spec: SceneSpec, root: str) -> None:
    if spec.task_name != "scene2_resample":
        return

    await _run(
        session,
        dedent(
            f"""\
            python3 - <<'PY'
            import csv
            import gzip
            import struct
            import zlib
            from pathlib import Path

            import numpy as np

            NIFTI_DTYPES = {{
                2: np.uint8,
                4: np.int16,
                8: np.int32,
                16: np.float32,
                64: np.float64,
                256: np.int8,
                512: np.uint16,
                768: np.uint32,
            }}


            def _unpack(endian: str, fmt: str, header: bytes, start: int):
                size = struct.calcsize(fmt)
                return struct.unpack(f"{{endian}}{{fmt}}", header[start:start + size])


            def load_nifti(path: Path):
                with gzip.open(path, "rb") as handle:
                    payload = handle.read()

                if struct.unpack("<I", payload[:4])[0] == 348:
                    endian = "<"
                elif struct.unpack(">I", payload[:4])[0] == 348:
                    endian = ">"
                else:
                    raise ValueError("invalid_nifti_header")

                header = payload[:348]
                dim = _unpack(endian, "8h", header, 40)
                ndim = int(dim[0])
                shape = tuple(int(v) for v in dim[1:ndim + 1])
                datatype = int(_unpack(endian, "h", header, 70)[0])
                dtype = NIFTI_DTYPES[datatype]
                vox_offset = int(round(_unpack(endian, "f", header, 108)[0]))
                slope = float(_unpack(endian, "f", header, 112)[0])
                inter = float(_unpack(endian, "f", header, 116)[0])
                sform_code = int(_unpack(endian, "h", header, 254)[0])
                if sform_code > 0:
                    affine = np.array(
                        [
                            _unpack(endian, "4f", header, 280),
                            _unpack(endian, "4f", header, 296),
                            _unpack(endian, "4f", header, 312),
                            (0.0, 0.0, 0.0, 1.0),
                        ],
                        dtype=np.float64,
                    )
                else:
                    pixdim = _unpack(endian, "8f", header, 76)
                    affine = np.array(
                        [
                            [pixdim[1], 0.0, 0.0, 0.0],
                            [0.0, pixdim[2], 0.0, 0.0],
                            [0.0, 0.0, pixdim[3], 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    )
                data = np.frombuffer(
                    payload,
                    dtype=np.dtype(f"{{endian}}{{dtype().dtype.str[1:]}}"),
                    offset=vox_offset,
                )
                expected_size = int(np.prod(shape, dtype=np.int64))
                data = data[:expected_size].reshape(shape, order="F").copy()
                if slope not in (0.0, 1.0) or inter != 0.0:
                    data = data.astype(np.float64) * (1.0 if slope == 0.0 else slope) + inter
                return payload, affine, data


            def write_simple_png(path: Path) -> None:
                width = 900
                height = 540
                bg = bytes((246, 247, 249))
                ink = bytes((45, 54, 72))
                accent = bytes((40, 116, 166))

                rows = []
                for y in range(height):
                    row = bytearray(bg * width)
                    if 44 <= y < 120:
                        row[:] = bg * width
                    if 96 <= y < 102 or 150 <= y < 156 or 204 <= y < 210:
                        row[88 * 3:808 * 3] = accent * 720
                    if 120 <= y < 144:
                        row[88 * 3:560 * 3] = ink * 472
                    if 258 <= y < 448 and (y - 258) % 2 == 0:
                        row[160 * 3:740 * 3] = bytes((220, 224, 229)) * 580
                    rows.append(b"\\x00" + bytes(row))

                compressed = zlib.compress(b"".join(rows), level=9)

                def chunk(tag: bytes, data: bytes) -> bytes:
                    return (
                        struct.pack(">I", len(data))
                        + tag
                        + data
                        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
                    )

                png = bytearray(b"\\x89PNG\\r\\n\\x1a\\n")
                png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                png += chunk(b"IDAT", compressed)
                png += chunk(b"IEND", b"")
                path.write_bytes(png)


            src_mask_path = Path("{task_root(spec.task_name)}/input/roi_mask_1mm.nii.gz")
            statmap_path = Path("{task_root(spec.task_name)}/input/statmap_z_2mm.nii.gz")
            output_mask_path = Path("{root}/roi_mask_2mm_nn.nii.gz")
            stats_path = Path("{root}/scene2_stats.csv")
            screenshot_path = Path("{root}/resample_settings.png")

            src_payload, src_affine, src_data = load_nifti(src_mask_path)
            stat_payload, stat_affine, stat_data = load_nifti(statmap_path)

            target_shape = stat_data.shape
            grid = np.indices(target_shape, dtype=np.float64)
            ijk = np.stack(
                [grid[0].ravel(), grid[1].ravel(), grid[2].ravel(), np.ones(grid[0].size)],
                axis=0,
            )
            world = stat_affine @ ijk
            src_coords = np.linalg.inv(src_affine) @ world
            src_idx = np.rint(src_coords[:3]).astype(np.int64)
            for axis, dim in enumerate(src_data.shape):
                src_idx[axis] = np.clip(src_idx[axis], 0, dim - 1)
            sampled = src_data[src_idx[0], src_idx[1], src_idx[2]].reshape(target_shape, order="C")
            mask = (sampled > 0.5).astype(np.float32)

            with gzip.open(output_mask_path, "wb") as handle:
                handle.write(stat_payload[:352])
                handle.write(np.asarray(mask, dtype=np.float32, order="F").tobytes(order="F"))

            values = stat_data[mask > 0.5]
            with stats_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("mean", "max", "voxel_count"))
                writer.writeheader()
                writer.writerow(
                    {{
                        "mean": f"{{float(values.mean()):.10f}}",
                        "max": f"{{float(values.max()):.10f}}",
                        "voxel_count": int((mask > 0.5).sum()),
                    }}
                )

            write_simple_png(screenshot_path)
            PY
            """
        ),
        timeout=300.0,
    )


async def _create_generated_inputs(session, spec: SceneSpec, input_root: str) -> None:
    if spec.task_name != "scene4_workbench":
        return

    await _run(
        session,
        dedent(
            f"""\
            python3 - <<'PY'
            from pathlib import Path

            import numpy as np
            from PIL import Image

            ref_path = Path("{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}/thickness.png")
            out_path = Path("{input_root}/screenshot_layout_template.png")

            with Image.open(ref_path) as image:
                rgba = image.convert("RGBA")

            arr = np.asarray(rgba)
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3]

            # Preserve only stable framing cues: hemisphere silhouettes, L/R labels, and color bar.
            non_white = (rgb < 245).any(axis=2) & (alpha > 0)
            mask = np.where(non_white, 0, 255).astype("uint8")
            Image.fromarray(mask, mode="L").save(out_path)
            PY
            """
        ),
        timeout=300.0,
    )


async def sync_scene_assets(task_name: str, vm_ip: str, *, api_port: int = 5000, sync_gcs: bool = True) -> None:
    spec = scene_spec(task_name)
    computer = Computer(
        os_type="linux",
        use_host_computer_server=True,
        api_host=vm_ip,
        api_port=api_port,
    )
    await computer.run()
    session = computer.interface

    data_task_root = task_root(task_name)
    desktop_task_root = desktop_root(task_name)
    bucket_task_root = gcs_root(task_name)
    desktop_parent = desktop_task_root.rsplit("/", 1)[0]

    await _run(
        session,
        dedent(
            f"""\
            mkdir -p "{data_task_root}" "{desktop_parent}"
            mkdir -p "{data_task_root}/input" "{data_task_root}/output" "{data_task_root}/reference" "{data_task_root}/output_test_pos" "{data_task_root}/output_test_neg" "{data_task_root}/software"
            ln -sfn "{data_task_root}" "{desktop_task_root}"
            rm -rf "{data_task_root}/input"/*
            rm -rf "{data_task_root}/reference"/*
            rm -rf "{data_task_root}/output_test_pos"/*
            rm -rf "{data_task_root}/output_test_neg"/*
            rm -rf "{data_task_root}/software"/*
            for name in {" ".join(spec.input_files)}; do
              if [ -f "{RAW_BUNDLE_ROOT}/task/input/{spec.scene_slug}/$name" ]; then
                cp "{RAW_BUNDLE_ROOT}/task/input/{spec.scene_slug}/$name" "{data_task_root}/input/$name"
              fi
            done
            cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}"/* "{data_task_root}/reference/"
            cp "{RAW_BUNDLE_ROOT}/task/output/{spec.scene_slug}"/* "{data_task_root}/output_test_pos/"
            cp "{RAW_BUNDLE_ROOT}/containers_manifest.yaml" "{data_task_root}/software/containers_manifest.yaml"
            chmod 755 "{data_task_root}"
            """
        ),
        timeout=300.0,
    )

    await _write_text(session, f"{data_task_root}/software/launch_gui.sh", _launcher_script(spec))
    await _write_text(session, f"{data_task_root}/software/README.txt", _software_notes(spec))
    await _run(session, f'chmod +x "{data_task_root}/software/launch_gui.sh"')
    await _create_generated_inputs(session, spec, f"{data_task_root}/input")
    await _create_positive_fixture(session, spec, f"{data_task_root}/output_test_pos")

    await _create_negative_fixture(session, spec, f"{data_task_root}/output_test_neg")

    verify = await _run(
        session,
        dedent(
            f"""\
            python3 - <<'PY'
            import json
            from pathlib import Path

            root = Path("{data_task_root}")
            payload = {{
                "task_root": str(root),
                "input_files": sorted(p.name for p in (root / "input").iterdir() if p.is_file()),
                "reference_files": sorted(p.name for p in (root / "reference").iterdir() if p.is_file()),
                "pos_files": sorted(p.name for p in (root / "output_test_pos").iterdir() if p.is_file()),
                "neg_files": sorted(p.name for p in (root / "output_test_neg").iterdir() if p.is_file()),
                "software_files": sorted(p.name for p in (root / "software").iterdir() if p.is_file()),
                "desktop_link_exists": Path("{desktop_task_root}").exists(),
            }}
            print(json.dumps(payload))
            PY
            """
        ),
    )
    payload = json.loads(verify["stdout"])

    if sync_gcs:
        await _run(
            session,
            dedent(
                f"""\
                gsutil -m rsync -r "{data_task_root}/input" "{bucket_task_root}/input"
                gsutil -m rsync -r "{data_task_root}/reference" "{bucket_task_root}/reference"
                gsutil -m rsync -r "{data_task_root}/output_test_pos" "{bucket_task_root}/output_test_pos"
                gsutil -m rsync -r "{data_task_root}/output_test_neg" "{bucket_task_root}/output_test_neg"
                gsutil -m rsync -r "{data_task_root}/software" "{bucket_task_root}/software"
                """
            ),
            timeout=1800.0,
        )

    payload.update(
        {
            "task_name": task_name,
            "variant_name": DEFAULT_VARIANT,
            "vm_name": VM_NAME,
            "vm_ip": vm_ip,
            "gcs_root": bucket_task_root,
        }
    )
    print(json.dumps(payload, indent=2))


def build_parser(default_task_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm-ip", required=True)
    parser.add_argument("--api-port", type=int, default=5000)
    parser.add_argument("--task-name", default=default_task_name)
    parser.add_argument("--skip-gcs-sync", action="store_true")
    return parser


def main(default_task_name: str) -> None:
    args = build_parser(default_task_name).parse_args()
    asyncio.run(
        sync_scene_assets(
            args.task_name,
            args.vm_ip,
            api_port=args.api_port,
            sync_gcs=not args.skip_gcs_sync,
        )
    )
