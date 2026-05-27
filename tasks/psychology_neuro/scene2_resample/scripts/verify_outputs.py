#!/usr/bin/env python
import argparse
import csv
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

TOL = 1e-6
NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def read_stats(path: Path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        row = next(csv.DictReader(handle))
    return {key: float(value) for key, value in row.items()}


def _unpack(endian: str, fmt: str, header: bytes, start: int):
    return struct.unpack(f'{endian}{fmt}', header[start:start + struct.calcsize(fmt)])


def load_nifti(path: Path):
    with gzip.open(path, 'rb') as handle:
        payload = handle.read()

    if len(payload) < 352:
        raise ValueError('truncated_nifti')

    if struct.unpack('<I', payload[:4])[0] == 348:
        endian = '<'
    elif struct.unpack('>I', payload[:4])[0] == 348:
        endian = '>'
    else:
        raise ValueError('invalid_nifti_header')

    header = payload[:348]
    dim = _unpack(endian, '8h', header, 40)
    ndim = int(dim[0])
    shape = tuple(int(v) for v in dim[1:ndim + 1])
    datatype = int(_unpack(endian, 'h', header, 70)[0])
    dtype = NIFTI_DTYPES.get(datatype)
    if dtype is None:
        raise ValueError(f'unsupported_datatype:{datatype}')

    vox_offset = int(round(_unpack(endian, 'f', header, 108)[0]))
    slope = float(_unpack(endian, 'f', header, 112)[0])
    inter = float(_unpack(endian, 'f', header, 116)[0])
    sform_code = int(_unpack(endian, 'h', header, 254)[0])

    if sform_code > 0:
        affine = np.array(
            [
                _unpack(endian, '4f', header, 280),
                _unpack(endian, '4f', header, 296),
                _unpack(endian, '4f', header, 312),
                (0.0, 0.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
    else:
        pixdim = _unpack(endian, '8f', header, 76)
        affine = np.array(
            [
                [pixdim[1], 0.0, 0.0, 0.0],
                [0.0, pixdim[2], 0.0, 0.0],
                [0.0, 0.0, pixdim[3], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    data = np.frombuffer(payload, dtype=np.dtype(f'{endian}{dtype().dtype.str[1:]}'), offset=vox_offset)
    expected_size = int(np.prod(shape, dtype=np.int64))
    if data.size < expected_size:
        raise ValueError('truncated_nifti_data')
    data = data[:expected_size].reshape(shape, order='F')
    if slope not in (0.0, 1.0) or inter != 0.0:
        data = data.astype(np.float64) * (1.0 if slope == 0.0 else slope) + inter
    return data, affine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.output_dir)
    required = ('roi_mask_2mm_nn.nii.gz', 'resample_settings.png', 'scene2_stats.csv')
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        print(json.dumps({'score': 0.0, 'reasons': [f'missing:{name}' for name in missing]}))
        return

    try:
        with Image.open(out_dir / 'resample_settings.png') as image:
            image.verify()
    except Exception:
        print(json.dumps({'score': 0.0, 'reasons': ['unreadable_resample_settings_png']}))
        return

    try:
        stat_data, stat_affine = load_nifti(input_dir / 'statmap_z_2mm.nii.gz')
        mask_data, mask_affine = load_nifti(out_dir / 'roi_mask_2mm_nn.nii.gz')
    except Exception as exc:
        print(json.dumps({'score': 0.0, 'reasons': [f'unreadable_nifti:{exc}']}))
        return

    if mask_data.shape != stat_data.shape:
        print(json.dumps({'score': 0.0, 'reasons': ['mask_shape_mismatch']}))
        return
    if not np.allclose(mask_affine, stat_affine):
        print(json.dumps({'score': 0.0, 'reasons': ['mask_affine_mismatch']}))
        return

    mask = mask_data > 0.5
    if mask.sum() == 0:
        print(json.dumps({'score': 0.0, 'reasons': ['empty_mask']}))
        return

    values = stat_data[mask]
    derived = {
        'mean': float(values.mean()),
        'max': float(values.max()),
        'voxel_count': float(mask.sum()),
    }
    reference = read_stats(ref_dir / 'scene2_stats.csv')
    agent_csv = read_stats(out_dir / 'scene2_stats.csv')

    reasons = []
    score = 0.0
    derived_ok = True
    csv_ok = True
    for key in ('mean', 'max', 'voxel_count'):
        if abs(derived[key] - reference[key]) > TOL:
            derived_ok = False
            reasons.append(f'derived_{key}_mismatch')
        if abs(agent_csv[key] - reference[key]) > TOL:
            csv_ok = False
            reasons.append(f'csv_{key}_mismatch')
    if derived_ok:
        score += 0.7
    if csv_ok:
        score += 0.3
    print(json.dumps({'score': score, 'passed': score == 1.0, 'reasons': reasons or ['ok']}))


if __name__ == '__main__':
    main()
