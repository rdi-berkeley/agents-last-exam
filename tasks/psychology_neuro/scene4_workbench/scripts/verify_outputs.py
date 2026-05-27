#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path: Path) -> str:
    with Image.open(path) as image:
        grayscale = image.convert('L')
        resized = grayscale.resize((9, 8))
    image_data = np.asarray(resized)
    bits = image_data[:, 1:] > image_data[:, :-1]
    return ''.join('1' if value else '0' for value in bits.flatten())


def hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def mae(agent_path: Path, ref_path: Path) -> float:
    with Image.open(agent_path) as agent_image:
        agent = np.asarray(agent_image.convert('L'))
    with Image.open(ref_path) as ref_image:
        ref = np.asarray(ref_image.convert('L'))
    if agent.shape != ref.shape:
        with Image.open(agent_path) as agent_image:
            agent = np.asarray(agent_image.convert('L').resize((ref.shape[1], ref.shape[0])))
    return float(abs(agent.astype("float32") - ref.astype("float32")).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ref_dir = Path(args.reference_dir)
    required = ('thickness.png', 'myelin.png')
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        print(json.dumps({'score': 0.0, 'reasons': [f'missing:{name}' for name in missing]}))
        return

    reasons = []
    passed = True
    for name in required:
        try:
            dist = hamming(dhash(out_dir / name), dhash(ref_dir / name))
            mean_abs_err = mae(out_dir / name, ref_dir / name)
        except Exception as exc:
            reasons.append(f'{name}:unreadable:{exc}')
            passed = False
            continue
        reasons.append(f'{name}:hamming={dist}:mae={mean_abs_err:.3f}')
        if dist > 6 or mean_abs_err > 4.0:
            passed = False
    print(json.dumps({'score': 1.0 if passed else 0.0, 'passed': passed, 'reasons': reasons}))


if __name__ == '__main__':
    main()
