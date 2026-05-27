#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from PIL import Image


def norm_verdict(value: str) -> str:
    return value.strip().lower().replace('-', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ref_dir = Path(args.reference_dir)
    required = ('qc_result.json', 'qc_ax.png', 'qc_cor.png', 'qc_sag.png')
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        print(json.dumps({'score': 0.0, 'reasons': [f'missing:{name}' for name in missing]}))
        return

    agent = json.loads((out_dir / 'qc_result.json').read_text(encoding='utf-8'))
    ref = json.loads((ref_dir / 'qc_result.json').read_text(encoding='utf-8'))
    reasons = []
    if norm_verdict(agent.get('verdict', '')) != norm_verdict(ref.get('verdict', '')):
        reasons.append('verdict_mismatch')
    if not str(agent.get('rationale', '')).strip():
        reasons.append('missing_rationale')
    for name in ('qc_ax.png', 'qc_cor.png', 'qc_sag.png'):
        try:
            with Image.open(out_dir / name) as image:
                image.verify()
        except Exception:
            reasons.append(f'unreadable:{name}')
    score = 1.0 if not reasons else 0.0
    print(json.dumps({'score': score, 'passed': score == 1.0, 'reasons': reasons or ['ok']}))


if __name__ == '__main__':
    main()
