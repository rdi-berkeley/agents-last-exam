#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def norm_verdict(value: str) -> str:
    return value.strip().lower().replace('-', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()

    out_path = Path(args.output_dir) / 'qc_result.json'
    ref_path = Path(args.reference_dir) / 'qc_result.json'
    if not out_path.exists():
        print(json.dumps({'score': 0.0, 'reasons': ['missing:qc_result.json']}))
        return

    agent = json.loads(out_path.read_text(encoding='utf-8'))
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    reasons = []
    if agent.get('chosen_mask') != ref.get('chosen_mask'):
        reasons.append('chosen_mask_mismatch')
    if norm_verdict(agent.get('verdict', '')) != norm_verdict(ref.get('verdict', '')):
        reasons.append('verdict_mismatch')
    if not str(agent.get('rationale', '')).strip():
        reasons.append('missing_rationale')
    score = 1.0 if not reasons else 0.0
    print(json.dumps({'score': score, 'passed': score == 1.0, 'reasons': reasons or ['ok']}))


if __name__ == '__main__':
    main()
