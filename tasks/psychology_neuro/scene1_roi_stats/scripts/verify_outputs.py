#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

TOL = 1e-6


def load_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name')
    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    refdir = Path(args.reference_dir)

    if not (outdir / 'topk.csv').exists():
        print(json.dumps({'score': 0.0, 'reasons': ['missing:topk.csv']}))
        return

    ref_rows = load_rows(refdir / 'topk.csv')
    agent_rows = load_rows(outdir / 'topk.csv')

    if not agent_rows:
        print(json.dumps({'score': 0.0, 'reasons': ['empty:topk.csv']}))
        return

    if list(agent_rows[0].keys()) != list(ref_rows[0].keys()):
        print(json.dumps({'score': 0.0, 'reasons': ['header_mismatch:topk.csv']}))
        return

    agent_by_label = {}
    for row in agent_rows:
        agent_by_label[row['label_id']] = row

    matched = 0
    reasons = []
    for ref_row in ref_rows:
        lid = ref_row['label_id']
        agent_row = agent_by_label.get(lid)
        if agent_row is None:
            reasons.append(f'missing_label_{lid}')
            continue
        if agent_row.get('roi_name') != ref_row['roi_name']:
            reasons.append(f'roi_name_mismatch_label_{lid}')
            continue
        value_ok = True
        for key in ('mean', 'max', 'voxel_count'):
            if abs(float(agent_row[key]) - float(ref_row[key])) > TOL:
                reasons.append(f'{key}_mismatch_label_{lid}')
                value_ok = False
                break
        if value_ok:
            matched += 1

    score = matched / len(ref_rows) if ref_rows else 0.0
    reasons.insert(0, f'matched:{matched}/{len(ref_rows)}')
    print(json.dumps({'score': score, 'passed': score == 1.0, 'reasons': reasons}))


if __name__ == '__main__':
    main()
