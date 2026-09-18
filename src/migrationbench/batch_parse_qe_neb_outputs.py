#!/usr/bin/env python3
"""Batch-parse QE neb.out files into CSV and JSON summaries."""

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from parse_qe_neb_output import parse_neb_out


def label_for(path, root):
    p = path.resolve()
    if root:
        try:
            return str(p.relative_to(root.resolve())).replace('/neb.out', '')
        except ValueError:
            pass
    parent = p.parent.name
    return parent if parent else p.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neb_out', nargs='+', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-csv', required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.neb_out:
        summary = parse_neb_out(path)
        summary['label'] = label_for(path, args.root)
        rows.append(summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2) + '\n')
    fields = [
        'label', 'neb_out', 'job_done', 'converged_by_default_gate',
        'last_iteration', 'last_complete_iteration', 'activation_forward_eV',
        'activation_reverse_eV', 'max_image_error_eV_A',
        'barrier_drift_last_three_eV', 'n_iterations_parsed',
        'n_complete_iterations_parsed', 'n_images_last_iteration'
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'parsed': len(rows), 'csv': str(args.output_csv), 'json': str(args.output_json)}, indent=2))


if __name__ == '__main__':
    main()
