#!/usr/bin/env python3
"""Inventory Victor QE/NEB calculations for MigrationBench provenance.

The script is deliberately parser-light and standard-library only so it can run
on Rockfish login nodes. It does not treat collaborator docs as instructions;
it records observed files and convergence/provenance signals.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

CALC_RE = re.compile(r"calculation\s*=\s*['\"]?([A-Za-z0-9_-]+)", re.I)
NAT_RE = re.compile(r"\bnat\s*=\s*([0-9]+)", re.I)
NTYP_RE = re.compile(r"\bntyp\s*=\s*([0-9]+)", re.I)
PREFIX_RE = re.compile(r"\bprefix\s*=\s*['\"]?([^,'\"\s]+)", re.I)
ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.]+)\s+Ry")
NEB_ACT_RE = re.compile(r"activation energy \(->\)\s*=\s*([-+0-9.Ee*]+)\s*eV")
NEB_PATH_RE = re.compile(r"initial path length\s*=\s*([-+0-9.Ee*]+)\s*bohr")
NEB_IMAGES_RE = re.compile(r"num_of_images\s*=\s*([0-9]+)", re.I)
ERROR_HINTS = ["%%%%%%%%", "error in routine", "convergence NOT achieved", "too many bands", "stopping", "segmentation fault", "out of memory"]


def read_text(path, max_bytes=262144, skip_larger_than=None):
    try:
        if skip_larger_than is not None and path.stat().st_size > skip_larger_than:
            return '__SKIPPED_LARGE_FILE__ size_bytes={}'.format(path.stat().st_size)
        with open(path, 'rb') as f:
            data = f.read(max_bytes)
        return data.decode('utf-8', errors='replace')
    except Exception as exc:
        return "__READ_ERROR__ {}".format(exc)


def first_match(regex, text):
    m = regex.search(text)
    return m.group(1) if m else None


def infer_kind(path, text):
    name = path.name.lower()
    if 'neb' in name or 'program neb' in text.lower() or 'begin_path_input' in text.lower() or 'num_of_images' in text.lower():
        return 'neb'
    calc = first_match(CALC_RE, text)
    if calc:
        return calc.lower()
    if path.suffix == '.in':
        return 'qe_input_unknown'
    if path.suffix == '.out':
        return 'qe_output_unknown'
    return 'structure_or_other'


def summarize_file(path, max_read_bytes=262144, skip_larger_than=None):
    text = read_text(path, max_read_bytes, skip_larger_than)
    lower = text.lower()
    kind = infer_kind(path, text)
    energies = ENERGY_RE.findall(text)
    act = NEB_ACT_RE.findall(text)
    err_hints = [h for h in ERROR_HINTS if h.lower() in lower]
    row = {
        'path': str(path),
        'directory': str(path.parent),
        'file': path.name,
        'suffix': path.suffix,
        'size_bytes': path.stat().st_size if path.exists() else None,
        'kind': kind,
        'calculation': first_match(CALC_RE, text),
        'nat': first_match(NAT_RE, text),
        'ntyp': first_match(NTYP_RE, text),
        'prefix': first_match(PREFIX_RE, text),
        'num_of_images': first_match(NEB_IMAGES_RE, text),
        'job_done': 'JOB DONE' in text,
        'has_error_hint': bool(err_hints),
        'error_hints': ';'.join(err_hints),
        'final_total_energy_Ry': energies[-1] if energies else None,
        'n_total_energy_lines': len(energies),
        'neb_first_activation_forward_eV': next((x for x in act if '*' not in x), None),
        'neb_activation_overflow': any('*' in x for x in act),
        'neb_initial_path_length_bohr': first_match(NEB_PATH_RE, text),
        'classification': None,
        'content_scan_limited': len(text) >= max_read_bytes,
        'content_skipped_large': text.startswith('__SKIPPED_LARGE_FILE__'),
    }
    if kind == 'neb':
        if row['neb_activation_overflow'] or (row['neb_initial_path_length_bohr'] and '*' in row['neb_initial_path_length_bohr']):
            row['classification'] = 'quarantine_neb_overflow'
        elif row['job_done'] and row['neb_first_activation_forward_eV']:
            row['classification'] = 'completed_neb_candidate'
        elif row['neb_first_activation_forward_eV']:
            row['classification'] = 'partial_neb_candidate'
        else:
            row['classification'] = 'neb_needs_input_or_restart_review'
    elif path.suffix == '.out':
        if row['job_done'] and not row['has_error_hint']:
            row['classification'] = 'completed_qe_output_candidate'
        elif row['has_error_hint']:
            row['classification'] = 'quarantine_or_restart_review'
        else:
            row['classification'] = 'partial_or_unknown_qe_output'
    elif path.suffix == '.in':
        row['classification'] = 'qe_input_template_candidate'
    else:
        row['classification'] = 'structure_candidate'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--max-files', type=int, default=5000)
    ap.add_argument('--max-read-bytes', type=int, default=262144)
    ap.add_argument('--skip-larger-than', type=int, default=50000000, help='Do not scan contents of files larger than this many bytes; still record path/size')
    ap.add_argument('--include-suffix', action='append', default=['.in', '.out', '.xyz', '.cif'])
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    suffixes = set(args.include_suffix)
    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Avoid known unreadable or huge plotting/cache internals without failing the inventory.
        dirnames[:] = [d for d in dirnames if d not in {'.git', '__pycache__'}]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix not in suffixes:
                continue
            rows.append(summarize_file(path, args.max_read_bytes, args.skip_larger_than))
            if len(rows) >= args.max_files:
                break
        if len(rows) >= args.max_files:
            break

    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'n_files': len(rows),
        'counts_by_classification': {},
        'counts_by_kind': {},
        'rows': rows,
    }
    for row in rows:
        summary['counts_by_classification'][row['classification']] = summary['counts_by_classification'].get(row['classification'], 0) + 1
        summary['counts_by_kind'][row['kind']] = summary['counts_by_kind'].get(row['kind'], 0) + 1

    outj = Path(args.output_json)
    outc = Path(args.output_csv)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outc.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fieldnames = list(rows[0].keys()) if rows else []
    with outc.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: summary[k] for k in ['root','n_files','counts_by_kind','counts_by_classification']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
