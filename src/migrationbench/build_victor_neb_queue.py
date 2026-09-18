#!/usr/bin/env python3
"""Build an actionable Victor NEB queue from the inventory CSV.

For each NEB output candidate, infer companion neb input, image path, and engine
template candidates in the same directory. The output is a review queue; it does
not mark values manuscript-grade.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def path_key(name):
    stem = Path(name).stem
    for token in ['_neb_run1','_neb_run2','_neb_run3','_neb_parallel','_neb_v2','_neb_cr','_neb']:
        stem = stem.replace(token, '')
    return stem


def first_existing(candidates, available_paths):
    for c in candidates:
        if c and str(c) in available_paths:
            return c
    return None


def remote_glob(available_paths, directory, predicate):
    prefix = str(directory).rstrip("/") + "/"
    return sorted(Path(x) for x in available_paths if x.startswith(prefix) and predicate(Path(x).name))


def infer_companions(row, available_paths):
    d = Path(row['directory'])
    key = path_key(row['file'])
    same_stem_in = d / (Path(row['file']).stem + '.in')
    possible_neb_in = [same_stem_in, d / (key + '_neb.in'), d / (key + '_neb_v2.in'), d / (key + '_neb_parallel.in'), d / (key + '_neb_cr.in'), d / 'neb.in']
    # Prefer path-specific images before generic directory-wide images.
    possible_xyz = [d / (key + '_v-sb2te3.xyz'), d / (key + '_cr-sb2te3.xyz')]
    possible_xyz += remote_glob(available_paths, d, lambda n: n.startswith(key) and n.endswith('.xyz'))
    possible_xyz += remote_glob(available_paths, d, lambda n: key in n and n.endswith('.xyz'))
    possible_xyz += [d / 'v-sb2te3.xyz', d / 'sb2te3.xyz']
    possible_pw = [d / 'pw_1.in', d / '0_qe_input.in', d / '1_qe_input.in']
    possible_pw += remote_glob(available_paths, d, lambda n: n.endswith('qe_input.in'))
    return {
        'path_key': key,
        'companion_neb_in': str(first_existing(possible_neb_in, available_paths) or ''),
        'companion_images': str(first_existing(possible_xyz, available_paths) or ''),
        'engine_template_source': str(first_existing(possible_pw, available_paths) or ''),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory-csv', required=True)
    ap.add_argument('--output-csv', required=True)
    ap.add_argument('--output-json', required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inventory_csv)))
    available_paths = set(r.get('path', '') for r in rows)
    rows_by_path = {r.get('path', ''): r for r in rows}
    candidates = []
    for row in rows:
        if row.get('kind') != 'neb':
            continue
        item = dict(row)
        item.update(infer_companions(row, available_paths))
        item['ready_for_standard_qe_input'] = bool(item['companion_images'] and item['engine_template_source'])
        item['review_notes'] = ''
        companion_input_row = rows_by_path.get(item['companion_neb_in'], {})
        if companion_input_row.get('num_of_images') == '2':
            item['review_notes'] = 'existing Victor input uses 2 images; treat as endpoint/path preflight, not barrier benchmark, until regenerated with intermediates'
        if item['classification'] == 'partial_neb_candidate' and item['neb_first_activation_forward_eV']:
            ea = float(item['neb_first_activation_forward_eV'])
            item['queue_decision'] = 'standardize_and_preflight' if ea < 10 else 'diagnostic_or_defer_high_barrier'
        elif item['classification'] == 'completed_neb_candidate':
            item['queue_decision'] = 'parse_and_standardize_completed_candidate'
        elif item['classification'] == 'quarantine_neb_overflow':
            item['queue_decision'] = 'quarantine_overflow'
        else:
            item['queue_decision'] = 'needs_input_or_restart_review'
        if not item['ready_for_standard_qe_input'] and item['queue_decision'] in {'standardize_and_preflight','parse_and_standardize_completed_candidate'}:
            item['queue_decision'] = 'needs_companion_files_before_preflight'
        candidates.append(item)

    decision_order = {
        'standardize_and_preflight': 0,
        'parse_and_standardize_completed_candidate': 1,
        'needs_companion_files_before_preflight': 2,
        'needs_input_or_restart_review': 3,
        'diagnostic_or_defer_high_barrier': 5,
        'quarantine_overflow': 9,
    }
    candidates.sort(key=lambda r: (decision_order.get(r['queue_decision'], 8), r.get('inventory_source',''), r.get('path','')))
    for i, row in enumerate(candidates, 1):
        row['queue_rank'] = i

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['queue_rank','queue_decision','inventory_source','classification','path_key','path','companion_neb_in','companion_images','engine_template_source','ready_for_standard_qe_input','neb_first_activation_forward_eV','neb_initial_path_length_bohr','job_done','has_error_hint','error_hints','review_notes']
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in candidates:
            w.writerow({k:r.get(k) for k in fields})
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'n_candidates': len(candidates), 'counts_by_decision': {}, 'top': [{k:r.get(k) for k in fields} for r in candidates[:20]]}
    for r in candidates:
        summary['counts_by_decision'][r['queue_decision']] = summary['counts_by_decision'].get(r['queue_decision'], 0) + 1
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
