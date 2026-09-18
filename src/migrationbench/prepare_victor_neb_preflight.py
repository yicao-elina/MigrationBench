#!/usr/bin/env python3
"""Prepare a standardized Victor NEB preflight input under scratch16.

This consumes victor_actionable_neb_queue.csv. For endpoint-only / 2-image
Victor paths, it regenerates explicit intermediate images by linear Cartesian
interpolation and records that lineage in a manifest. The resulting QE input is
for preflight only until a physical activation table is obtained.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path



def sha256_file(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_queue_row(queue_csv, queue_rank=None, path_contains=None):
    rows = list(csv.DictReader(open(queue_csv)))
    for row in rows:
        if queue_rank is not None and int(row['queue_rank']) == queue_rank:
            return row
        if path_contains and path_contains in row['path']:
            return row
    raise ValueError('no matching queue row')


def make_interpolated_images(source_images, out_images, n_images):
    from ase.io import read, write
    frames = read(source_images, index=':')
    if len(frames) >= n_images:
        write(out_images, frames, format='extxyz')
        return {
            'mode': 'copied_existing_path',
            'input_n_images': len(frames),
            'output_n_images': len(frames),
            'preserved_existing_intermediates': True,
        }
    if len(frames) < 2:
        raise ValueError('need at least two frames to interpolate')

    # Preserve the shape of any existing intermediate frames by interpolating
    # piecewise along the supplied path rather than collapsing everything to a
    # straight endpoint-to-endpoint line.
    images = []
    n_segments = len(frames) - 1
    for i in range(n_images):
        scaled = (i / float(n_images - 1)) * n_segments
        seg = min(int(scaled), n_segments - 1)
        alpha = scaled - seg
        left = frames[seg]
        right = frames[seg + 1]
        atoms = left.copy()
        p0 = left.get_positions()
        p1 = right.get_positions()
        atoms.set_positions((1.0 - alpha) * p0 + alpha * p1)
        images.append(atoms)
    write(out_images, images, format='extxyz')
    return {
        'mode': 'piecewise_linear_path_interpolation',
        'input_n_images': len(frames),
        'output_n_images': n_images,
        'preserved_existing_intermediates': len(frames) > 2,
    }


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError('command failed: {}\nSTDOUT:\n{}\nSTDERR:\n{}'.format(' '.join(cmd), proc.stdout, proc.stderr))
    return proc.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue-csv', required=True)
    ap.add_argument('--queue-rank', type=int)
    ap.add_argument('--path-contains')
    ap.add_argument('--output-root', default='/scratch16/pclancy3/yi/revision1_migrationbench_inputs')
    ap.add_argument('--path-family', default='victor_v_st_interstitial_candidate')
    ap.add_argument('--n-images', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--script-dir', default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    row = load_queue_row(args.queue_csv, args.queue_rank, args.path_contains)
    if row.get('queue_decision') not in {'standardize_and_preflight', 'parse_and_standardize_completed_candidate'}:
        raise ValueError('row is not ready for standardization: {}'.format(row.get('queue_decision')))
    if not row.get('companion_images') or not row.get('engine_template_source'):
        raise ValueError('row lacks companion images or template source')

    branch = 'victor_{}_{}_s{}'.format(row.get('path_key','path'), Path(row['path']).stem, args.seed)
    branch = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in branch)
    out_dir = Path(args.output_root) / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_images = out_dir / 'proposal_images.extxyz'
    engine_template = out_dir / 'engine_template.in'
    neb_in = out_dir / 'neb.in'
    qe_manifest = out_dir / 'qe_input_manifest.json'
    prep_manifest = out_dir / 'victor_preflight_manifest.json'
    geom_json = out_dir / 'geometry_check.json'
    geom_csv = out_dir / 'geometry_check.csv'

    image_lineage = make_interpolated_images(row['companion_images'], proposal_images, args.n_images)
    script_dir = Path(args.script_dir)
    run(['python3', str(script_dir / 'qe_engine_template_from_pw.py'), '--input', row['engine_template_source'], '--output', str(engine_template)])
    run(['python3', str(script_dir / 'qe_neb_from_images.py'), '--images', str(proposal_images), '--engine-template', str(engine_template), '--output', str(neb_in), '--manifest', str(qe_manifest), '--path-id', branch, '--path-family', args.path_family, '--measurement-purpose', 'victor_standardized_qe_preflight', '--source-neb-output', row['path'], '--restart-mode', 'from_scratch', '--nstep-path', '1000', '--opt-scheme', 'broyden', '--ci-scheme', 'auto', '--path-thr', '0.03'])
    geom_status = None
    try:
        run(['python3', str(script_dir / 'check_neb_image_geometry.py'), '--images', str(proposal_images), '--output-json', str(geom_json), '--output-csv', str(geom_csv), '--migrant-element', 'V'])
        geom_status = json.loads(geom_json.read_text())
    except Exception as exc:
        geom_status = {'status': 'check_failed', 'error': str(exc)}

    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'branch_id': branch,
        'queue_row': row,
        'seed': args.seed,
        'output_dir': str(out_dir),
        'outputs': {
            'proposal_images': str(proposal_images),
            'engine_template': str(engine_template),
            'neb_in': str(neb_in),
            'qe_input_manifest': str(qe_manifest),
            'geometry_check_json': str(geom_json),
            'geometry_check_csv': str(geom_csv),
        },
        'image_lineage': image_lineage,
        'geometry_status': geom_status,
        'sha256': {
            'source_images': sha256_file(row['companion_images']),
            'source_neb_output': sha256_file(row['path']),
            'proposal_images': sha256_file(proposal_images),
            'neb_in': sha256_file(neb_in),
        },
        'decision': 'ready_for_qe_preflight' if geom_status and geom_status.get('status') == 'pass' else 'inspect_before_submit',
        'notes': ['Victor-derived paths are provenance/preflight until regenerated QE produces a physical table and convergence is audited.']
    }
    prep_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
