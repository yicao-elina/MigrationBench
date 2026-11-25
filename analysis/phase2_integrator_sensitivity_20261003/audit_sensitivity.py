#!/usr/bin/env python3
"""Audit the isolated Phase-2 integrator/thermostat sensitivity matrix."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def arr(rows, key): return np.asarray([float(r[key]) for r in rows], dtype=float)


def first_crossing(rows):
    emin = emax = float(rows[0]['total_energy_eV'])
    for r in rows:
        e = float(r['total_energy_eV']); emin, emax = min(emin, e), max(emax, e)
        checks = {
            'force_gt_100_eV_A': float(r['max_force_eV_A']) >= 100,
            'velocity_gt_1000_A_ps': float(r['max_velocity_A_ps']) >= 1000,
            'min_distance_lt_1p5_A': float(r['min_distance_A']) < 1.5,
            'temperature_gt_2000_K': float(r['temperature_K']) >= 2000,
            'unwrapped_displacement_gt_50_A': float(r['max_unwrapped_displacement_A']) >= 50,
            'n_atoms_gt5_A_ge_100': float(r['n_atoms_unwrapped_displacement_gt5A']) >= 100,
            'total_energy_span_ge_250_eV': emax - emin >= 250,
        }
        bad = [k for k, v in checks.items() if v]
        if bad:
            return {'step': int(r['absolute_step']), 'time_ps': float(r['time_ps']), 'gate': bad[0]}
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for p in sorted(root.glob('snapshot_*/model_*/setting_*/metrics.csv')):
        rows = list(csv.DictReader(p.open()))
        meta = json.loads((p.parent / 'handoff_input.json').read_text())
        finite_fields = ['potential_energy_eV','kinetic_energy_eV','total_energy_eV','temperature_K',
                         'max_force_eV_A','max_velocity_A_ps','max_momentum_amu_A_ps','min_distance_A',
                         'max_unwrapped_displacement_A','rdf_l1_vs_initial']
        finite = all(np.isfinite(arr(rows, k)).all() for k in finite_fields)
        gate_ok = finite and all([
            np.nanmax(arr(rows,'max_force_eV_A')) < 100,
            np.nanmax(arr(rows,'max_velocity_A_ps')) < 1000,
            np.nanmin(arr(rows,'min_distance_A')) >= 1.5,
            np.nanmax(arr(rows,'temperature_K')) < 2000,
            np.nanmax(arr(rows,'max_unwrapped_displacement_A')) < 50,
            np.nanmax(arr(rows,'n_atoms_unwrapped_displacement_gt5A')) < 100,
            np.nanmax(arr(rows,'total_energy_eV')) - np.nanmin(arr(rows,'total_energy_eV')) < 250,
        ])
        if not gate_ok: label = 'runaway_or_nonfinite'
        elif np.nanmax(arr(rows,'max_unwrapped_displacement_A')) > 5 or np.nanmax(arr(rows,'n_atoms_unwrapped_displacement_gt5A')) > 0: label = 'bounded_with_structural_shift'
        else: label = 'stable_short_handoff'
        first = first_crossing(rows)
        records.append({
            'snapshot': p.parent.parent.parent.name.removeprefix('snapshot_'),
            'model': p.parent.parent.name.removeprefix('model_'),
            'setting': p.parent.name.removeprefix('setting_'),
            'integrator': meta['integrator'], 'timestep_fs': meta['timestep_fs'],
            'friction_1_per_fs': meta['friction_1_per_fs'], 'n_frames': len(rows),
            'classification': label, 'first_divergence': first,
            'max_force_eV_A': float(np.nanmax(arr(rows,'max_force_eV_A'))),
            'max_velocity_A_ps': float(np.nanmax(arr(rows,'max_velocity_A_ps'))),
            'min_distance_A': float(np.nanmin(arr(rows,'min_distance_A'))),
            'max_temperature_K': float(np.nanmax(arr(rows,'temperature_K'))),
            'max_unwrapped_displacement_A': float(np.nanmax(arr(rows,'max_unwrapped_displacement_A'))),
            'total_energy_span_eV': float(np.nanmax(arr(rows,'total_energy_eV')) - np.nanmin(arr(rows,'total_energy_eV'))),
            'input_sha256': meta['input_sha256'], 'positions_sha256': meta['positions_sha256'],
            'momenta_sha256': meta['momenta_sha256'], 'cell_sha256': meta['cell_sha256'],
            'model_sha256': meta['model_sha256'],
        })
    (out/'sensitivity_summary.json').write_text(json.dumps(records, indent=2, sort_keys=True)+'\n')
    cols=['snapshot','model','setting','integrator','timestep_fs','friction_1_per_fs','n_frames','classification',
          'first_divergence_step','first_divergence_time_ps','first_divergence_gate','max_force_eV_A',
          'max_velocity_A_ps','min_distance_A','max_temperature_K','max_unwrapped_displacement_A',
          'total_energy_span_eV','input_sha256','positions_sha256','momenta_sha256','cell_sha256','model_sha256']
    with (out/'sensitivity_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in records:
            first=r['first_divergence'] or {}
            w.writerow({**{k:r.get(k,'') for k in cols},'first_divergence_step':first.get('step',''),
                        'first_divergence_time_ps':first.get('time_ps',''),'first_divergence_gate':first.get('gate','')})
    counts={}
    for r in records: counts.setdefault((r['model'],r['snapshot']),{}).setdefault(r['classification'],0); counts[(r['model'],r['snapshot'])][r['classification']]+=1
    report={'runs':len(records),'classifications':{k:sum(r['classification']==k for r in records) for k in sorted({r['classification'] for r in records})},
            'by_model_snapshot':{f'{k[0]}/{k[1]}':v for k,v in sorted(counts.items())},
            'hash_consistency':len({(r['snapshot'],r['input_sha256'],r['positions_sha256'],r['momenta_sha256'],r['cell_sha256']) for r in records}) == 2}
    (out/'audit_summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == '__main__': main()
