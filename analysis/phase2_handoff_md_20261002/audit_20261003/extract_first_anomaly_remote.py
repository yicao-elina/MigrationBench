import csv, glob, json, os
import numpy as np
from ase.io import iread
from ase.neighborlist import neighbor_list

root = '/scratch16/pclancy3/yi/phase2_handoff_md_20261002/runs/retry2'

def first_step(rows):
    emin = emax = float(rows[0]['total_energy_eV'])
    for row in rows:
        vals = [float(row['max_force_eV_A']) >= 100,
                float(row['max_velocity_A_ps']) >= 1000,
                float(row['min_distance_A']) < 1.5,
                float(row['temperature_K']) >= 2000,
                float(row['max_unwrapped_displacement_A']) >= 50,
                float(row['n_atoms_unwrapped_displacement_gt5A']) >= 100]
        energy = float(row['total_energy_eV'])
        emin, emax = min(emin, energy), max(emax, energy)
        if emax - emin >= 250:
            vals.append(True)
        if any(vals):
            return int(row['absolute_step'])
    return None

for metrics in sorted(glob.glob(root + '/snapshot_*/model_*/metrics.csv')):
    rows = list(csv.DictReader(open(metrics)))
    first = first_step(rows)
    if first is None:
        continue
    traj = metrics.replace('/metrics.csv', '/trajectory.extxyz')
    snap = os.path.basename(os.path.dirname(os.path.dirname(metrics)))
    model = os.path.basename(os.path.dirname(metrics))
    record = {'snapshot': snap, 'model': model, 'first_gate_step': first, 'frames': []}
    for atoms in iread(traj, index=':'):
        step = int(atoms.info.get('absolute_step', -1))
        if step in (first - 20, first - 10, first, first + 10, first + 20):
            forces = np.asarray(atoms.arrays.get('forces', atoms.get_forces()))
            force_norm = np.linalg.norm(forces, axis=1)
            velocities = atoms.get_momenta() / atoms.get_masses()[:, None] * 1000.0
            velocity_norm = np.linalg.norm(velocities, axis=1)
            force_idx, velocity_idx = int(np.argmax(force_norm)), int(np.argmax(velocity_norm))
            i, j, d = neighbor_list('ijd', atoms, 6.0, self_interaction=False)
            neighbors = []
            for jj, dd in zip(j[i == force_idx], d[i == force_idx]):
                neighbors.append((float(dd), int(jj), atoms[int(jj)].symbol))
            for jj, dd in zip(i[j == force_idx], d[j == force_idx]):
                neighbors.append((float(dd), int(jj), atoms[int(jj)].symbol))
            record['frames'].append({
                'step': step, 'max_force_eV_A': float(force_norm.max()),
                'max_force_atom_index': force_idx, 'max_force_atom_symbol': atoms[force_idx].symbol,
                'max_force_atom_force_vector_eV_A': forces[force_idx].tolist(),
                'max_velocity_A_ps': float(velocity_norm.max()),
                'max_velocity_atom_index': velocity_idx, 'max_velocity_atom_symbol': atoms[velocity_idx].symbol,
                'min_distance_A': float(np.min(d)) if len(d) else None,
                'neighbors_of_max_force_atom': sorted(neighbors)[:12],
            })
        if step > first + 20:
            break
    print(json.dumps(record, sort_keys=True))
