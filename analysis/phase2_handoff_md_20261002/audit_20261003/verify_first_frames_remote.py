import glob, hashlib, json, os
import numpy as np
from ase.io import read

root = '/scratch16/pclancy3/yi/phase2_handoff_md_20261002'

def h(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

for traj in sorted(glob.glob(root + '/runs/retry2/snapshot_*/model_*/trajectory.extxyz')):
    snap = os.path.basename(os.path.dirname(os.path.dirname(traj))).removeprefix('snapshot_')
    inp = f'{root}/snapshots/from_scratch_source/snapshot_{snap}.extxyz'
    source, first = read(inp, index=-1), read(traj, index=0)
    values = {
        'positions': [h(source.positions), h(first.positions)],
        'momenta': [h(source.get_momenta()), h(first.get_momenta())],
        'cell': [h(source.cell.array), h(first.cell.array)],
        'species': [hashlib.sha256(''.join(source.get_chemical_symbols()).encode()).hexdigest(),
                    hashlib.sha256(''.join(first.get_chemical_symbols()).encode()).hexdigest()],
        'pbc': [source.pbc.tolist(), first.pbc.tolist()],
        'natoms': [len(source), len(first)],
    }
    print(json.dumps({
        'trajectory': traj, 'snapshot': snap,
        'model': os.path.basename(os.path.dirname(traj)),
        'matches': {k: v[0] == v[1] for k, v in values.items()},
        'values': values,
    }, sort_keys=True))
