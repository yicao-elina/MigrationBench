#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np


def min_distance(pos, cell, species):
    inv = np.linalg.inv(cell)
    frac = (pos @ inv) % 1.0
    n = len(pos)
    best = (float("inf"), None, None)
    for i0 in range(0, n, 100):
        ii = np.arange(i0, min(i0 + 100, n))
        d = frac[ii, None, :] - frac[None, :, :]
        d -= np.rint(d)
        dc = d @ cell
        ds = np.sqrt(np.sum(dc * dc, axis=2))
        ds[:, ii] = np.inf
        k = np.unravel_index(np.argmin(ds), ds.shape)
        if float(ds[k]) < best[0]:
            best = (float(ds[k]), int(ii[k[0]]), int(k[1]))
    return best[0], best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--start-step", type=int, default=200000)
    ap.add_argument("--save-interval", type=int, default=200)
    ap.add_argument("--target-step", type=int, default=489000)
    ap.add_argument("--radius", type=int, default=6)
    args = ap.parse_args()
    target_frame = (args.target_step - args.start_step) // args.save_interval
    wanted = set(range(target_frame - args.radius, target_frame + args.radius + 1))
    previous = None
    previous_frame = None
    with args.trajectory.open() as fh:
        frame = -1
        while True:
            first = fh.readline()
            if not first:
                break
            n = int(first)
            header = fh.readline().rstrip()
            match = re.search(r'Lattice="([^"]+)"', header)
            if not match:
                raise ValueError(f"No Lattice field at frame {frame + 1}")
            cell = np.array([float(x) for x in match.group(1).split()]).reshape(3, 3)
            species = []
            pos = np.empty((n, 3))
            extra = []
            for i in range(n):
                fields = fh.readline().split()
                species.append(fields[0])
                pos[i] = [float(x) for x in fields[1:4]]
                extra.append(fields[4:])
            frame += 1
            if frame in wanted:
                out = {
                    "frame": frame,
                    "step": args.start_step + frame * args.save_interval,
                    "header": header,
                    "natoms": n,
                    "pos_min": pos.min(axis=0).tolist(),
                    "pos_max": pos.max(axis=0).tolist(),
                    "com": pos.mean(axis=0).tolist(),
                    "extra_field_count": sorted(set(map(len, extra))),
                }
                momenta = np.array([[float(x) for x in row[:3]] for row in extra if len(row) >= 3])
                if len(momenta) == n:
                    norms = np.linalg.norm(momenta, axis=1)
                    out.update({"momentum_norm_max": float(norms.max()), "momentum_norm_mean": float(norms.mean())})
                forces = np.array([[float(x) for x in row[3:6]] for row in extra if len(row) >= 6])
                if len(forces) == n:
                    norms = np.linalg.norm(forces, axis=1)
                    out.update({"force_norm_max": float(norms.max()), "force_norm_mean": float(norms.mean())})
                if previous is not None and previous_frame == frame - 1:
                    inv = np.linalg.inv(cell)
                    dr = pos - previous
                    df = dr @ inv
                    df_mi = df - np.rint(df)
                    dr_mi = df_mi @ cell
                    raw_norm = np.linalg.norm(dr, axis=1)
                    pbc_norm = np.linalg.norm(dr_mi, axis=1)
                    translation = np.median(dr_mi, axis=0)
                    residual = dr_mi - translation
                    out.update({
                        "raw_step_max_A": float(raw_norm.max()),
                        "raw_step_rms_A": float(np.sqrt(np.mean(raw_norm ** 2))),
                        "pbc_step_max_A": float(pbc_norm.max()),
                        "pbc_step_rms_A": float(np.sqrt(np.mean(pbc_norm ** 2))),
                        "pbc_translation_median_A": translation.tolist(),
                        "translation_residual_rms_A": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
                        "pbc_atoms_over_5A": int(np.sum(pbc_norm > 5.0)),
                        "raw_atoms_over_5A": int(np.sum(raw_norm > 5.0)),
                        "largest_pbc_displacements": [
                            {"index": int(i), "species": species[i], "disp_A": float(pbc_norm[i])}
                            for i in np.argsort(pbc_norm)[-10:][::-1]
                        ],
                    })
                md, i, j = min_distance(pos, cell, species)
                out.update({"minimum_distance_A": md, "minimum_pair_indices": [i, j], "minimum_pair_species": [species[i], species[j]]})
                print(json.dumps(out))
            previous = pos
            previous_frame = frame


if __name__ == "__main__":
    main()
